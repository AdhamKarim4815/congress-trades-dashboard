"""The copy-trading dashboard.

Reads and writes the same modules the CLI uses -- there is no web-only code
path for any action. Long-running work (syncing filings, fetching prices,
scoring members, reading news) is dispatched to a worker thread through
`trade_bot.activity.run_job`, and the page streams progress back over
server-sent events so the status bar always shows what is happening right
now rather than a spinner.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trade_bot.activity import ACTIVITY, run_job
from trade_bot.committees import ConflictIndex, load_committee_edge
from trade_bot.config import Settings, get_settings
from trade_bot.database import Store
from trade_bot.digest import load_digest
from trade_bot.domain import TransactionType
from trade_bot.follow import build_portfolio, create_watchlist, mirror_watchlist
from trade_bot.ledger import accumulation_summary, build_positions
from trade_bot.market import market_status
from trade_bot.news import company_names
from trade_bot.performance import evaluate_trades, summarize
from trade_bot.pipeline import STAGES, run_refresh
from trade_bot.politics import mention_summary
from trade_bot.prices import PriceRepository
from trade_bot.recommend import load_recommendations
from trade_bot.secrets import delete_api_key, get_api_key, set_api_key
from trade_bot.validate import MAX_REPORTING_LAG_DAYS, load_report

PACKAGE_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


def _money(value: float | None) -> str:
    # The sign goes outside the currency mark. "$-8.0k" reads as a typo; the
    # ledger shows plenty of negative net exposure now that sales are tracked,
    # so this stopped being hypothetical.
    if value is None:
        return "—"
    sign = "-" if value < 0 else ""
    size = abs(value)
    if size >= 1_000_000:
        return f"{sign}${size / 1_000_000:,.1f}M"
    if size >= 1_000:
        return f"{sign}${size / 1_000:,.1f}k"
    return f"{sign}${size:,.2f}"


def _percent(value: float | None, digits: int = 1) -> str:
    """A return, always signed: the direction is the point."""
    if value is None:
        return "—"
    return f"{value:+.{digits}%}"


def _rate(value: float | None, digits: int = 0) -> str:
    """A proportion, never signed.

    A win rate is not a gain. "+64% of the time" reads as an increase in
    something, and the sign that makes a return legible makes a rate wrong.
    """
    if value is None:
        return "—"
    return f"{value:.{digits}%}"


def _ago(value: Any) -> str:
    """Humanise a timestamp, accepting datetimes, dates and ISO strings.

    Values reach templates from three places -- SQLAlchemy rows (datetime),
    disclosure columns (date) and JSON state blobs (ISO string) -- so the
    filter normalises rather than assuming one shape.
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return str(value)
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - moment
    elif isinstance(value, date):
        delta = datetime.now(UTC).date() - value
    else:
        return str(value)
    if not isinstance(delta, timedelta):
        return str(value)
    seconds = delta.total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    days = int(seconds // 86400)
    return f"{days}d ago" if days < 30 else f"{days // 30}mo ago"


templates.env.filters["money"] = _money
templates.env.filters["pct"] = _percent
templates.env.filters["rate"] = _rate
def _short_date(value: Any) -> str:
    """`12 Jun 26` — dense enough that a table column never wraps mid-date."""
    if value is None:
        return "—"
    if isinstance(value, str):
        try:
            parsed: date = datetime.fromisoformat(value).date()
        except ValueError:
            return value
        value = parsed
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return f"{value.day} {value.strftime('%b')} {value.strftime('%y')}"
    return str(value)


templates.env.filters["ago"] = _ago
def _bracket(low: Any, high: Any = None) -> str:
    """Render a disclosed amount as the range it actually is.

    Congress discloses brackets, never amounts. Showing the midpoint as
    "$8.0k" implies a precision the filing does not contain -- and since 72%
    of all disclosures fall in the single $1,001-$15,000 band, that one
    number would otherwise appear over and over as though it were measured.
    """
    try:
        low_value = float(low or 0)
        high_value = float(high or 0)
    except (TypeError, ValueError):
        return "—"
    if not low_value and not high_value:
        return "—"
    if not high_value or abs(high_value - low_value) < 1:
        return _money(low_value)
    return f"{_money(low_value)}\u2013{_money(high_value)}"


templates.env.filters["shortdate"] = _short_date
templates.env.filters["bracket"] = _bracket


# The lettered amount bands printed down the right of every House PTR. The
# filing checks one box; the stored low/high is that box's range, so the letter
# is recoverable and is the most faithful way to show a size.
AMOUNT_BANDS: tuple[tuple[str, float, str], ...] = (
    ("A", 1_001, "$1,001 \u2013 $15,000"),
    ("B", 15_001, "$15,001 \u2013 $50,000"),
    ("C", 50_001, "$50,001 \u2013 $100,000"),
    ("D", 100_001, "$100,001 \u2013 $250,000"),
    ("E", 250_001, "$250,001 \u2013 $500,000"),
    ("F", 500_001, "$500,001 \u2013 $1,000,000"),
    ("G", 1_000_001, "$1,000,001 \u2013 $5,000,000"),
    ("H", 5_000_001, "$5,000,001 \u2013 $25,000,000"),
    ("I", 25_000_001, "$25,000,001 \u2013 $50,000,000"),
    ("J", 50_000_001, "over $50,000,000"),
)


def _band_letter(low: float | None) -> str:
    """The letter checked on the form for this amount, or a dash."""
    index = _band(low)
    return AMOUNT_BANDS[index][0] if index >= 0 else "—"


def _band(low: float | None) -> int:
    """Index of the bracket a filed amount falls in, or -1 if unknown."""
    if low is None:
        return -1
    last = -1
    for index, (_letter, floor, _range) in enumerate(AMOUNT_BANDS):
        if low >= floor:
            last = index
    return last


templates.env.filters["band"] = _band
templates.env.filters["band_letter"] = _band_letter
templates.env.globals["AMOUNT_BANDS"] = AMOUNT_BANDS
# Flipped by `trade_bot.export` so the published snapshot drops the live
# status bar and the action buttons that need a server behind them.
templates.env.globals["static_export"] = False
templates.env.globals["exported_at"] = None
templates.env.globals["static_market"] = None
# The line past which a reporting lag is a misread date rather than a late
# filing. Templates flag anything beyond it rather than printing it straight.
templates.env.globals["MAX_LAG_DAYS"] = MAX_REPORTING_LAG_DAYS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    app.state.settings = settings
    app.state.store = store
    app.state.prices = PriceRepository(settings)
    app.state.tasks = {}
    ACTIVITY.emit("app", "dashboard ready", level="success")
    yield
    for task in list(app.state.tasks.values()):
        if not task.done():
            task.cancel()


def create_app() -> FastAPI:
    application = FastAPI(title="Congress Copy-Trading", lifespan=lifespan)
    static_dir = PACKAGE_ROOT / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # `FastAPI.state` is deliberately untyped, so narrow once here rather
    # than sprinkling casts through every route.
    def _settings() -> Settings:
        return cast(Settings, application.state.settings)

    def _store() -> Store:
        return cast(Store, application.state.store)

    def _prices() -> PriceRepository:
        return cast(PriceRepository, application.state.prices)

    def _launch(name: str, channel: str, work: Any) -> bool:
        """Start a tracked background job unless one of that name is running."""
        if ACTIVITY.is_running(name):
            return False

        async def _run() -> None:
            await asyncio.to_thread(run_job, ACTIVITY, name, channel, work)

        application.state.tasks[name] = asyncio.create_task(_run())
        return True

    # -- status / activity -------------------------------------------------

    @application.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "env": _settings().env}

    @application.get("/api/status")
    def status() -> dict[str, object]:
        store = _store()
        return {
            "activity": ACTIVITY.status(),
            "market": market_status(),
            "jobs": [job.as_dict() for job in ACTIVITY.jobs()[:6]],
            "counts": {
                **store.trade_counts(),
                "filings": store.filing_counts(),
                "news": store.news_counts(),
            },
            "last_event_id": ACTIVITY.last_id(),
        }

    @application.get("/api/activity")
    def activity(since: int = Query(0)) -> dict[str, object]:
        events = ACTIVITY.since(since)
        return {
            "events": [event.as_dict() for event in events],
            "last_id": events[-1].id if events else since,
            "status": ACTIVITY.status(),
        }

    @application.get("/api/activity/stream")
    async def activity_stream(request: Request, since: int = Query(0)) -> StreamingResponse:
        """Server-sent events feeding the status bar.

        Polls the activity ring buffer rather than holding a queue per client,
        so a reconnecting browser simply resumes from the last id it saw and
        no events are lost or duplicated in the handover.
        """

        async def _events() -> AsyncIterator[str]:
            cursor = since
            while True:
                if await request.is_disconnected():
                    break
                new_events = ACTIVITY.since(cursor)
                if new_events:
                    cursor = new_events[-1].id
                payload = {
                    "events": [event.as_dict() for event in new_events],
                    "status": ACTIVITY.status(),
                    "last_id": cursor,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- actions -----------------------------------------------------------

    @application.post("/actions/refresh")
    async def refresh(
        stages: list[str] = Form([]),
        filing_limit: int = Form(0),
    ) -> RedirectResponse:
        settings, store = _settings(), _store()
        selected = [stage for stage in stages if stage in STAGES] or None
        started = _launch(
            "refresh",
            "sync",
            lambda reporter: run_refresh(
                settings,
                store,
                reporter,
                stages=selected,
                filing_limit=filing_limit or None,
            ),
        )
        if not started:
            ACTIVITY.emit("sync", "a refresh is already running", level="warn")
        return RedirectResponse(url="/", status_code=303)

    # -- opportunity board -------------------------------------------------

    @application.get("/", response_class=HTMLResponse)
    def board(request: Request) -> HTMLResponse:
        store, settings = _store(), _settings()
        items, generated_at = load_recommendations(store)
        counts = store.trade_counts()
        filings = store.filing_counts()
        sync_state = store.get_state("sync.disclosures") or {}
        return templates.TemplateResponse(
            request,
            "board.html",
            {
                "recommendations": items,
                "generated_at": generated_at,
                "counts": counts,
                "filings": filings,
                "coverage": _prices().coverage(),
                "sync_sources": (sync_state.get("sources") or {}),
                "latest_disclosure": store.latest_disclosure_date(),
                "backtest": store.get_state("board.backtest"),
                "world_news": store.list_news(category="world", limit=8),
                "business_news": store.list_news(category="business", limit=8),
                "stages": STAGES,
                "settings": settings,
            },
        )

    # -- evidence ----------------------------------------------------------

    @application.get("/evidence", response_class=HTMLResponse)
    def evidence(request: Request) -> HTMLResponse:
        """The board's own scorecard.

        Kept as a first-class page rather than a footnote: a copy-trading
        dashboard that cannot say whether copy-trading worked is asking for
        trust it has not earned.
        """
        store, settings = _store(), _settings()
        return templates.TemplateResponse(
            request,
            "evidence.html",
            {
                "backtest": store.get_state("board.backtest"),
                "bands": _band_verdicts(store.get_state("board.backtest")),
                "committee_edge": load_committee_edge(store),
                "validation": load_report(store),
                "benchmark": settings.prices.benchmark_symbol,
                "settings": settings,
            },
        )

    # -- members -----------------------------------------------------------

    def _band_verdicts(backtest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        """Turn each measured rank band into a sentence the data supports.

        The prose used to be written into the template: "a coin flip", "still
        lost to holding the index", "the rank at which the measurement changes
        sign". All three were true of one measurement. Correcting the backtest
        to rank on purchases changed the numbers and two of the three became
        false while the page went on asserting them — a hard-coded conclusion
        is a claim that cannot notice it has stopped being right.

        So the verdict is derived. If the bands ever invert again, the page
        says so on its own.
        """
        bands: dict[str, dict[str, Any]] = {}
        for row in (backtest or {}).get("by_rank") or []:
            mean, rate = row.get("mean_excess", 0.0), row.get("beat_rate", 0.0)
            if mean > 0.01 and rate > 0.52:
                verdict = "beat the index"
            elif mean < -0.01 and rate < 0.48:
                verdict = "lost to the index"
            else:
                verdict = "came out level with the index"
            bands[row["rank"]] = {**row, "verdict": verdict}
        # Is the ranking monotone? The page may not claim a gradient it does
        # not have, and with the corrected ranking it does not have one.
        order = [bands[key]["mean_excess"] for key in ("1-3", "4-6", "7+") if key in bands]
        best_is_top = bool(order) and order[0] == max(order)
        return {
            **bands,
            "_shape": {
                "monotone": order == sorted(order, reverse=True),
                "best_is_top": best_is_top,
            },
        }

    @application.get("/members", response_class=HTMLResponse)
    def members(request: Request, sort: str = Query("buy")) -> HTMLResponse:
        """Who to copy — which is a different question for each direction.

        The default is the *buy* record, because that is the one the backtest
        measures and the one "copy this member" usually means. Ranking on the
        combined score put a member with twenty-five sales and no purchases at
        all second on a page headed by a buys-only result.
        """
        store, settings = _store(), _settings()
        horizon = settings.ranking.primary_horizon_days
        by_side = {
            side: {
                score["politician_id"]: score
                for score in store.list_member_scores(horizon_days=horizon, side=side)
            }
            for side in ("all", "buy", "sell")
        }
        # `trades` sorts by activity and is not a ranking; everything else
        # ranks on the side it names.
        ranking_side = sort if sort in ("buy", "sell", "all") else "buy"
        scores = by_side[ranking_side]

        rows = []
        for member in store.list_politicians(with_trades_only=True):
            politician_id = member["politician_id"]
            rows.append(
                {
                    **member,
                    "score": scores.get(politician_id),
                    "buy": by_side["buy"].get(politician_id),
                    "sell": by_side["sell"].get(politician_id),
                    "overall": by_side["all"].get(politician_id),
                }
            )
        if sort == "trades":
            rows.sort(
                key=lambda row: (row["overall"] or {}).get("trades_scored", 0), reverse=True
            )
        else:
            rows.sort(
                key=lambda row: (
                    (row["score"] or {}).get("rankable", False),
                    (row["score"] or {}).get("score", -99),
                ),
                reverse=True,
            )

        # Rank is assigned here rather than counted in the template, because
        # the page's whole job is to say where the line is and a template
        # counter cannot be trusted with that.
        rank = 0
        best = 0.0
        for row in rows:
            score = row["score"]
            if score and score["rankable"]:
                rank += 1
                row["rank"] = rank
                best = max(best, abs(score["score"]))
        for row in rows:
            score = row["score"]
            # Relative standing, drawn rather than left to be inferred from a
            # column of similar-looking percentages.
            row["bar"] = round(100 * abs(score["score"]) / best) if score and best else 0
            # A score is shrunk toward zero by sample size, so a member barely
            # past the minimum has a number that is still mostly prior. Saying
            # which ones those are is the difference between a ranking and a
            # recommendation.
            row["thin"] = bool(
                score
                and score["rankable"]
                and score["trades_scored"] < settings.ranking.shrinkage_prior_trades * 2
            )
            # A record with nothing on the other side is not a criticism, but
            # it changes what following them means and the page has to say so.
            row["one_sided"] = (
                "sell" if row["buy"] is None else "buy" if row["sell"] is None else None
            )
        return templates.TemplateResponse(
            request,
            "members.html",
            {
                "members": rows,
                "podium": [row for row in rows if row.get("rank", 99) <= 3],
                "rankable_count": rank,
                "horizon": horizon,
                "sort": sort,
                "ranking_side": ranking_side,
                "backtest": store.get_state("board.backtest"),
                "bands": _band_verdicts(store.get_state("board.backtest")),
                "settings": settings,
            },
        )

    @application.get("/members/{politician_id}", response_class=HTMLResponse)
    def member_detail(request: Request, politician_id: str) -> HTMLResponse:
        store, settings, prices = _store(), _settings(), _prices()
        member = store.get_politician(politician_id)
        if member is None:
            return HTMLResponse("<h1>Unknown member</h1>", status_code=404)

        trades = store.list_trades(politician_id=politician_id, limit=400)
        actionable = [trade for trade in trades if trade["symbol"]]
        performances = evaluate_trades(
            actionable,
            prices,
            settings.prices.benchmark_symbol,
            horizon_days=settings.ranking.primary_horizon_days,
        )
        by_trade = {item.trade_id: item for item in performances}
        # Every horizon, and every side of each. A member good at buying and
        # poor at selling has one combined number that describes neither, so
        # the page shows the split rather than the average of the two.
        scores = [
            {
                "horizon_days": horizon,
                **store.member_score_sides(politician_id, horizon),
            }
            for horizon in sorted(set(settings.ranking.horizons_days))
        ]
        scores = [row for row in scores if row.get("all")]
        # Running positions rather than a flat feed: "still building NVDA"
        # is a different fact from "traded NVDA six times".
        positions = build_positions(actionable, {politician_id: member}, prices)
        conflicts = ConflictIndex(settings, store)
        return templates.TemplateResponse(
            request,
            "member.html",
            {
                "member": member,
                "trades": trades,
                "performance": by_trade,
                "summary": summarize(performances),
                "scores": [score for score in scores if score],
                "positions": [
                    {
                        **position.as_dict(),
                        "committees": conflicts.overlaps(politician_id, position.symbol),
                        "industry": conflicts.industry(position.symbol),
                    }
                    for position in positions[:40]
                ],
                "seats": store.list_committee_memberships(politician_id=politician_id),
                "news": store.list_news(category="member", subject=politician_id, limit=15),
                "horizon": settings.ranking.primary_horizon_days,
                "settings": settings,
            },
        )

    # -- trades ------------------------------------------------------------

    @application.get("/trades", response_class=HTMLResponse)
    def trades(
        request: Request,
        symbol: str = Query(""),
        chamber: str = Query(""),
        kind: str = Query(""),
        actionable: int = Query(1),
    ) -> HTMLResponse:
        store = _store()
        rows = store.list_trades(
            symbol=symbol or None,
            chamber=chamber or None,
            transaction_type=kind or None,
            actionable_only=bool(actionable),
            limit=400,
        )
        members = {
            member["politician_id"]: member for member in store.list_politicians()
        }
        return templates.TemplateResponse(
            request,
            "trades.html",
            {
                "trades": rows,
                "members": members,
                "symbol": symbol,
                "chamber": chamber,
                "kind": kind,
                "actionable": actionable,
                "types": [item.value for item in TransactionType],
                "settings": _settings(),
            },
        )

    # -- ticker ------------------------------------------------------------

    @application.get("/tickers/{symbol}", response_class=HTMLResponse)
    def ticker_detail(request: Request, symbol: str) -> HTMLResponse:
        store, settings, prices = _store(), _settings(), _prices()
        upper = symbol.upper()
        rows = store.list_trades(symbol=upper, actionable_only=True, limit=200)
        members = {
            member["politician_id"]: member for member in store.list_politicians()
        }
        frame = prices.load(upper)
        history = (
            [
                {"date": row["date"].date().isoformat(), "close": float(row["close"])}
                for _, row in frame.tail(180).iterrows()
            ]
            if not frame.empty
            else []
        )
        last_price, last_date = prices.latest_close(upper)

        # Markers are built here rather than in the template because a trade
        # date can fall on a weekend and simply not exist in `history`, and the
        # repository already knows how to resolve that.
        session_index = {row["date"]: index for index, row in enumerate(history)}

        def _slot(when: date | None) -> int | None:
            """Index of the session a date belongs to, snapping forward."""
            if when is None or not history:
                return None
            iso = when.isoformat()
            if iso in session_index:
                return session_index[iso]
            _price, resolved = prices.close_on_or_after(upper, when)
            if resolved is not None and resolved.isoformat() in session_index:
                return session_index[resolved.isoformat()]
            # Before the window opens the marker has no position; after it, the
            # last session is the honest place for it.
            first = date.fromisoformat(history[0]["date"])
            return None if when < first else len(history) - 1

        # A heavily traded ticker has dozens of disclosures, and every one
        # drawn at once turns the chart into noise -- overlapping lag brackets
        # and a blotched blackout. The most recent handful is what anyone
        # actually reads; the full list is in the record below, and the caption
        # says how many were left off rather than quietly dropping them.
        MARKER_LIMIT = 10
        plotted = rows[:MARKER_LIMIT]
        markers = []
        for row in plotted:
            traded, filed = _slot(row["transaction_date"]), _slot(row["disclosure_date"])
            if filed is None:
                continue
            markers.append(
                {
                    "name": members.get(row["politician_id"], {}).get(
                        "full_name", row["politician_id"]
                    ),
                    "member": row["politician_id"],
                    "kind": "buy" if "purchase" in row["transaction_type"] else "sell",
                    "traded": traded,
                    "filed": filed,
                    "lag": row["reporting_lag_days"],
                    "traded_on": row["transaction_date"].isoformat(),
                    "filed_on": row["disclosure_date"].isoformat(),
                    # A trade older than the price window has no left edge; say
                    # so rather than drawing it as though it started on screen.
                    "clipped": traded is None,
                    "party": members.get(row["politician_id"], {}).get("party"),
                    "chamber": row["chamber"],
                    "band": _band_letter(row["amount_low"]),
                    "size": _bracket(row["amount_low"], row["amount_high"]),
                    "type": row["transaction_type"].replace("_", " "),
                    "asset": row["asset_name"],
                }
            )

        # One record per member, so a later sale reads against the buy that
        # preceded it instead of being a separate line in a flat list.
        by_member: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = by_member.setdefault(
                row["politician_id"],
                {
                    "politician_id": row["politician_id"],
                    "name": members.get(row["politician_id"], {}).get(
                        "full_name", row["politician_id"]
                    ),
                    "party": members.get(row["politician_id"], {}).get("party"),
                    "trades": [],
                    "buys": 0,
                    "sells": 0,
                },
            )
            entry["trades"].append(row)
            entry["buys" if "purchase" in row["transaction_type"] else "sells"] += 1
        for entry in by_member.values():
            entry["trades"].sort(key=lambda t: t["disclosure_date"], reverse=True)
        records = sorted(
            by_member.values(),
            key=lambda e: e["trades"][0]["disclosure_date"],
            reverse=True,
        )

        # The same disclosures folded into running positions, which is the
        # only view that answers "who is building here and who is leaving".
        positions = build_positions(rows, members, prices, upper)
        by_position = {position.politician_id: position for position in positions}
        conflicts = ConflictIndex(settings, store)
        for entry in records:
            entry["committees"] = conflicts.overlaps(entry["politician_id"], upper)
            held = by_position.get(entry["politician_id"])
            entry["position"] = held.as_dict() if held else None
            # Identical lines from one member are separate lots, not separate
            # decisions -- a member holding one name in two accounts files one
            # line per account. Printing them as two identical rows invites the
            # reader to count two events, so they are collapsed into one row
            # that says how many lots it stands for.
            collapsed: dict[tuple[Any, ...], dict[str, Any]] = {}
            for row in entry["trades"]:
                key = (
                    row["transaction_date"],
                    row["disclosure_date"],
                    row["transaction_type"],
                    row["amount_low"],
                    row["amount_high"],
                )
                existing = collapsed.get(key)
                if existing is None:
                    collapsed[key] = {**row, "lots": 1}
                else:
                    existing["lots"] += 1
            entry["rows"] = list(collapsed.values())

        return templates.TemplateResponse(
            request,
            "ticker.html",
            {
                "symbol": upper,
                "company": company_names(store).get(upper),
                "industry": conflicts.industry(upper),
                "trades": rows,
                "members": members,
                "history": history,
                "markers": markers,
                "markers_shown": len(markers),
                "markers_total": len(rows),
                "records": records,
                "positions": [position.as_dict() for position in positions],
                "accumulation": accumulation_summary(positions),
                "buy_count": sum(1 for r in rows if "purchase" in r["transaction_type"]),
                "sell_count": sum(1 for r in rows if "purchase" not in r["transaction_type"]),
                "last_price": last_price,
                "last_date": last_date,
                "news": store.list_news(category="ticker", subject=upper, limit=15),
                "mentions": store.list_mentions(symbol=upper, limit=15),
                "settings": settings,
            },
        )

    # -- watchlists --------------------------------------------------------

    @application.get("/watchlists", response_class=HTMLResponse)
    def watchlists(request: Request) -> HTMLResponse:
        store, settings, prices = _store(), _settings(), _prices()
        rows = []
        for watchlist in store.list_watchlists():
            try:
                portfolio = build_portfolio(settings, store, prices, watchlist["watchlist_id"])
            except KeyError:
                continue
            rows.append({"watchlist": watchlist, "portfolio": portfolio})
        members = store.list_politicians(with_trades_only=True)
        scores = {
            score["politician_id"]: score
            for score in store.list_member_scores(
                horizon_days=settings.ranking.primary_horizon_days
            )
        }
        members.sort(
            key=lambda member: scores.get(member["politician_id"], {}).get("score", -99),
            reverse=True,
        )
        return templates.TemplateResponse(
            request,
            "watchlists.html",
            {
                "rows": rows,
                "members": members,
                "scores": scores,
                "settings": settings,
            },
        )

    @application.post("/watchlists/create")
    async def watchlist_create(
        name: str = Form(...),
        politician_ids: list[str] = Form([]),
        allocation_mode: str = Form("fixed_dollars"),
        allocation_value: float = Form(1000.0),
        mirror_sells: bool = Form(False),
    ) -> RedirectResponse:
        settings, store, prices = _settings(), _store(), _prices()
        watchlist = create_watchlist(
            settings,
            store,
            name=name,
            politician_ids=politician_ids,
            allocation_mode=allocation_mode,
            allocation_value=allocation_value,
            mirror_sells=mirror_sells,
        )
        _launch(
            f"mirror:{watchlist.watchlist_id}",
            "watchlists",
            lambda reporter: mirror_watchlist(
                settings, store, prices, watchlist.watchlist_id, reporter
            ),
        )
        return RedirectResponse(url="/watchlists", status_code=303)

    @application.post("/watchlists/{watchlist_id}/refresh")
    async def watchlist_refresh(watchlist_id: str) -> RedirectResponse:
        settings, store, prices = _settings(), _store(), _prices()
        _launch(
            f"mirror:{watchlist_id}",
            "watchlists",
            lambda reporter: mirror_watchlist(settings, store, prices, watchlist_id, reporter),
        )
        return RedirectResponse(url="/watchlists", status_code=303)

    @application.post("/watchlists/{watchlist_id}/delete")
    def watchlist_delete(watchlist_id: str) -> RedirectResponse:
        _store().delete_watchlist(watchlist_id)
        return RedirectResponse(url="/watchlists", status_code=303)

    # -- politics and news -------------------------------------------------

    @application.get("/politics", response_class=HTMLResponse)
    def politics(request: Request) -> HTMLResponse:
        store, settings = _store(), _settings()
        return templates.TemplateResponse(
            request,
            "politics.html",
            {
                "figures": settings.politics.figures,
                "mentions": store.list_mentions(limit=80),
                "summary": mention_summary(store, days=settings.politics.lookback_days),
                "figure_news": {
                    figure.slug: store.list_news(
                        category="figure", subject=figure.slug, limit=20
                    )
                    for figure in settings.politics.figures
                },
                "settings": settings,
            },
        )

    @application.get("/news", response_class=HTMLResponse)
    def news(request: Request) -> HTMLResponse:
        store = _store()
        return templates.TemplateResponse(
            request,
            "news.html",
            {
                "world": store.list_news(category="world", limit=40),
                "business": store.list_news(category="business", limit=40),
                "tickers": store.list_news(category="ticker", limit=40),
                "members": store.list_news(category="member", limit=40),
                "settings": _settings(),
            },
        )

    # -- data / settings ---------------------------------------------------

    @application.get("/data", response_class=HTMLResponse)
    def data(request: Request) -> HTMLResponse:
        store, prices = _store(), _prices()
        return templates.TemplateResponse(
            request,
            "data.html",
            {
                "counts": store.trade_counts(),
                "filings": store.filing_counts(),
                "news_counts": store.news_counts(),
                "coverage": prices.coverage(),
                "providers": [provider.name for provider in prices.providers],
                "sync_state": store.get_state("sync.disclosures") or {},
                "last_refresh": store.get_state("sync.last_refresh") or {},
                "validation": load_report(store),
                "symbols_wanted": len(store.distinct_trade_symbols()),
                "failed_filings": [
                    filing
                    for filing in store.list_filings(parsed=False, limit=60)
                    if filing["parse_error"]
                ],
                "stages": STAGES,
                "settings": _settings(),
            },
        )

    @application.get("/digest", response_class=HTMLResponse)
    def digest(request: Request) -> HTMLResponse:
        """What changed since the last run, for someone who did not visit."""
        store = _store()
        return templates.TemplateResponse(
            request,
            "digest.html",
            {"digest": load_digest(store), "settings": _settings()},
        )

    @application.get("/secrets", response_class=HTMLResponse)
    def secrets_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "secrets.html", _secrets_context(request))

    def _secrets_context(request: Request) -> dict[str, object]:
        settings = _settings()
        return {
            "rows": [
                {"label": label, "service": service, "stored": get_api_key(service) is not None}
                for label, service in (
                    (
                        "OpenAI — reads scanned paper filings and briefs top ideas",
                        settings.openai_keyring_name,
                    ),
                    ("Twelve Data (optional price fallback)", settings.twelve_data_keyring_name),
                    ("Quiver Quantitative (optional)", settings.quiver_keyring_name),
                )
            ],
            "settings": settings,
        }

    @application.post("/secrets/set")
    def secrets_set(service: str = Form(...), api_key: str = Form(...)) -> RedirectResponse:
        if api_key.strip():
            set_api_key(service, api_key.strip())
        return RedirectResponse(url="/secrets", status_code=303)

    @application.post("/secrets/delete")
    def secrets_delete(service: str = Form(...)) -> RedirectResponse:
        delete_api_key(service)
        return RedirectResponse(url="/secrets", status_code=303)

    @application.get("/api/board")
    def api_board() -> JSONResponse:
        items, generated_at = load_recommendations(_store())
        return JSONResponse({"generated_at": generated_at, "items": items})

    return application


app = create_app()
