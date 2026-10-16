"""What changed today, for someone who did not visit.

The product is "what is newly public", and until now that required opening a
page and comparing it against a memory of yesterday. The daily run already
knows precisely what is new — it inserted the rows — so the digest costs
nothing but the decision to write it down.

The filter is the point. Roughly a hundred filings arrive on a busy day and
almost none of them matter; a digest that lists all of them is a second feed
to ignore. So it carries only what the board would have ranked anyway:
disclosures from members whose measured record puts them near the top, and
tickers where several members moved the same way at once. Everything else is
counted in one line and left on the site.

Delivery is deliberately plain. The digest is written to `var/reports/` as
Markdown, stored for the dashboard to render, and appended to the GitHub
Actions job summary when one exists — which means the scheduled run already
delivers it to somewhere a person looks, with no mail server, no webhook and
no new credential. A push channel can read the same stored payload later.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import TransactionType
from trade_bot.recommend import load_recommendations

STATE_KEY = "digest.latest"

# The same window the board uses for a cluster, and for the same reason: past
# a quarter apart, two members trading one name are two decisions that
# coincided in the filing queue, not agreement about anything.
CLUSTER_SPAN_DAYS = 90


def _band(low: float, high: float) -> str:
    def money(value: float) -> str:
        if value >= 1_000_000:
            return f"${value / 1_000_000:,.1f}M"
        if value >= 1_000:
            return f"${value / 1_000:,.0f}k"
        return f"${value:,.0f}"

    if not high or abs(high - low) < 1:
        return money(low)
    return f"{money(low)}\u2013{money(high)}"


def build_digest(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    since: date | None = None,
    top_rank: int = 3,
    today: date | None = None,
) -> dict[str, Any]:
    """Collect what became public since `since`, filtered to what matters.

    `since` defaults to the day after the last digest, so a run that happens
    twice in a day does not repeat itself and a missed day is caught up
    rather than skipped.
    """
    reporter = reporter or NullReporter()
    today = today or date.today()
    previous = store.get_state(STATE_KEY) or {}
    if since is None:
        covered = previous.get("covers_to")
        since = (
            date.fromisoformat(covered) + timedelta(days=1)
            if covered
            else today - timedelta(days=1)
        )
    # A stored digest from the future, or a clock that moved backwards, must
    # not produce an empty window that silently hides a day of filings.
    since = min(since, today)

    horizon = settings.ranking.primary_horizon_days
    # Ranked per direction. A purchase is worth reporting when the member is
    # near the top *at buying*; their skill at exiting says nothing about it.
    ranks: dict[str, dict[str, int]] = {}
    for side in ("buy", "sell"):
        ordered = sorted(
            (
                score
                for score in store.list_member_scores(horizon_days=horizon, side=side)
                if score["rankable"]
            ),
            key=lambda score: score["score"],
            reverse=True,
        )
        ranks[side] = {
            score["politician_id"]: position for position, score in enumerate(ordered, start=1)
        }
    members = {member["politician_id"]: member for member in store.list_politicians()}

    fresh = [
        trade
        for trade in store.list_trades(actionable_only=True, disclosed_since=since, limit=20_000)
        if trade["disclosure_date"] <= today
    ]

    headline: list[dict[str, Any]] = []
    for trade in fresh:
        kind = TransactionType(trade["transaction_type"])
        if not (kind.is_buy or kind.is_sell):
            continue
        side = "buy" if kind.is_buy else "sell"
        rank = ranks[side].get(trade["politician_id"])
        if rank is None or rank > top_rank:
            continue
        member = members.get(trade["politician_id"], {})
        headline.append(
            {
                "symbol": trade["symbol"],
                "asset_name": trade["asset_name"][:80],
                "member": member.get("full_name", trade["politician_id"]),
                "politician_id": trade["politician_id"],
                "party": member.get("party"),
                "rank": rank,
                "side": side,
                "size": _band(trade["amount_low"], trade["amount_high"]),
                "transaction_date": trade["transaction_date"].isoformat(),
                "disclosure_date": trade["disclosure_date"].isoformat(),
                "lag_days": trade["reporting_lag_days"],
                "source_url": trade.get("source_url", ""),
            }
        )
    headline.sort(key=lambda row: (row["rank"], row["symbol"] or ""))

    # Several members moving the same way on one name in one window is the
    # signal that does not depend on anyone's track record, so it is reported
    # even when none of the movers is ranked.
    # A cluster is members acting close together in *time*, not members whose
    # filings happened to land in the same window. Grouping on symbol alone
    # reported five members buying MSFT "together" when their transactions
    # were 1,497 days apart -- an artefact of the reporting lag, presented as
    # the one signal that needs nobody's track record.
    by_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in fresh:
        kind = TransactionType(trade["transaction_type"])
        if not (kind.is_buy or kind.is_sell) or not trade["symbol"]:
            continue
        by_side[(trade["symbol"], "buy" if kind.is_buy else "sell")].append(trade)

    clusters: list[dict[str, Any]] = []
    for (symbol, side), rows in by_side.items():
        latest = max(row["transaction_date"] for row in rows)
        together = [
            row
            for row in rows
            if (latest - row["transaction_date"]).days <= CLUSTER_SPAN_DAYS
        ]
        who = {row["politician_id"] for row in together}
        if len(who) < 2:
            continue
        span = (latest - min(row["transaction_date"] for row in together)).days
        clusters.append(
            {
                "symbol": symbol,
                "side": side,
                "span_days": span,
                "members": sorted(
                    members.get(politician_id, {}).get("full_name", politician_id)
                    for politician_id in who
                ),
            }
        )
    clusters.sort(key=lambda row: (len(row["members"]), -row["span_days"]), reverse=True)

    board, _generated = load_recommendations(store)
    top_board = [
        {
            "symbol": item["symbol"],
            "action": item.get("action", "buy"),
            "conviction": item.get("conviction", 0.0),
            "tier": item.get("tier", "unranked"),
        }
        for item in board[:5]
    ]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "covers_from": since.isoformat(),
        "covers_to": today.isoformat(),
        "new_disclosures": len(fresh),
        "top_rank": top_rank,
        "headline": headline,
        "clusters": clusters[:8],
        "board": top_board,
        "quiet": not headline and not clusters,
    }
    store.set_state(STATE_KEY, payload)
    reporter.note(
        f"digest: {len(headline)} from top-{top_rank} members and "
        f"{len(clusters)} clusters out of {len(fresh)} new disclosures"
    )
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    """The digest as plain Markdown, for a file or a job summary."""
    lines = [
        f"# Congressional disclosures — {payload['covers_to']}",
        "",
        (
            f"{payload['new_disclosures']} new tradeable disclosure"
            f"{'' if payload['new_disclosures'] == 1 else 's'} became public "
            f"since {payload['covers_from']}."
        ),
        "",
    ]
    if payload["quiet"]:
        lines += [
            "Nothing from a top-ranked member and no clustered activity. "
            "This is the normal state on most days and is reported rather "
            "than skipped, so a silent digest and a broken one look different.",
            "",
        ]

    if payload["headline"]:
        lines += [f"## From top-{payload['top_rank']} members", ""]
        for row in payload["headline"]:
            lines.append(
                f"- **{row['symbol']}** — {row['side']} {row['size']} by "
                f"{row['member']} (#{row['rank']}), traded "
                f"{row['transaction_date']}, filed {row['disclosure_date']} "
                f"(+{row['lag_days']}d)"
            )
        lines.append("")

    if payload["clusters"]:
        lines += ["## Several members, same direction", ""]
        for row in payload["clusters"]:
            lines.append(
                f"- **{row['symbol']}** — {len(row['members'])} "
                f"{'buyers' if row['side'] == 'buy' else 'sellers'} within "
                f"{row['span_days']} days: {', '.join(row['members'])}"
            )
        lines.append("")

    if payload["board"]:
        lines += ["## Top of the board now", ""]
        for row in payload["board"]:
            lines.append(
                f"- **{row['symbol']}** — {row['action']}, conviction "
                f"{row['conviction'] * 100:.0f}/100 ({row['tier']})"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "Disclosure lags the trade by up to 45 days by law, so this is newly "
        "*public* information, not fresh trading. Sizes are the brackets "
        "Congress discloses, never exact amounts.",
    ]
    return "\n".join(lines)


def publish_digest(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the digest, write it to disk, and deliver it where one can.

    "Where one can" today means the GitHub Actions job summary, which the
    scheduled run already produces and a person already opens. That is a real
    delivery channel that costs nothing and needs no credential; anything
    with a mail server or a push certificate can read the same stored payload.
    """
    reporter = reporter or NullReporter()
    payload = build_digest(settings, store, reporter, **kwargs)
    markdown = render_markdown(payload)

    reports = settings.project_root / "var" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"digest-{payload['covers_to']}.md"
    path.write_text(markdown, encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    delivered = False
    if summary:
        try:
            with Path(summary).open("a", encoding="utf-8") as handle:
                handle.write(markdown + "\n")
            delivered = True
        except OSError as exc:  # a summary file that cannot be written is not fatal
            reporter.note(f"could not write the job summary: {exc}", level="warn")

    return {
        "headline": len(payload["headline"]),
        "clusters": len(payload["clusters"]),
        "new_disclosures": payload["new_disclosures"],
        "path": str(path),
        "delivered_to_job_summary": delivered,
    }


def load_digest(store: Store) -> dict[str, Any] | None:
    return store.get_state(STATE_KEY)
