"""Which committee a member sits on, and whether it covers what they bought.

This is the one signal in congressional disclosure that no general stock tool
can produce. A House Energy and Commerce member buying a pipeline operator
three weeks before a markup is a categorically different event from the same
purchase by someone on Agriculture, and until now the board treated them
identically.

Two free, keyless sources make it possible:

* `unitedstates/congress-legislators` publishes current committee rosters
  keyed on bioguide id -- the same id `sync_legislators` already attaches to
  members, which is what makes the join possible at all;
* the SEC's own submissions API reports each registrant's SIC code, so a
  ticker can be placed in an industry without a data vendor.

**The jurisdiction map is editorial and deliberately narrow.** There is no
authoritative machine-readable mapping from a committee to the industries it
regulates, so `JURISDICTIONS` below is a judgement call. It is written to be
wrong in one direction only: a committee whose remit is broad or diffuse --
Ways and Means, Judiciary, Rules -- is left out entirely rather than matched
to everything. Claiming a conflict that does not exist would discredit the
ones that do.

Nothing here asserts wrongdoing. A member of a committee may buy a company it
oversees for entirely ordinary reasons, and most do. What the overlap says is
that this particular disclosure is worth reading more carefully than the
average one, which is exactly what the board is for.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import SyncReport

logger = logging.getLogger(__name__)

COMMITTEES_URL = "https://unitedstates.github.io/congress-legislators/committees-current.json"
MEMBERSHIP_URL = (
    "https://unitedstates.github.io/congress-legislators/committee-membership-current.json"
)
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SIC prefixes, matched against the leading digits of a registrant's code.
#
# Only committees with a clear, nameable market remit appear. The omissions
# are as considered as the entries: Ways and Means touches every taxpayer,
# Judiciary reaches every merger, and a rule that matches everything reports
# a conflict on every trade and therefore says nothing.
JURISDICTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    # -- House ------------------------------------------------------------
    "HSAG": ("agriculture and food", ("01", "02", "07", "08", "09", "20", "2870", "5140")),
    "HSAS": ("defence contracting", ("372", "376", "348", "3812", "3761")),
    "HSBA": ("banks, insurers and asset managers", ("60", "61", "62", "63", "64", "67")),
    "HSIF": (
        "energy, health and communications",
        ("13", "29", "46", "48", "49", "283", "384", "80"),
    ),
    "HSII": ("mining, drilling and public land", ("10", "12", "13", "14", "08")),
    "HSPW": ("transport and construction", ("15", "16", "17", "40", "41", "42", "44", "45", "47")),
    "HSSY": ("aerospace and computing", ("376", "357", "367", "737")),
    "HSHM": ("security services", ("7382", "3669")),
    "HSVR": ("veterans' health provision", ("80",)),
    # -- Senate -----------------------------------------------------------
    "SSAF": ("agriculture and food", ("01", "02", "07", "08", "09", "20", "2870", "5140")),
    "SSAS": ("defence contracting", ("372", "376", "348", "3812", "3761")),
    "SSBK": ("banks, insurers and asset managers", ("60", "61", "62", "63", "64", "65", "67")),
    "SSCM": ("transport, telecoms and aerospace", ("37", "40", "41", "42", "44", "45", "47", "48")),
    "SSEG": ("energy and natural resources", ("10", "12", "13", "14", "29", "49")),
    "SSEV": ("utilities, chemicals and construction", ("15", "16", "17", "28", "49", "4950")),
    "SSHR": ("pharmaceuticals and health services", ("283", "384", "80", "82")),
    "SSFI": ("insurers and health financing", ("63", "64", "80")),
}


def _reference(settings: Settings, name: str) -> Path:
    directory = settings.project_root / "var" / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


# -- industry classification ------------------------------------------------


def load_sic_cache(settings: Settings) -> dict[str, dict[str, Any]]:
    path = _reference(settings, "sec_sic.json")
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except Exception:
        path.unlink(missing_ok=True)
        return {}


def save_sic_cache(settings: Settings, cache: dict[str, dict[str, Any]]) -> None:
    _reference(settings, "sec_sic.json").write_text(json.dumps(cache, indent=0, sort_keys=True))


def sync_industries(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    limit: int = 400,
) -> dict[str, Any]:
    """Look up the SIC code for tickers that do not have one cached yet.

    One request per company against the SEC's submissions API, which is free
    and needs no key but does ask for a contact in the User-Agent and a rate
    under ten a second. The result never changes in practice, so it is cached
    on disk permanently and a run only pays for symbols it has not seen.

    `limit` bounds a single run the way the scanned-filing budget does: a
    first pass over two thousand tickers is a few minutes of polite requests,
    and the remainder is picked up next time rather than all at once.
    """
    reporter = reporter or NullReporter()
    cache = load_sic_cache(settings)
    symbols = [symbol for symbol in store.distinct_trade_symbols() if symbol not in cache]
    if not symbols:
        return {"looked_up": 0, "cached": len(cache), "pending": 0}

    from trade_bot.listings import fetch as fetch_registry

    try:
        registry = fetch_registry(settings)
    except Exception as exc:
        reporter.note(f"SEC ticker registry unavailable: {exc}", level="warn")
        return {"looked_up": 0, "cached": len(cache), "pending": len(symbols), "error": str(exc)}

    by_ticker = {
        str(row["ticker"]).upper(): int(row["cik_str"])
        for row in registry.values()
        if isinstance(row, dict) and row.get("ticker") and row.get("cik_str") is not None
    }

    queue = [symbol for symbol in symbols if symbol in by_ticker][:limit]
    unlisted = [symbol for symbol in symbols if symbol not in by_ticker]
    looked_up = failed = 0
    headers = {"User-Agent": settings.disclosures.user_agent, "Accept": "application/json"}
    timeout = settings.disclosures.request_timeout_seconds
    with httpx.Client(timeout=timeout, headers=headers) as client:
        for index, symbol in enumerate(queue, start=1):
            if index % 50 == 0:
                reporter.step(f"industry {index}/{len(queue)}: {symbol}", index, len(queue))
            try:
                response = client.get(SEC_SUBMISSIONS_URL.format(cik=by_ticker[symbol]))
                response.raise_for_status()
                payload = response.json()
            except Exception:
                failed += 1
                continue
            cache[symbol] = {
                "sic": str(payload.get("sic") or ""),
                "industry": payload.get("sicDescription") or "",
                "name": payload.get("name") or "",
            }
            looked_up += 1

    # Symbols the registry does not carry are recorded as looked-at-and-absent,
    # so a run does not re-ask about them for ever.
    for symbol in unlisted:
        cache.setdefault(symbol, {"sic": "", "industry": "", "name": ""})
    save_sic_cache(settings, cache)
    reporter.note(f"industries: {looked_up} looked up, {failed} unavailable")
    return {
        "looked_up": looked_up,
        "failed": failed,
        "cached": len(cache),
        "pending": max(0, len(symbols) - len(queue) - len(unlisted)),
    }


# -- committee rosters ------------------------------------------------------


def sync_committees(
    settings: Settings, store: Store, reporter: JobReporter | None = None
) -> SyncReport:
    """Attach current committee membership to members already seen filing.

    Joined on `bioguide_id`, which `sync_legislators` supplies. A member with
    no bioguide id -- someone who left office, or whose filing name does not
    match the roster -- simply gets no committees, and `validate` already
    names them.
    """
    reporter = reporter or NullReporter()
    started = datetime.now(UTC)
    reporter.step("fetching committee rosters")
    try:
        with httpx.Client(
            timeout=settings.disclosures.request_timeout_seconds,
            headers={"User-Agent": settings.disclosures.user_agent},
            follow_redirects=True,
        ) as client:
            committees = client.get(COMMITTEES_URL)
            committees.raise_for_status()
            membership = client.get(MEMBERSHIP_URL)
            membership.raise_for_status()
    except Exception as exc:
        return SyncReport(
            source="congress-committees",
            ok=False,
            error=f"could not fetch committee rosters: {exc}",
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    names: dict[str, dict[str, str]] = {}
    for entry in committees.json():
        thomas = entry.get("thomas_id")
        if not thomas:
            continue
        names[thomas] = {"name": entry.get("name", thomas), "chamber": entry.get("type", "")}
        for sub in entry.get("subcommittees") or []:
            sub_id = f"{thomas}{sub.get('thomas_id', '')}"
            names[sub_id] = {
                "name": f"{entry.get('name', thomas)} — {sub.get('name', '')}",
                "chamber": entry.get("type", ""),
            }

    by_bioguide: dict[str, str] = {
        member["bioguide_id"]: member["politician_id"]
        for member in store.list_politicians()
        if member.get("bioguide_id")
    }

    rows: list[dict[str, Any]] = []
    for committee_id, members in membership.json().items():
        meta = names.get(committee_id)
        if meta is None:
            continue
        for member in members:
            politician_id = by_bioguide.get(member.get("bioguide", ""))
            if politician_id is None:
                continue
            label, _prefixes = JURISDICTIONS.get(committee_id, ("", ()))
            rows.append(
                {
                    "politician_id": politician_id,
                    "committee_id": committee_id,
                    "committee_name": meta["name"],
                    "chamber": meta["chamber"],
                    "title": member.get("title") or "",
                    "rank": int(member.get("rank") or 0),
                    "jurisdiction": label,
                }
            )

    stored = store.replace_committee_memberships(rows)
    covered = len({row["politician_id"] for row in rows})
    reporter.note(f"committees: {stored} seats across {covered} members")
    return SyncReport(
        source="congress-committees",
        ok=True,
        details={
            "committees": stored,
            "members": covered,
            "unmatched_members": len(by_bioguide) - covered,
        },
        started_at=started,
        finished_at=datetime.now(UTC),
    )


# -- the signal -------------------------------------------------------------


def jurisdiction_for(sic: str) -> set[str]:
    """Committee ids whose remit covers a registrant with this SIC code."""
    if not sic:
        return set()
    return {
        committee
        for committee, (_label, prefixes) in JURISDICTIONS.items()
        if any(sic.startswith(prefix) for prefix in prefixes)
    }


class ConflictIndex:
    """Answers "does this member's committee cover this company?".

    Built once per board rebuild and handed around, because the alternative is
    a database round-trip per buyer per ticker.
    """

    def __init__(self, settings: Settings, store: Store) -> None:
        self.industries = load_sic_cache(settings)
        self.by_member: dict[str, list[dict[str, Any]]] = {}
        for seat in store.list_committee_memberships():
            self.by_member.setdefault(seat["politician_id"], []).append(seat)

    def industry(self, symbol: str) -> str:
        return str((self.industries.get(symbol.upper()) or {}).get("industry", ""))

    def sic(self, symbol: str) -> str:
        return str((self.industries.get(symbol.upper()) or {}).get("sic", ""))

    def overlaps(self, politician_id: str, symbol: str) -> list[dict[str, Any]]:
        """Every seat this member holds whose jurisdiction covers this company.

        Subcommittees are excluded from matching -- only the parent committee
        carries a mapped remit -- so a member is never credited with the same
        conflict twice for sitting on a committee and one of its panels.
        """
        covering = jurisdiction_for(self.sic(symbol))
        if not covering:
            return []
        return [
            seat
            for seat in self.by_member.get(politician_id, [])
            if seat["committee_id"] in covering
        ]

    def strength(self, politician_id: str, symbol: str) -> float:
        """How much weight an overlap deserves, from 0 to 1.

        A chair or ranking member sets the agenda; a junior member of the same
        committee sees the same briefings but decides much less, so the two
        are not the same evidence and are not scored as one.
        """
        seats = self.overlaps(politician_id, symbol)
        if not seats:
            return 0.0
        leading = any(
            "chair" in (seat["title"] or "").lower()
            or "ranking" in (seat["title"] or "").lower()
            for seat in seats
        )
        return 1.0 if leading else 0.6

    def any_seat(self, politician_id: str) -> list[dict[str, Any]]:
        return self.by_member.get(politician_id, [])

    def describe(self, politician_id: str, symbol: str, member_name: str) -> str | None:
        """A sentence for the board's reasons, or nothing to say."""
        seats = self.overlaps(politician_id, symbol)
        if not seats:
            return None
        seat = max(seats, key=lambda item: bool(item["title"]))
        role = seat["title"] or "sits on"
        industry = self.industry(symbol) or "this industry"
        return (
            f"{member_name} {role.lower() if seat['title'] else role} "
            f"{seat['committee_name']}, which covers {seat['jurisdiction'] or industry} — "
            f"and {symbol} is classified as {industry}. That is not evidence of "
            "wrongdoing; it is a reason to read this disclosure more closely."
        )


# -- does the overlap actually predict anything? ----------------------------

EDGE_STATE_KEY = "board.committee_edge"


def measure_committee_edge(
    settings: Settings,
    store: Store,
    repository: Any,
    reporter: JobReporter | None = None,
    horizon_days: int | None = None,
) -> dict[str, Any]:
    """Compare disclosures inside a member's remit against everything else.

    Written before the signal was wired into anything, because "a committee
    member trading what they oversee is the strongest signal in this data" is
    a claim, and this application's standing rule is that a claim gets
    measured rather than asserted.

    Measuring it changed the design. Over three years of disclosures at the
    primary horizon, purchases inside a member's remit did *worse* than
    purchases outside it, while sales inside it did markedly better. So the
    overlap earns weight on the sell side of the board and none on the buy
    side, and the numbers behind that decision are published on `/evidence`
    rather than kept in a commit message.
    """
    from trade_bot.domain import TransactionType
    from trade_bot.performance import evaluate_trades

    reporter = reporter or NullReporter()
    horizon = horizon_days or settings.ranking.primary_horizon_days
    index = ConflictIndex(settings, store)
    if not index.by_member:
        return {"note": "no committee memberships stored yet"}

    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=settings.ranking.lookback_days)
    trades = [
        trade
        for trade in store.list_trades(actionable_only=True, limit=200_000)
        if trade["disclosure_date"] >= cutoff
    ]
    reporter.step(f"measuring the committee overlap over {len(trades)} disclosures")

    sides: dict[str, dict[str, list[float]]] = {
        "buy": {"conflicted": [], "other": []},
        "sell": {"conflicted": [], "other": []},
    }
    performances = {
        item.trade_id: item
        for item in evaluate_trades(
            trades, repository, settings.prices.benchmark_symbol, horizon_days=horizon
        )
    }
    for trade in trades:
        kind = TransactionType(trade["transaction_type"])
        side = "buy" if kind.is_buy else "sell" if kind.is_sell else None
        item = performances.get(trade["trade_id"])
        if side is None or item is None or not item.priced or item.excess_return is None:
            continue
        overlapping = index.overlaps(trade["politician_id"], trade["symbol"])
        bucket = "conflicted" if overlapping else "other"
        sides[side][bucket].append(item.excess_return)

    def _summarise(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        middle = len(ordered) // 2
        return {
            "n": len(ordered),
            "median_excess": (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            ),
            "mean_excess": sum(ordered) / len(ordered),
            "beat_rate": sum(1 for value in ordered if value > 0) / len(ordered),
        }

    result: dict[str, Any] = {
        "horizon_days": horizon,
        "lookback_days": settings.ranking.lookback_days,
        "measured_at": datetime.now(UTC).isoformat(),
        "buy": {key: _summarise(values) for key, values in sides["buy"].items()},
        "sell": {key: _summarise(values) for key, values in sides["sell"].items()},
    }
    for side in ("buy", "sell"):
        conflicted = result[side]["conflicted"]
        other = result[side]["other"]
        result[side]["edge"] = (
            conflicted.get("median_excess", 0.0) - other.get("median_excess", 0.0)
            if conflicted.get("n") and other.get("n")
            else None
        )
    store.set_state(EDGE_STATE_KEY, result)
    reporter.note(
        "committee overlap measured: "
        f"buys {result['buy']['edge']:+.2%}, sells {result['sell']['edge']:+.2%} "
        "against comparable disclosures"
        if result["buy"]["edge"] is not None and result["sell"]["edge"] is not None
        else "committee overlap measured"
    )
    return result


def load_committee_edge(store: Store) -> dict[str, Any] | None:
    return store.get_state(EDGE_STATE_KEY)
