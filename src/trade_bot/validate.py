"""Whether the data is plausible, not merely whether the code ran.

Every significant defect this project has shipped was found by looking at a
rendered page, never by a test. The suite stayed green while 207 rows carried
a sentence as a company name, every senator had no party, one filing claimed
a 36,535-day reporting lag, and a price series was quietly suspected of being
two different instruments spliced together.

That is not a gap in the tests. Tests check that the code does what it says;
nothing was checking that the *result* was believable. This module is that
missing check. It runs as the last stage of every refresh, reads the stored
data the way a reader would, and records what does not add up.

Three rules shape what belongs here:

* **A check must be able to fire on real data.** Every rule below was written
  against a defect that had actually shipped, not a hypothetical one.
* **A finding is never a fix.** Nothing here rewrites a row. It names the
  rows, links to them, and leaves the decision to a human or to a repair
  pass that is explicit about what it is changing.
* **Silence must be earned.** The senator enrichment failed for all 35
  senators while reporting `ok`, because "the request succeeded" and "the
  data changed" are different claims. `check_inert_sources` exists solely
  because nothing distinguished them.

Findings are graded. `error` means the data is wrong and something downstream
is being computed from it; `warn` means it is suspect and worth an eye;
`info` means it is a fact worth knowing that is not itself a defect.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.disclosures import NOT_TICKERS as PARSER_NOT_TICKERS
from trade_bot.domain import MAX_PLAUSIBLE_LAG_DAYS
from trade_bot.prices import PriceRepository

Severity = str  # "error" | "warn" | "info"

STATE_KEY = "validation.report"

# The same line the parser's repair uses, so "too old to repair" and "too old
# to believe" can never drift apart into two different numbers.
MAX_REPORTING_LAG_DAYS = MAX_PLAUSIBLE_LAG_DAYS

# Ratios a corporate action actually uses. An unadjusted split shows up as a
# close-to-close step at one of these; a real crash almost never lands within
# a couple of percent of one *and* opens exactly there.
SPLIT_RATIOS: tuple[float, ...] = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 1.5, 2.5)
# Both tolerances are tight on purpose. A split is arithmetic: the price is
# divided, so the ex-date open equals the previous close over the ratio to
# within rounding. Loosening this to a few percent starts collecting biotechs
# that halved on trial data, which is a real move and not a defect. Agios
# gapping 45.49 to 23.29 is 2.4% off a clean 2-for-1 and is correctly ignored.
SPLIT_TOLERANCE = 0.01
SPLIT_OPEN_TOLERANCE = 0.01

# Prose, not a noun phrase: lower-case words running into a full stop at the
# end. This is the shape a previous row's free-text description leaves behind
# when it bleeds into the next row's asset name. Real names end in a sentence
# too ("... 5.000% Perpetual Preferred Stock"), so length carries the rest of
# the weight -- a bled description is a paragraph, not a name.
PROSE_RE = re.compile(r"[a-z]{3,}\s+[a-z]{3,}.*\.\s*$")
MIN_PROSE_LENGTH = 120
# The ticker convention on a filing puts the symbol in parentheses at the very
# end of the asset name. Parentheses elsewhere carry something else entirely --
# "(REIT)", "(DE)" for the state of incorporation, "(HC)" for a holding company
# -- and reading those as tickers reports a contradiction on every one.
PAREN_TICKER_RE = re.compile(r"\(([A-Z][A-Z.\-]{0,5})\)\s*$")
# What a ticker can look like: one to six characters, a dot or hyphen for a
# share class. Deliberately the same shape the parser is willing to read.
SYMBOL_RE = re.compile(r"[A-Z][A-Z0-9.\-]{0,5}")
# The same exclusions the parser uses, so a token the parser refuses to read as
# a ticker is not then reported as a contradiction of one. Plus the countries
# and articles that only ever appear in this trailing position on a filing.
NOT_TICKERS = PARSER_NOT_TICKERS | {"FRANCE", "SPAIN", "JAPAN", "CANADA", "CHINA", "BRAZIL"}


class Finding(BaseModel):
    """One thing that does not add up, with enough detail to go and look."""

    code: str
    severity: Severity = "warn"
    title: str
    detail: str
    count: int = 0
    samples: list[dict[str, Any]] = Field(default_factory=list)


def _sample(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return rows[:limit]


# -- filings and trades ----------------------------------------------------


def check_reporting_lag(
    trades: list[dict[str, Any]], max_days: int = MAX_REPORTING_LAG_DAYS
) -> Finding | None:
    """A transaction date implausibly far before the filing that reports it.

    The real example was 36,535 days: a scanned filing whose year was read as
    1925 rather than 2025. `disclosures.repair_transaction_date` fixes the
    ones that are unambiguously a century out; anything still here needs a
    person, because the alternative is inventing a date.
    """
    offenders = [
        {
            "trade_id": trade["trade_id"],
            "politician_id": trade["politician_id"],
            "symbol": trade["symbol"],
            "transaction_date": str(trade["transaction_date"]),
            "disclosure_date": str(trade["disclosure_date"]),
            "lag_days": trade["reporting_lag_days"],
            "source_url": trade.get("source_url", ""),
        }
        for trade in trades
        if trade["reporting_lag_days"] > max_days
    ]
    if not offenders:
        return None
    offenders.sort(key=lambda row: row["lag_days"], reverse=True)
    worst = offenders[0]["lag_days"]
    return Finding(
        code="reporting_lag",
        severity="error" if worst > max_days * 5 else "warn",
        title=f"{len(offenders)} trades report a lag over {max_days} days",
        detail=(
            f"The worst is {worst:,} days. A lag beyond about a year is not late "
            "filing, it is a misread date. The copier's return is unaffected — it "
            "starts at the filing date, which is never in doubt — but the "
            "member's own entry price cannot be priced from a date like this, so "
            "these rows carry no member return and are kept out of the average lag."
        ),
        count=len(offenders),
        samples=_sample(offenders),
    )


def check_asset_names(trades: list[dict[str, Any]]) -> Finding | None:
    """An asset name that reads as a sentence rather than a company."""
    offenders = [
        {
            "trade_id": trade["trade_id"],
            "politician_id": trade["politician_id"],
            "asset_name": trade["asset_name"][:160],
            "source_url": trade.get("source_url", ""),
        }
        for trade in trades
        if trade.get("asset_name")
        and len(trade["asset_name"]) >= MIN_PROSE_LENGTH
        and PROSE_RE.search(trade["asset_name"])
    ]
    if not offenders:
        return None
    return Finding(
        code="prose_asset_name",
        severity="warn",
        title=f"{len(offenders)} asset names read as prose",
        detail=(
            "The PTR form's description column wraps without a label, so a "
            "description can bleed into the next row's asset name. These are the "
            "rows the re-parse could not resolve unambiguously; they are left "
            "rather than guessed at, because a confidently wrong company name "
            "looks right."
        ),
        count=len(offenders),
        samples=_sample(offenders),
    )


def check_symbol_names(trades: list[dict[str, Any]]) -> Finding | None:
    """A stored symbol contradicted by the ticker printed in its own name.

    Electronic filings write the ticker in parentheses after the company:
    ``KEYSIGHT TECHNOLOGIES INC. COMMON STOCK (KEYS)``. When the stored symbol
    and that parenthesised one disagree, one of them is a parser artefact and
    every price, return and board position for the row is against the wrong
    company.
    """
    offenders = []
    for trade in trades:
        symbol, name = trade.get("symbol"), trade.get("asset_name") or ""
        if not symbol:
            continue
        printed = PAREN_TICKER_RE.search(name)
        if printed and printed.group(1).upper() in NOT_TICKERS:
            continue
        if printed and printed.group(1).upper() != str(symbol).upper():
            offenders.append(
                {
                    "trade_id": trade["trade_id"],
                    "stored_symbol": symbol,
                    "printed_symbol": printed.group(1).upper(),
                    "asset_name": name[:120],
                    "source_url": trade.get("source_url", ""),
                }
            )
    if not offenders:
        return None
    return Finding(
        code="symbol_contradiction",
        severity="warn",
        title=f"{len(offenders)} trades store a symbol their own name contradicts",
        detail=(
            "The asset name ends in a parenthesised ticker that is not the "
            "stored symbol, so one of the two came from a neighbouring row. "
            "`trade-bot repair` fixes the unambiguous case — where both appear "
            "in the same name and the stored one is the earlier parenthesis — "
            "and leaves these, because nothing here says which of the two is "
            "the wrong one."
        ),
        count=len(offenders),
        samples=_sample(offenders),
    )


def check_unpriceable_symbols(
    trades: list[dict[str, Any]], repository: PriceRepository
) -> Finding | None:
    """A symbol treated as tradeable that no provider has any history for."""
    cached = set(repository.cached_symbols())
    wanted = Counter(
        str(trade["symbol"]).upper()
        for trade in trades
        if trade.get("symbol") and trade.get("asset_class") in ("stock", "fund")
    )
    missing: list[dict[str, Any]] = [
        {"symbol": symbol, "trades": count}
        for symbol, count in wanted.most_common()
        if symbol not in cached
    ]
    if not missing:
        return None
    return Finding(
        code="unpriceable_symbol",
        severity="warn",
        title=f"{len(missing)} tradeable symbols have no price series",
        detail=(
            "These rows count as actionable disclosures but can never be scored, "
            "mirrored or ranked. Four different problems wear this one label, and "
            "they need different fixes: unsponsored ADRs the free providers do not "
            "carry (ADDYY, ALIZY); companies since delisted or acquired (ABC, "
            "ANTM, AMED); mutual funds and preferred classes that are not equities "
            "(ABNFX, AHL-C); and symbols the parser invented, which are reported "
            "separately above."
        ),
        count=len(missing),
        samples=_sample(missing),
    )


def check_malformed_symbols(trades: list[dict[str, Any]]) -> Finding | None:
    """A stored symbol that is not shaped like a ticker at all.

    `-- AMCR` reached the price fetcher as a symbol, where it can only ever
    fail. A ticker is one to six characters of letters, with a dot or hyphen
    for a share class; anything else came out of the parser rather than out of
    a filing, and it is cheaper to say so than to watch it fail silently in
    the unpriceable list every run.
    """
    offenders = []
    for trade in trades:
        symbol = trade.get("symbol")
        if symbol and not SYMBOL_RE.fullmatch(str(symbol)):
            offenders.append(
                {
                    "trade_id": trade["trade_id"],
                    "symbol": symbol,
                    "asset_name": (trade.get("asset_name") or "")[:90],
                    "source_url": trade.get("source_url", ""),
                }
            )
    if not offenders:
        return None
    return Finding(
        code="malformed_symbol",
        severity="error",
        title=f"{len(offenders)} trades carry something that is not a ticker",
        detail=(
            "The stored symbol does not have the shape of a ticker, so it can "
            "never be priced and the row counts as a tradeable disclosure that "
            "will fail for ever. These come from the parser, not from a filing."
        ),
        count=len(offenders),
        samples=_sample(offenders),
    )


def duplicate_groups(trades: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Trades identical in every field a distinct decision would differ in.

    Same member, same symbol, same direction, same transaction *and*
    disclosure date, same disclosed bracket. These are real rows in real
    filings -- a member buying one name in two accounts files two lines -- so
    they are not deleted. They are grouped, because counting them as two
    independent decisions overstates conviction.
    """
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        buckets[
            (
                trade["politician_id"],
                trade.get("symbol") or trade.get("asset_name"),
                trade["transaction_type"],
                trade["transaction_date"],
                trade["disclosure_date"],
                trade["amount_low"],
                trade["amount_high"],
            )
        ].append(trade)
    return [group for group in buckets.values() if len(group) > 1]


def check_duplicates(trades: list[dict[str, Any]]) -> Finding | None:
    groups = duplicate_groups(trades)
    if not groups:
        return None
    groups.sort(key=len, reverse=True)
    samples = [
        {
            "politician_id": group[0]["politician_id"],
            "symbol": group[0].get("symbol"),
            "transaction_date": str(group[0]["transaction_date"]),
            "disclosure_date": str(group[0]["disclosure_date"]),
            "transaction_type": group[0]["transaction_type"],
            "lots": len(group),
            "source_url": group[0].get("source_url", ""),
        }
        for group in groups
    ]
    return Finding(
        code="duplicate_lots",
        severity="info",
        title=f"{len(groups)} groups of identical disclosure lines",
        detail=(
            "Same member, symbol, direction, both dates and bracket. These are "
            "kept as separate lots — a member routinely files one line per "
            "account — but they are collapsed before counting buyers, so two "
            "lines from one person never read as two people agreeing."
        ),
        count=sum(len(group) for group in groups),
        samples=_sample(samples),
    )


# -- prices ----------------------------------------------------------------


def _nearest_split_ratio(ratio: float) -> float | None:
    """The corporate-action ratio this step matches, if any."""
    for candidate in SPLIT_RATIOS:
        for value in (candidate, 1 / candidate):
            if abs(ratio - value) / value <= SPLIT_TOLERANCE:
                return value
    return None


def scan_series_for_splits(frame: Any) -> list[dict[str, Any]]:
    """Close-to-close steps that look like an unadjusted corporate action.

    Two conditions have to hold together, and the second is what keeps a
    biotech halving on trial results out of the list: the step must land on a
    split ratio, *and* the session must have opened there. A split gaps
    overnight and then trades an ordinary day; a collapse gaps and then keeps
    going, so its open and close disagree.

    A step that comes straight back is marked `reverted`. That is not a split
    at all -- a split is permanent -- it is a single bad bar from the
    provider, which is a different defect with a different fix.
    """
    if frame is None or getattr(frame, "empty", True) or len(frame) < 3:
        return []
    closes = frame["close"].astype(float).tolist()
    opens = frame["open"].astype(float).tolist()
    dates = list(frame["date"])
    stamps = [getattr(value, "date", lambda value=value: value)() for value in dates]
    hits: list[dict[str, Any]] = []
    for index in range(1, len(closes)):
        previous, current = closes[index - 1], closes[index]
        if not previous or not current or previous <= 0 or current <= 0:
            continue
        ratio = previous / current
        matched = _nearest_split_ratio(ratio)
        if matched is None:
            continue
        opening = opens[index]
        if not opening or math.isnan(opening) or opening <= 0:
            continue
        # Did the whole step happen before the session started?
        implied = previous / opening
        if abs(implied - matched) / matched > SPLIT_OPEN_TOLERANCE:
            continue
        # A split holds. A level that is restored within three sessions either
        # side is a spike, not a corporate action -- and both bars of a spike
        # match, so the window is checked in both directions.
        after = closes[index + 1 : index + 4]
        before = closes[max(0, index - 4) : index - 1]
        reverted = any(value > 0 and abs(value / previous - 1) < 0.10 for value in after) or any(
            value > 0 and abs(value / current - 1) < 0.10 for value in before
        )
        hits.append(
            {
                "date": str(stamps[index]),
                "previous_close": round(previous, 4),
                "open": round(opening, 4),
                "close": round(current, 4),
                "ratio": round(matched, 3),
                "reverted": reverted,
            }
        )
    return hits


def check_price_adjustment(
    repository: PriceRepository, symbols: list[str], reporter: JobReporter
) -> list[Finding]:
    """Are the cached series split-adjusted, and are any of them discontinuous?

    This is the check the backtest depends on. Providers here return
    back-adjusted closes today, so a 4-for-1 split leaves no trace; if a
    provider is ever swapped or silently changes behaviour, every return
    spanning a split becomes a ~75% phantom loss and nothing else in the
    system would notice.
    """
    findings: list[Finding] = []
    suspects: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    scanned = 0

    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        if total > 200 and index % 200 == 0:
            reporter.step(f"validating price series {index}/{total}", index, total)
        frame = repository.load(symbol)
        if frame.empty:
            continue
        scanned += 1
        for hit in scan_series_for_splits(frame):
            suspects.append({"symbol": symbol, **hit})
        window = frame.tail(180)
        low = float(window["close"].min())
        high = float(window["close"].max())
        if low > 0 and high / low >= 5:
            spans.append(
                {
                    "symbol": symbol,
                    "low": round(low, 2),
                    "high": round(high, 2),
                    "span": round(high / low, 1),
                }
            )

    spikes = [hit for hit in suspects if hit["reverted"]]
    steps = [hit for hit in suspects if not hit["reverted"]]
    affected = {hit["symbol"] for hit in steps}

    if spikes:
        findings.append(
            Finding(
                code="price_spike",
                severity="error",
                title=f"{len(spikes)} single bars jump and come straight back",
                detail=(
                    "The price steps by a clean multiple and returns to where it "
                    "was within three sessions. No corporate action does that — "
                    "it is one bad bar from the provider, and any return window "
                    "with an endpoint on it is wrong by that multiple."
                ),
                count=len(spikes),
                samples=_sample(spikes),
            )
        )

    if steps:
        # Isolated matches are almost always real gaps that happen to land near
        # a round number; a provider that stopped back-adjusting would produce
        # them across dozens of symbols at once. The count is the signal.
        systemic = len(affected) >= 10
        findings.append(
            Finding(
                code="unadjusted_split",
                severity="error" if systemic else "warn",
                title=(
                    f"{len(steps)} permanent price steps sit on a split ratio "
                    f"({len(affected)} symbols)"
                ),
                detail=(
                    "A close-to-close step landing on a corporate-action ratio, "
                    "with the session opening exactly there and staying. At this "
                    + ("scale it means the provider has stopped back-adjusting, and "
                       "every return spanning a split is wrong by its ratio — a "
                       "4-for-1 renders as a 75% loss."
                       if systemic else
                       "count it is more likely a handful of genuine overnight "
                       "gaps that happen to land near a round multiple. Worth an "
                       "eye; it becomes an error if the count climbs.")
                ),
                count=len(steps),
                samples=_sample(steps),
            )
        )
    else:
        findings.append(
            Finding(
                code="split_adjustment",
                severity="info",
                title=f"{scanned} price series carry no unadjusted split",
                detail=(
                    "Checked against every corporate-action ratio a split uses. "
                    "The providers return back-adjusted closes, so returns "
                    "spanning a split are measured correctly."
                ),
                count=scanned,
            )
        )

    if spans:
        spans.sort(key=lambda row: row["span"], reverse=True)
        findings.append(
            Finding(
                code="wide_price_span",
                severity="info",
                title=f"{len(spans)} series moved more than 5x in 180 sessions",
                detail=(
                    "A wide range is not itself a defect — it is what a chart "
                    "axis has to accommodate. Listed here so an extreme axis can "
                    "be recognised as real rather than investigated twice. A "
                    "genuinely broken series shows up under unadjusted splits."
                ),
                count=len(spans),
                samples=_sample(spans),
            )
        )
    return findings


# -- sources ---------------------------------------------------------------


def check_inert_sources(store: Store) -> Finding | None:
    """A source reporting `ok` while changing nothing.

    This is how the senator enrichment failed silently for all 35 senators: it
    fetched the roster, matched nobody, and reported success. "The request
    worked" and "the data changed" are different claims, and only the first
    was ever being made.

    The roster sources report their effect in `details`, not in the filing
    counters, so a check that only reads `filings_parsed` cannot see them.
    """
    state = store.get_state("sync.disclosures") or {}
    sources = state.get("sources") or {}
    inert = []
    for name, report in sources.items():
        if not report.get("ok"):
            continue
        details = report.get("details") or {}
        # Anything a source can claim to have done. A source that reports
        # nothing under any of these did nothing it can account for.
        effect = (
            report.get("filings_parsed", 0)
            + report.get("trades_added", 0)
            + report.get("politicians_added", 0)
            + int(details.get("enriched", 0) or 0)
            + int(details.get("committees", 0) or 0)
            + int(details.get("queued", 0) or 0)
        )
        # A source that *saw* filings and queued none of them is healthy: it
        # looked, and there was nothing new. Inert means it cannot account for
        # having done anything at all.
        if effect == 0 and not report.get("filings_seen"):
            inert.append({"source": name, "details": details})
    if not inert:
        return None
    return Finding(
        code="inert_source",
        severity="warn",
        title=f"{len(inert)} sources reported ok but changed nothing",
        detail=(
            "A source that succeeds and has no effect looks identical to one "
            "that is quietly broken. On a run with no new filings this is "
            "expected; on a first run, or for a roster source, it means the "
            "match failed."
        ),
        count=len(inert),
        samples=_sample(inert),
    )


def check_member_enrichment(store: Store) -> Finding | None:
    """Members the congressional roster never matched.

    Every one of these is missing a party, a state and a photo, and cannot be
    joined to committee membership. Naming them is the point: a surname
    fallback cannot resolve them and guessing would attach the wrong person's
    record to a real member.
    """
    unmatched = [
        {
            "politician_id": member["politician_id"],
            "full_name": member["full_name"],
            "chamber": member["chamber"],
        }
        for member in store.list_politicians(with_trades_only=True)
        if not member.get("bioguide_id")
    ]
    if not unmatched:
        return None
    return Finding(
        code="unmatched_member",
        severity="warn",
        title=f"{len(unmatched)} filing members are not on the current roster",
        detail=(
            "No bioguide id, so no party, photo or committee membership. These "
            "are members who left office, changed the name they file under, or "
            "share a surname with a sitting member of the same chamber. "
            "Deliberately not guessed."
        ),
        count=len(unmatched),
        samples=_sample(unmatched, limit=20),
    )


# -- the stage -------------------------------------------------------------


def implausible_trade_ids(
    trades: list[dict[str, Any]], max_days: int = MAX_REPORTING_LAG_DAYS
) -> set[str]:
    """Trades whose dates cannot be believed, and so must not be scored.

    Shared with `ranking` and `backtest` so a bad date is excluded once, in
    one place, rather than by each consumer inventing its own threshold.
    """
    return {
        trade["trade_id"]
        for trade in trades
        if trade["reporting_lag_days"] > max_days
        or trade["reporting_lag_days"] < 0
    }


def run_validation(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    price_symbol_limit: int = 4000,
) -> dict[str, Any]:
    """Run every check and store the report for `/data` to render."""
    reporter = reporter or NullReporter()
    reporter.step("validating stored disclosures")
    trades = store.list_trades(limit=500_000)

    findings: list[Finding] = []
    for check in (
        check_reporting_lag,
        check_asset_names,
        check_symbol_names,
        check_malformed_symbols,
        check_duplicates,
    ):
        found = check(trades)
        if found is not None:
            findings.append(found)

    found = check_unpriceable_symbols(trades, repository)
    if found is not None:
        findings.append(found)

    reporter.step("validating price series")
    symbols = repository.cached_symbols()[:price_symbol_limit]
    findings.extend(check_price_adjustment(repository, symbols, reporter))

    for source_check in (check_inert_sources, check_member_enrichment):
        found = source_check(store)
        if found is not None:
            findings.append(found)

    order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda item: (order.get(item.severity, 3), -item.count))
    counts = Counter(item.severity for item in findings)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "trades_checked": len(trades),
        "series_checked": len(symbols),
        "errors": counts.get("error", 0),
        "warnings": counts.get("warn", 0),
        "notes": counts.get("info", 0),
        "findings": [item.model_dump(mode="json") for item in findings],
    }
    store.set_state(STATE_KEY, report)

    if counts.get("error"):
        reporter.note(
            f"validation found {counts['error']} errors and {counts.get('warn', 0)} warnings",
            level="error",
            detail="; ".join(item.title for item in findings if item.severity == "error"),
        )
    elif counts.get("warn"):
        reporter.note(
            f"validation found {counts['warn']} warnings", level="warn",
            detail="; ".join(item.title for item in findings if item.severity == "warn"),
        )
    else:
        reporter.note("validation found nothing implausible", level="success")
    return {
        "errors": report["errors"],
        "warnings": report["warnings"],
        "notes": report["notes"],
        "trades_checked": report["trades_checked"],
    }


def load_report(store: Store) -> dict[str, Any] | None:
    return store.get_state(STATE_KEY)


__all__ = [
    "MAX_REPORTING_LAG_DAYS",
    "Finding",
    "duplicate_groups",
    "implausible_trade_ids",
    "load_report",
    "run_validation",
    "scan_series_for_splits",
]
