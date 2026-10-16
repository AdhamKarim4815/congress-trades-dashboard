"""Does following these people actually work?

Everything else in this application takes that premise for granted. This
module tests it, walk-forward, with no lookahead:

At each rebalance date the members are ranked using **only** trades whose
outcome was already known on that date -- a trade disclosed three weeks
earlier has not finished its 90-day horizon, so it cannot inform a ranking
made today. The top members by that as-of ranking are then followed through
the next period: every ticker they disclose is bought at the
disclosure-date close, held for the horizon, and compared against the
benchmark over exactly the same window.

The two rules that make this honest rather than flattering are the ones that
run through the whole codebase. Entries are at the *disclosure* price, never
the member's, because that is the first price a copier could pay. And the
ranking at time T is built from information available at time T, which is
what stops the test from picking the members who turned out well.

A result here is a measurement, not a promise, and this module is now
responsible for saying how much of one. Three things were being asserted in
prose on the evidence page and are computed here instead:

* **an interval.** The headline was a point estimate over a couple of hundred
  positions with a standard deviation seven times its mean. Every figure now
  carries a bootstrap confidence interval, and a result whose interval
  straddles zero is reported as inconclusive rather than as a number.
* **a control.** Walk-forward stops the test seeing the future; it does not
  stop it picking three lucky members out of eighty, twenty-two times. So the
  same walk is run again choosing members *at random*, and the real result is
  placed in that distribution. If it sits inside, the ranking is decoration.
* **every horizon.** Four are computed and one used to headline, which is a
  choice that can flatter. All four are now reported together.

None of that is free, so the trade pricing is memoised across the whole run
(see `_Pricer`) -- which it should always have been, since the same trade was
being re-priced at every rebalance it was eligible for.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import comb
from statistics import mean, median
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.committees import ConflictIndex
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import TradePerformance, TransactionType
from trade_bot.performance import evaluate_trade
from trade_bot.prices import PriceRepository
from trade_bot.ranking import shrink
from trade_bot.recommend import damp_unproven, structural_conviction


class _Pricer:
    """Prices a trade once per horizon and remembers the answer.

    The ranking re-prices every resolved trade at every rebalance, so one
    trade was being valued up to twenty-two times over a two-year walk. The
    result is identical each time and the reason is worth stating, because it
    is the thing that makes the cache safe: `_rank_members_as_of` only admits
    trades whose full horizon had already closed by `as_of`, so the window end
    is `disclosure + horizon` and never the clamped `today`. Nothing about the
    valuation depends on which rebalance is asking.

    Memoising it is what pays for the confidence intervals, the random control
    and the extra horizons -- together they multiply the number of walks by
    about five, and this divides the cost of each by rather more.
    """

    def __init__(self, repository: PriceRepository, benchmark: str) -> None:
        self._repository = repository
        self._benchmark = benchmark
        self._cache: dict[tuple[str, int], TradePerformance] = {}
        self.hits = 0
        self.misses = 0

    def price(self, trade: dict[str, Any], horizon: int, today: date) -> TradePerformance:
        key = (trade["trade_id"], horizon)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = evaluate_trade(
            trade, self._repository, self._benchmark, horizon_days=horizon, today=today
        )
        self._cache[key] = result
        return result


def bootstrap_interval(
    values: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 7
) -> tuple[float, float] | None:
    """A percentile bootstrap interval around the mean.

    Seeded, because this number is stored and rendered: an interval that moved
    when nothing else did would be indistinguishable from the data changing.
    """
    if len(values) < 8:
        return None
    generator = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(generator.choices(values, k=size)) / size for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def coin_flip_probability(wins: int, total: int) -> float | None:
    """Exact two-sided binomial probability of a beat rate this lopsided.

    A beat rate is the claim most likely to be over-read: 55% sounds like an
    edge and, over a couple of hundred positions, is what a fair coin does
    roughly one time in ten.
    """
    if total == 0 or total > 4000:  # exact terms get unwieldy, and n is never this large
        return None
    extreme = max(wins, total - wins)
    tail = sum(comb(total, k) for k in range(extreme, total + 1)) / 2**total
    return float(min(1.0, 2 * tail))


class _ClusterIndex:
    """What was knowable about a disclosure on the day it became public.

    The board's conviction score blends six things, and only some of them can
    be reconstructed for a date in the past. How many members had bought the
    same name recently, and for how much, can: it is in the disclosure record
    and nowhere else. News tone and political mentions cannot -- the headlines
    were fetched this week, and pretending to know last year's sentiment would
    be the exact lookahead the rest of this module refuses.
    """

    def __init__(self, trades: list[dict[str, Any]], window_days: int) -> None:
        self._buys: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._sells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            kind = TransactionType(trade["transaction_type"])
            bucket = self._buys if kind.is_buy else self._sells if kind.is_sell else None
            if bucket is not None and trade["symbol"]:
                bucket[trade["symbol"]].append(trade)
        self._window = window_days

    def _within(
        self, table: dict[str, list[dict[str, Any]]], symbol: str, as_of: date
    ) -> list[dict[str, Any]]:
        start = as_of - timedelta(days=self._window)
        return [
            trade
            for trade in table.get(symbol, [])
            if start <= trade["disclosure_date"] <= as_of
        ]

    def buyers(self, symbol: str, as_of: date) -> tuple[int, float]:
        rows = self._within(self._buys, symbol, as_of)
        return (
            len({row["politician_id"] for row in rows}),
            sum(row.get("amount_midpoint", 0.0) for row in rows),
        )

    def sellers(self, symbol: str, as_of: date) -> int:
        return len({row["politician_id"] for row in self._within(self._sells, symbol, as_of)})


def conviction_as_of(
    settings: Settings,
    rank: int | None,
    buyers: int,
    dollars: float,
    sellers: int,
) -> float:
    """The board's own conviction formula, over the parts knowable at entry.

    Calls straight into `recommend`, so the score being tested is literally
    the score being shipped. Keeping a second copy here meant that removing a
    damper from the board left this measurement grading the old formula and
    dutifully reporting that nothing had changed.

    News tone and political mentions are absent: both are fetched in the
    present and cannot be known as of a past disclosure. `sellers` is accepted
    and unused, kept because the caller has it and because the opposing-side
    damper it fed is exactly the thing the measurement removed -- a signature
    that still names it is easier to restore than one that forgot.

    Recency is at full value for every position and therefore constant: a
    position is taken on the disclosure date. That is worth knowing on its own
    -- it is 12% of the score a reader sees, and it can only ever separate
    cards by how stale they have become.
    """
    del sellers
    score = structural_conviction(
        settings, rank, buyers, dollars, days_ago=0,
        window=settings.recommendations.disclosure_window_days,
    )
    return damp_unproven(settings, score, rank)


@dataclass
class FollowedTrade:
    """One position the strategy actually took."""

    symbol: str
    politician_id: str
    member_name: str
    disclosure_date: date
    entry_price: float
    exit_price: float
    trade_return: float
    benchmark_return: float
    excess_return: float
    rank_at_entry: int
    # The board's own score, rebuilt from what was knowable that day. None
    # when the walk was run without the context needed to reconstruct it.
    conviction_at_entry: float | None = None
    # How many distinct members had disclosed a purchase of this name in the
    # window ending that day. Carried separately because the blend is only
    # useful if you can also ask which part of it is doing the work.
    buyers_at_entry: int | None = None


@dataclass
class BacktestResult:
    """Positions the strategy took. Purchases only -- see `run_backtest`."""

    horizon_days: int
    top_n: int
    rebalance_days: int
    start: date | None = None
    end: date | None = None
    positions: list[FollowedTrade] = field(default_factory=list)
    rebalances: int = 0
    skipped_unpriced: int = 0

    def summary(self) -> dict[str, Any]:
        if not self.positions:
            return {
                "positions": 0,
                "note": "no positions were taken; widen the window or lower min trades",
            }
        returns = [position.trade_return for position in self.positions]
        excess = [position.excess_return for position in self.positions]
        benchmark = [position.benchmark_return for position in self.positions]
        wins = sum(1 for value in excess if value > 0)
        interval = bootstrap_interval(excess)
        # "Inconclusive" is a result. A mean of +3.4% whose interval runs from
        # +0.03% is not evidence of an edge, and printing the 3.4% alone
        # invites a reader to treat it as one.
        conclusive = bool(interval and (interval[0] > 0 or interval[1] < 0))
        return {
            "positions": len(self.positions),
            "rebalances": self.rebalances,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "horizon_days": self.horizon_days,
            "top_n": self.top_n,
            "mean_return": mean(returns),
            "median_return": median(returns),
            "mean_benchmark_return": mean(benchmark),
            "mean_excess_return": mean(excess),
            "median_excess_return": median(excess),
            "excess_ci_low": interval[0] if interval else None,
            "excess_ci_high": interval[1] if interval else None,
            "conclusive": conclusive,
            "beat_benchmark_rate": wins / len(excess),
            "beat_benchmark_wins": wins,
            "beat_rate_probability": coin_flip_probability(wins, len(excess)),
            "best": max(self.positions, key=lambda p: p.excess_return).symbol,
            "worst": min(self.positions, key=lambda p: p.excess_return).symbol,
            "unique_symbols": len({position.symbol for position in self.positions}),
            "unique_members": len({position.politician_id for position in self.positions}),
            "skipped_unpriced": self.skipped_unpriced,
            # Carried in the payload so a template cannot quote the number
            # without the scope being available to quote alongside it.
            "side": "buy",
        }


def _rank_members_as_of(
    trades: list[dict[str, Any]],
    pricer: _Pricer,
    as_of: date,
    horizon_days: int,
    lookback_days: int,
    min_trades: int,
    shrinkage_prior: float,
) -> list[tuple[str, float, int]]:
    """Rank members by their *buying* using only outcomes resolved by `as_of`.

    Two rules, and the second was learned the hard way. The horizon cut
    prevents lookahead: a trade disclosed two weeks before `as_of` has not
    finished its 90-day window, so its result is not knowable yet and must not
    influence the ranking that selects it. And the ranking is built from
    purchases alone, so that the members it picks are the members whose
    purchases the strategy is about to copy.
    """
    window_start = as_of - timedelta(days=lookback_days)
    resolved = [
        trade
        for trade in trades
        if window_start <= trade["disclosure_date"]
        and trade["disclosure_date"] + timedelta(days=horizon_days) <= as_of
        # Ranked on purchases, because purchases are what the strategy then
        # copies. Ranking on the combined record and buying the result meant
        # "the top 3" named one set of members here and a different set on the
        # page quoting this number -- a member can lead the combined ranking on
        # sales alone and contribute no buys at all.
        and TransactionType(trade["transaction_type"]).is_buy
    ]
    by_member: dict[str, list[float]] = defaultdict(list)
    for trade in resolved:
        performance = pricer.price(trade, horizon_days, as_of)
        if performance.priced and performance.excess_return is not None:
            by_member[trade["politician_id"]].append(performance.excess_return)

    ranked = [
        (
            politician_id,
            shrink(mean(values), len(values), shrinkage_prior),
            len(values),
        )
        for politician_id, values in by_member.items()
        if len(values) >= min_trades
    ]
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked


# Every top-level key `refresh_backtest` writes. Used to decide whether a
# cached measurement is merely old or actually out of date with the code.
PAYLOAD_KEYS = frozenset(
    {
        "summary",
        "wide",
        "by_rank",
        "top_members",
        "control",
        "by_conviction",
        "by_cluster",
        "by_opposition",
        "exits",
        "by_horizon",
        "coverage",
        "sensitivity",
        "sensitivity_shape",
        "family_threshold",
        "computed_at",
    }
)


@dataclass
class _Rebalance:
    """One decision point: who was rankable, and what they went on to file."""

    at: date
    ranked: list[tuple[str, float, int]]
    candidates: list[dict[str, Any]]


def _plan_walk(
    settings: Settings,
    buys: list[dict[str, Any]],
    all_trades: list[dict[str, Any]],
    pricer: _Pricer,
    horizon: int,
    rebalance_days: int,
    years: float,
    min_trades: int,
    today: date,
    reporter: JobReporter,
) -> tuple[list[_Rebalance], date, date]:
    """Every rebalance, with its ranking and the disclosures that followed it.

    Separated from taking positions so the same walk can be replayed under a
    different selection rule. Ranking is the expensive half and does not
    depend on who you then choose, so the random control costs almost nothing
    once this has run.
    """
    earliest = min(trade["disclosure_date"] for trade in all_trades)
    start = max(earliest + timedelta(days=horizon + 180), today - timedelta(days=365 * years))
    last_entry = today - timedelta(days=horizon)

    plan: list[_Rebalance] = []
    cursor = start
    total_steps = max(int((last_entry - start).days / rebalance_days), 1)
    step = 0
    while cursor <= last_entry:
        step += 1
        reporter.step(f"backtest {horizon}d: {cursor} ({step}/{total_steps})", step, total_steps)
        ranked = _rank_members_as_of(
            all_trades,
            pricer,
            as_of=cursor,
            horizon_days=horizon,
            lookback_days=settings.ranking.lookback_days,
            min_trades=min_trades,
            shrinkage_prior=settings.ranking.shrinkage_prior_trades,
        )
        period_end = min(cursor + timedelta(days=rebalance_days), last_entry + timedelta(days=1))
        plan.append(
            _Rebalance(
                at=cursor,
                ranked=ranked,
                candidates=[
                    trade for trade in buys if cursor < trade["disclosure_date"] <= period_end
                ],
            )
        )
        cursor += timedelta(days=rebalance_days)
    return plan, start, min(cursor, last_entry)


def _take_positions(
    plan: list[_Rebalance],
    selector: Any,
    pricer: _Pricer,
    members: dict[str, dict[str, Any]],
    horizon: int,
    top_n: int,
    rebalance_days: int,
    start: date,
    end: date,
    today: date,
    settings: Settings | None = None,
    context: _ClusterIndex | None = None,
) -> BacktestResult:
    """Replay a planned walk, choosing members with `selector`."""
    result = BacktestResult(
        horizon_days=horizon,
        top_n=top_n,
        rebalance_days=rebalance_days,
        start=start,
        end=end,
        rebalances=len(plan),
    )
    for rebalance in plan:
        selected = selector(rebalance)
        if not selected:
            continue
        ranks = {politician_id: index for index, (politician_id, _, _) in
                 enumerate(rebalance.ranked, start=1)}
        for trade in rebalance.candidates:
            if trade["politician_id"] not in selected:
                continue
            performance = pricer.price(trade, horizon, today)
            if (
                not performance.priced
                or performance.disclosure_price is None
                or performance.exit_price is None
                or performance.achievable_return is None
            ):
                result.skipped_unpriced += 1
                continue
            result.positions.append(
                FollowedTrade(
                    symbol=trade["symbol"],
                    politician_id=trade["politician_id"],
                    member_name=members.get(trade["politician_id"], {}).get(
                        "full_name", trade["politician_id"]
                    ),
                    disclosure_date=trade["disclosure_date"],
                    entry_price=performance.disclosure_price,
                    exit_price=performance.exit_price,
                    trade_return=performance.achievable_return,
                    benchmark_return=performance.benchmark_return or 0.0,
                    # The signed one off the performance record, not the
                    # subtraction re-derived here. They agree today only
                    # because the input is filtered to purchases; the day
                    # a sell position is added, re-deriving it would give
                    # every one of them the wrong sign in silence.
                    excess_return=performance.excess_return
                    if performance.excess_return is not None
                    else performance.achievable_return
                    - (performance.benchmark_return or 0.0),
                    rank_at_entry=selected[trade["politician_id"]],
                    buyers_at_entry=(
                        None
                        if context is None
                        else context.buyers(trade["symbol"], trade["disclosure_date"])[0]
                    ),
                    conviction_at_entry=(
                        None
                        if settings is None or context is None
                        else conviction_as_of(
                            settings,
                            # The rank the *board* would have used, which is
                            # the member's standing overall rather than their
                            # slot in whatever this run happened to select.
                            ranks.get(trade["politician_id"]),
                            *context.buyers(trade["symbol"], trade["disclosure_date"]),
                            context.sellers(trade["symbol"], trade["disclosure_date"]),
                        )
                    ),
                )
            )
    return result


def _top_selector(top_n: int) -> Any:
    def choose(rebalance: _Rebalance) -> dict[str, int]:
        return {
            politician_id: position
            for position, (politician_id, _, _) in enumerate(
                rebalance.ranked[:top_n], start=1
            )
        }

    return choose


def _random_selector(top_n: int, generator: random.Random) -> Any:
    """The control: `top_n` rankable members, chosen without looking at rank.

    Same eligibility bar, same period, same pricing -- the only thing removed
    is the ranking. Whatever the strategy earns above this is what the ranking
    is worth; whatever it earns below is what picking three names out of
    eighty earns on its own.
    """

    def choose(rebalance: _Rebalance) -> dict[str, int]:
        pool = [politician_id for politician_id, _, _ in rebalance.ranked]
        if not pool:
            return {}
        picked = generator.sample(pool, k=min(top_n, len(pool)))
        return {politician_id: index for index, politician_id in enumerate(picked, start=1)}

    return choose


def run_backtest(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    top_n: int = 10,
    horizon_days: int | None = None,
    rebalance_days: int = 30,
    years: float = 2.0,
    min_trades: int = 8,
    today: date | None = None,
) -> BacktestResult:
    """Follow the top-ranked members forward and see what it earned."""
    reporter = reporter or NullReporter()
    today = today or date.today()
    horizon = horizon_days or settings.ranking.primary_horizon_days

    trades = store.list_trades(actionable_only=True, limit=200_000)
    # Purchases only, and every page quoting this number has to say so. A
    # short is not something a copier can do from a disclosure, so there is no
    # honest way to take the sell side as a position here -- but that makes
    # the result a measurement of the *buy* board, not of the board.
    buys = [trade for trade in trades if TransactionType(trade["transaction_type"]).is_buy]
    if not buys:
        return BacktestResult(horizon_days=horizon, top_n=top_n, rebalance_days=rebalance_days)

    pricer = _Pricer(repository, settings.prices.benchmark_symbol)
    plan, start, end = _plan_walk(
        settings, buys, trades, pricer, horizon, rebalance_days, years, min_trades,
        today, reporter,
    )
    members = {member["politician_id"]: member for member in store.list_politicians()}
    result = _take_positions(
        plan, _top_selector(top_n), pricer, members, horizon, top_n, rebalance_days,
        start, end, today,
    )
    reporter.note(
        f"backtest: {len(result.positions)} positions across {result.rebalances} rebalances"
    )
    return result


def refresh_backtest(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    top_n: int | None = None,
    max_age_days: float = 7.0,
    today: date | None = None,
    control_trials: int = 25,
    rebalance_days: int = 30,
    years: float = 2.0,
    min_trades: int = 8,
) -> dict[str, Any]:
    """Run the strategy test, its control and its intervals, and cache them.

    Four things come out of one walk per horizon, because the walk is the
    expensive part and the selection rule is not:

    * **narrow** -- the top `trusted_rank` members, which is the cutoff the
      board's conviction scoring keys off. This is the headline.
    * **wide** -- positions taken deep into the ranking, so the rank-bucket
      breakdown can say whether the edge is concentrated at the top or the
      narrow run simply got lucky.
    * **control** -- the same walk choosing members at random. Walk-forward
      prevents lookahead but not selection: picking three from eighty, twenty
      times, will find winners by chance, and this is the only thing that says
      how many.
    * **every horizon** -- narrow at all four, because reporting the one that
      headlines is a choice a reader cannot see.
    """
    reporter = reporter or NullReporter()
    # Still the most expensive stage in the pipeline, though far less so than
    # it was: pricing is memoised across every walk, which is what made room
    # for the control and the extra horizons. What it measures moves far too
    # slowly to justify recomputing it daily.
    cached = store.get_state("board.backtest") or {}
    computed_at = cached.get("computed_at")
    if computed_at and max_age_days > 0:
        try:
            age = ((today or date.today()) - date.fromisoformat(computed_at[:10])).days
        except ValueError:
            age = None
        missing = PAYLOAD_KEYS - set(cached)
        if age is not None and age < max_age_days and not missing:
            reporter.note(f"backtest: reusing the measurement from {computed_at[:10]}")
            summary: dict[str, Any] = cached.get("summary") or {}
            return summary
        if age is not None and age < max_age_days:
            # Recent but incomplete. Age alone would have kept a cache written
            # before a new measurement existed, and every page guards on these
            # keys by rendering nothing -- so a deploy that added a section
            # would have shipped it invisibly until the cache aged out a week
            # later, with no error anywhere to say why.
            reporter.note(
                f"backtest: the cached measurement predates {', '.join(sorted(missing))}; "
                "recomputing rather than serving a page with holes in it",
                level="warn",
            )

    today = today or date.today()
    narrow_n = top_n or settings.recommendations.trusted_rank
    wide_n = max(narrow_n * 3, 10)
    primary = settings.ranking.primary_horizon_days
    horizons = sorted({*settings.ranking.horizons_days, primary})

    trades = store.list_trades(actionable_only=True, limit=200_000)
    buys = [trade for trade in trades if TransactionType(trade["transaction_type"]).is_buy]
    if not buys:
        reporter.note("backtest: no purchases to follow", level="warn")
        return {"positions": 0, "note": "no disclosed purchases to follow"}

    members = {member["politician_id"]: member for member in store.list_politicians()}
    pricer = _Pricer(repository, settings.prices.benchmark_symbol)
    context = _ClusterIndex(trades, settings.recommendations.disclosure_window_days)

    payload: dict[str, Any] = {}
    by_horizon: list[dict[str, Any]] = []
    for horizon in horizons:
        reporter.note(f"backtest: walking the {horizon}-day horizon")
        plan, start, end = _plan_walk(
            settings, buys, trades, pricer, horizon, rebalance_days, years,
            min_trades, today, reporter,
        )
        def take(selector: Any, size: int, _plan: list[_Rebalance] = plan,
                 _horizon: int = horizon, _start: date = start, _end: date = end
                 ) -> BacktestResult:
            return _take_positions(
                _plan, selector, pricer, members, _horizon, size, rebalance_days,
                _start, _end, today, settings, context,
            )

        narrow = take(_top_selector(narrow_n), narrow_n)
        narrow_summary = narrow.summary()
        by_horizon.append({**narrow_summary, "horizon_days": horizon})

        if horizon != primary:
            continue

        wide = take(_top_selector(wide_n), wide_n)
        # The control shares the plan, so each extra trial costs one replay
        # rather than one walk.
        reporter.note(f"backtest: {control_trials} random-{narrow_n} control runs")
        trials = []
        for trial in range(control_trials):
            picked = take(_random_selector(narrow_n, random.Random(1000 + trial)), narrow_n)
            if picked.positions:
                trials.append(mean(item.excess_return for item in picked.positions))
        payload.update(
            {
                "summary": narrow_summary,
                "wide": wide.summary(),
                "by_rank": _by_rank(wide),
                "top_members": _top_members(wide),
                "control": _control_summary(trials, narrow_summary.get("mean_excess_return")),
                # From the wide run: the narrow one takes too few positions to
                # split four ways, and conviction is a property of the card
                # rather than of the slot it was selected into.
                "by_conviction": _by_conviction(wide),
                "by_cluster": _by_cluster(wide),
                "by_opposition": _by_opposition(wide, context),
                "exits": _exit_test(
                    wide, trades, repository, settings.prices.benchmark_symbol, horizon,
                    ConflictIndex(settings, store),
                ),
            }
        )

    # Four horizons is four tests, and a 5% bar applied to each of them is
    # not a 5% bar on the family: one in five sets of four coin flips throws
    # up a p under 0.05 somewhere. The table colours significance, so it has
    # to colour it against a threshold that survives the arithmetic.
    family = max(len(by_horizon), 1)
    threshold = 0.05 / family
    for row in by_horizon:
        probability = row.get("beat_rate_probability")
        row["significant"] = probability is not None and probability < threshold
    payload["family_threshold"] = threshold
    payload["family_size"] = family
    reporter.note("backtest: sweeping the four parameters that were never varied")
    plan_context = (buys, trades, members, context)
    sensitivity = _sensitivity(
        settings, store, repository, pricer, *plan_context, primary, today, reporter
    )
    payload["sensitivity"] = sensitivity
    payload["sensitivity_shape"] = _sensitivity_shape(sensitivity)
    payload["by_horizon"] = by_horizon
    payload["coverage"] = _coverage(trades, repository)
    payload["computed_at"] = today.isoformat()
    payload["pricing_cache"] = {"hits": pricer.hits, "misses": pricer.misses}
    store.set_state("board.backtest", payload)
    return dict(payload.get("summary") or {})


def _sensitivity(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    pricer: _Pricer,
    buys: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    members: dict[str, dict[str, Any]],
    context: _ClusterIndex,
    horizon: int,
    today: date,
    reporter: JobReporter,
) -> list[dict[str, Any]]:
    """The headline under every setting it was not run with.

    Four numbers are chosen and never varied: how many resolved trades a
    member needs before they can be ranked, how often the strategy
    rebalances, how far back the walk starts, and how many members it
    follows. Each was a judgement, and a result that only survives the
    particular four is a property of the choice rather than of the members.

    Reported as a spread rather than a best case. If the headline sits at the
    top of this table it was tuned, whether or not anyone meant to tune it.
    """
    grid: list[tuple[str, dict[str, Any]]] = [
        ("as published", {}),
        ("min 4 resolved trades", {"min_trades": 4}),
        ("min 12 resolved trades", {"min_trades": 12}),
        ("rebalance every 14 days", {"rebalance_days": 14}),
        ("rebalance every 60 days", {"rebalance_days": 60}),
        ("one year of history", {"years": 1.0}),
        ("three years of history", {"years": 3.0}),
        ("follow the top 5", {"top_n": 5}),
        ("follow the top 10", {"top_n": 10}),
    ]
    rows: list[dict[str, Any]] = []
    for label, overrides in grid:
        reporter.step(f"sensitivity: {label}")
        top_n = overrides.get("top_n", settings.recommendations.trusted_rank)
        plan, start, end = _plan_walk(
            settings,
            buys,
            trades,
            pricer,
            horizon,
            overrides.get("rebalance_days", 30),
            overrides.get("years", 2.0),
            overrides.get("min_trades", 8),
            today,
            reporter,
        )
        result = _take_positions(
            plan, _top_selector(top_n), pricer, members, horizon, top_n,
            overrides.get("rebalance_days", 30), start, end, today, settings, context,
        )
        summary = result.summary()
        if not summary.get("positions"):
            continue
        rows.append(
            {
                "label": label,
                "positions": summary["positions"],
                "mean_excess": summary["mean_excess_return"],
                "ci_low": summary.get("excess_ci_low"),
                "ci_high": summary.get("excess_ci_high"),
                "conclusive": summary.get("conclusive", False),
                "beat_rate": summary["beat_benchmark_rate"],
            }
        )
    return rows


def _sensitivity_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Is the published setting a plateau or the peak of a spike?"""
    if len(rows) < 3:
        return {}
    published = next((row for row in rows if row["label"] == "as published"), rows[0])
    others = [row["mean_excess"] for row in rows if row is not published]
    better = sum(1 for value in others if value > published["mean_excess"])
    positive = sum(1 for row in rows if row["mean_excess"] > 0)
    return {
        "variants": len(rows),
        "published": published["mean_excess"],
        "best": max(row["mean_excess"] for row in rows),
        "worst": min(row["mean_excess"] for row in rows),
        "median": median([row["mean_excess"] for row in rows]),
        "settings_that_did_better": better,
        "settings_still_positive": positive,
        "conclusive_count": sum(1 for row in rows if row["conclusive"]),
    }


def _control_summary(trials: list[float], actual: float | None) -> dict[str, Any]:
    """Where the real result falls among runs that ignored the ranking.

    A percentile, not a verdict. If the strategy beats 60% of random draws it
    is not distinguishable from luck, and the number says so more plainly than
    any adjective.
    """
    if not trials or actual is None:
        return {"trials": 0}
    ordered = sorted(trials)
    beaten = sum(1 for value in ordered if actual > value)
    return {
        "trials": len(ordered),
        "actual": actual,
        "median": median(ordered),
        "best": ordered[-1],
        "worst": ordered[0],
        "percentile": beaten / len(ordered),
        # The usual bar. Beating 19 of 20 random draws is the one-sided 5%
        # equivalent, and anything short of it is not evidence of skill.
        "beats_chance": beaten / len(ordered) >= 0.95,
    }


def _coverage(trades: list[dict[str, Any]], repository: PriceRepository) -> dict[str, Any]:
    """How much of the record the measurement could not see.

    A trade with no price series is dropped from every score and every
    position silently. That is survivorship: the companies that vanish are
    exactly the ones with no series left to fetch, and they are concentrated
    in the oldest years of the window. Reporting the share and the skew is the
    difference between a known limitation and a hidden one.
    """
    cached = {symbol.upper() for symbol in repository.cached_symbols()}
    priced_years: dict[int, int] = defaultdict(int)
    unpriced_years: dict[int, int] = defaultdict(int)
    unpriced_symbols: set[str] = set()
    for trade in trades:
        symbol = (trade["symbol"] or "").upper()
        year = trade["disclosure_date"].year
        if symbol in cached:
            priced_years[year] += 1
        else:
            unpriced_years[year] += 1
            unpriced_symbols.add(symbol)
    total = sum(priced_years.values()) + sum(unpriced_years.values())
    unpriced = sum(unpriced_years.values())
    years = sorted({*priced_years, *unpriced_years})
    return {
        "actionable_trades": total,
        "unpriced_trades": unpriced,
        "unpriced_share": unpriced / total if total else 0.0,
        "unpriced_symbols": len(unpriced_symbols),
        "by_year": [
            {
                "year": year,
                "priced": priced_years.get(year, 0),
                "unpriced": unpriced_years.get(year, 0),
                "unpriced_share": (
                    unpriced_years.get(year, 0)
                    / (priced_years.get(year, 0) + unpriced_years.get(year, 0))
                ),
            }
            for year in years
        ],
    }


def _exit_test(
    result: BacktestResult,
    trades: list[dict[str, Any]],
    repository: PriceRepository,
    benchmark: str,
    horizon: int,
    conflicts: Any = None,
) -> dict[str, Any]:
    """Is following their *sells* worth anything?

    I said earlier that the sell side could not be tested because a
    disclosure cannot be copied into a short. That is true of a standalone
    position and false of the thing this application already does: if you
    bought because a member bought, their sale is an **exit from something you
    hold**, and exiting is entirely copyable.

    So the test reuses the positions the buy strategy actually took. For each
    one, if the same member later disclosed a sale of the same name before the
    horizon closed, the position is priced twice -- held to the horizon, and
    sold at the first close after their filing -- and the difference is what
    honouring their exit was worth. Same positions, same entries, one decision
    changed.

    **What happens to the money after the exit is the assumption that makes
    this fair, so it is stated rather than buried:** the proceeds go into the
    benchmark. Both legs are then measured as excess return over the *same*
    full window, and the exited leg simply stops accruing excess at the sale.
    Comparing an excess return over forty days against one over ninety would
    hand the win to whichever leg was invested longer, which is a property of
    the clock rather than of the decision.
    """
    by_member_symbol: dict[tuple[str, str], list[date]] = defaultdict(list)
    for trade in trades:
        if TransactionType(trade["transaction_type"]).is_sell and trade["symbol"]:
            by_member_symbol[(trade["politician_id"], trade["symbol"])].append(
                trade["disclosure_date"]
            )

    held: list[float] = []
    exited: list[float] = []
    conflicted: list[bool] = []
    signalled = 0
    for position in result.positions:
        window_end = position.disclosure_date + timedelta(days=horizon)
        sales = sorted(
            day
            for day in by_member_symbol.get(
                (position.politician_id, position.symbol), []
            )
            if position.disclosure_date < day <= window_end
        )
        if not sales:
            continue
        exit_price, exit_on = repository.close_on_or_after(position.symbol, sales[0])
        if exit_price is None or exit_on is None or position.entry_price <= 0:
            continue
        early = (exit_price - position.entry_price) / position.entry_price
        market, _, _ = repository.return_between(
            benchmark, position.disclosure_date, exit_on
        )
        signalled += 1
        held.append(position.excess_return)
        exited.append(early - (market or 0.0))
        conflicted.append(
            bool(conflicts and conflicts.overlaps(position.politician_id, position.symbol))
        )

    if signalled < 8:
        return {"positions": signalled, "note": "too few exit signals to measure"}
    differences = [a - b for a, b in zip(exited, held, strict=True)]
    conflicted_split = _exit_conflicts(conflicted, differences)
    interval = bootstrap_interval(differences)
    return {
        "positions": signalled,
        "of_total": len(result.positions),
        "held_mean_excess": mean(held),
        "exited_mean_excess": mean(exited),
        "difference": mean(differences),
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "conclusive": bool(interval and (interval[0] > 0 or interval[1] < 0)),
        "helped_rate": sum(1 for value in differences if value > 0) / len(differences),
        **conflicted_split,
    }


def _exit_conflicts(flags: list[bool], differences: list[float]) -> dict[str, Any]:
    """Do the exits that pay belong to members overseeing the company?

    This is the only thing that reconciles two results that otherwise sit
    awkwardly together. The board weights a committee overlap on the sell side
    because sales inside a member's remit measured markedly better than sales
    outside it. The exit test then says that acting on a sale in a position you
    hold *costs* money. Both can be true -- they measure different actions --
    but if the remit weighting is picking out anything real, the exits worth
    honouring should be the conflicted ones.
    """
    inside = [value for flag, value in zip(flags, differences, strict=True) if flag]
    outside = [value for flag, value in zip(flags, differences, strict=True) if not flag]
    if len(inside) < 8 or len(outside) < 8:
        return {"conflicted_exits": len(inside), "conflicted_note": "too few to split"}
    return {
        "conflicted_exits": len(inside),
        "conflicted_difference": mean(inside),
        "unconflicted_difference": mean(outside),
        "conflict_helps": mean(inside) > mean(outside),
    }


def _by_conviction(result: BacktestResult, buckets: int = 4) -> dict[str, Any]:
    """Does the number on every card predict anything?

    Conviction is the most prominent figure in the product -- a 0 to 100 score
    with a bar beside it on every card -- and until now only one of its six
    components had ever been measured. This buckets past positions by the
    score they would have carried at entry and asks whether the buckets come
    out in order.

    Quartiles rather than fixed bands, because the reconstructed score spans a
    narrower range than the live one (news and the political bonus are absent)
    and fixed bands would pile every position into one.
    """
    scored = [
        position for position in result.positions if position.conviction_at_entry is not None
    ]
    if len(scored) < buckets * 8:
        return {"buckets": [], "note": "too few positions to split"}
    scored.sort(key=lambda position: position.conviction_at_entry or 0.0)
    size = len(scored) // buckets
    rows: list[dict[str, Any]] = []
    for index in range(buckets):
        lower = index * size
        upper = len(scored) if index == buckets - 1 else (index + 1) * size
        slice_ = scored[lower:upper]
        excess = [position.excess_return for position in slice_]
        interval = bootstrap_interval(excess)
        rows.append(
            {
                "quartile": index + 1,
                "conviction_low": slice_[0].conviction_at_entry,
                "conviction_high": slice_[-1].conviction_at_entry,
                "positions": len(slice_),
                "mean_excess": mean(excess),
                "ci_low": interval[0] if interval else None,
                "ci_high": interval[1] if interval else None,
                "beat_rate": sum(1 for value in excess if value > 0) / len(excess),
            }
        )
    means = [row["mean_excess"] for row in rows]
    return {
        "buckets": rows,
        "monotone": means == sorted(means),
        "top_beats_bottom": means[-1] > means[0],
        "spread": means[-1] - means[0],
        # Named so a reader knows what the score being tested is missing.
        "excluded": ["news sentiment", "political mentions", "committee overlap"],
        "constant": ["recency"],
    }


def _by_cluster(result: BacktestResult) -> list[dict[str, Any]]:
    """How agreement between members actually pays.

    Clustering carries 22% of the conviction score, second only to the buyer's
    record, on the reasoning that independent buyers rarely coincide by
    chance. That is an argument, not a measurement, and it is the obvious
    suspect whenever the blended score fails to order outcomes.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for position in result.positions:
        count = position.buyers_at_entry
        if count is None:
            continue
        label = "1 buyer" if count <= 1 else "2" if count == 2 else "3" if count == 3 else "4+"
        buckets[label].append(position.excess_return)
    order = ["1 buyer", "2", "3", "4+"]
    return [
        {
            "buyers": label,
            "positions": len(buckets[label]),
            "mean_excess": mean(buckets[label]),
            "beat_rate": sum(1 for value in buckets[label] if value > 0) / len(buckets[label]),
        }
        for label in order
        if len(buckets.get(label, [])) >= 8
    ]


def _by_opposition(
    result: BacktestResult, context: _ClusterIndex | None
) -> list[dict[str, Any]]:
    """What happens when other members are selling the same name.

    Conviction multiplies down by up to 45% when members went the other way,
    on the reasoning that it is "real counter-evidence". That is the damper
    most likely to explain why the blend inverts: a top-ranked buyer with
    sellers against them is pushed into a low quartile, and the low quartiles
    are the ones performing.
    """
    if context is None:
        return []
    buckets: dict[str, list[float]] = defaultdict(list)
    for position in result.positions:
        sellers = context.sellers(position.symbol, position.disclosure_date)
        label = "nobody selling" if sellers == 0 else "1 seller" if sellers == 1 else "2+ selling"
        buckets[label].append(position.excess_return)
    return [
        {
            "opposition": label,
            "positions": len(buckets[label]),
            "mean_excess": mean(buckets[label]),
            "beat_rate": sum(1 for value in buckets[label] if value > 0) / len(buckets[label]),
        }
        for label in ("nobody selling", "1 seller", "2+ selling")
        if len(buckets.get(label, [])) >= 8
    ]


def _by_rank(result: BacktestResult) -> list[dict[str, Any]]:
    """Whether the ranking itself carried information.

    If rank 1 does no better than rank 10, the scoring is noise, however good
    the headline number looks.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for position in result.positions:
        rank = position.rank_at_entry
        bucket = "1-3" if rank <= 3 else "4-6" if rank <= 6 else "7+"
        buckets[bucket].append(position.excess_return)
    return [
        {
            "rank": bucket,
            "positions": len(values),
            "mean_excess": mean(values),
            "beat_rate": sum(1 for value in values if value > 0) / len(values),
        }
        for bucket, values in sorted(buckets.items())
        if values
    ]


def _top_members(result: BacktestResult, limit: int = 10) -> list[dict[str, Any]]:
    by_member: dict[str, list[FollowedTrade]] = defaultdict(list)
    for position in result.positions:
        by_member[position.politician_id].append(position)
    scored: list[tuple[float, dict[str, Any]]] = []
    for politician_id, positions in by_member.items():
        excess = mean(position.excess_return for position in positions)
        scored.append(
            (
                excess,
                {
                    "politician_id": politician_id,
                    "name": positions[0].member_name,
                    "positions": len(positions),
                    "mean_excess": excess,
                    "beat_rate": sum(
                        1 for position in positions if position.excess_return > 0
                    )
                    / len(positions),
                },
            )
        )
    scored.sort(key=lambda row: row[0], reverse=True)
    return [row for _, row in scored[:limit]]
