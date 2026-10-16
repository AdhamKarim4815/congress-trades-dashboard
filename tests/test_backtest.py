"""The walk-forward test, and the lookahead it must not have."""

from __future__ import annotations

import random
from datetime import date, timedelta

from trade_bot.backtest import (
    PAYLOAD_KEYS,
    BacktestResult,
    FollowedTrade,
    _by_conviction,
    _control_summary,
    _coverage,
    _exit_test,
    _Pricer,
    _random_selector,
    _rank_members_as_of,
    _Rebalance,
    bootstrap_interval,
    coin_flip_probability,
    conviction_as_of,
    refresh_backtest,
    run_backtest,
)
from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, Politician, TransactionType


def add_member(store, politician_id: str, name: str) -> None:
    store.upsert_politicians(
        [
            Politician(
                politician_id=politician_id,
                full_name=name,
                first_name=name.split()[0],
                last_name=name.split()[-1],
                chamber=Chamber.HOUSE,
                state="TX",
            )
        ]
    )


def add_trade(store, trade_id, politician_id, symbol, disclosure_date):
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=trade_id,
                filing_id="f1",
                politician_id=politician_id,
                chamber=Chamber.HOUSE,
                symbol=symbol,
                asset_name=symbol,
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=disclosure_date - timedelta(days=20),
                disclosure_date=disclosure_date,
                amount_low=1001,
                amount_high=15000,
            )
        ]
    )


def test_ranking_as_of_ignores_trades_whose_horizon_has_not_elapsed(settings, store, prices):
    """The whole test is worthless if a ranking can see its own future.

    A trade disclosed last week cannot have a known 90-day outcome, so it
    must not contribute to a ranking made today.
    """
    add_member(store, "m1", "Ann One")
    as_of = date(2026, 6, 1)
    # Resolved well before as_of.
    for index in range(9):
        add_trade(store, f"old-{index}", "m1", "ACME", date(2026, 1, 5))
    # Disclosed days ago: its 90-day outcome is unknowable at as_of.
    add_trade(store, "fresh", "m1", "ACME", as_of - timedelta(days=5))

    trades = store.list_trades(actionable_only=True, limit=1000)
    ranked = _rank_members_as_of(
        trades, _Pricer(prices, "SPY"), as_of=as_of, horizon_days=90,
        lookback_days=1095, min_trades=1, shrinkage_prior=8.0,
    )
    assert ranked, "the resolved trades should still produce a ranking"
    _, _, sample = ranked[0]
    assert sample == 9, "the unresolved trade must be excluded"


def test_a_member_with_too_few_resolved_trades_is_not_selected(settings, store, prices):
    add_member(store, "thin", "Thin Record")
    add_trade(store, "one", "thin", "ACME", date(2026, 1, 5))
    trades = store.list_trades(actionable_only=True, limit=1000)
    ranked = _rank_members_as_of(
        trades, _Pricer(prices, "SPY"), as_of=date(2026, 6, 1), horizon_days=90,
        lookback_days=1095, min_trades=8, shrinkage_prior=8.0,
    )
    assert ranked == []


def test_backtest_enters_at_the_disclosure_price(settings, store, prices):
    """Entries must use the price a copier could pay, never the member's."""
    add_member(store, "m1", "Ann One")
    for index in range(10):
        add_trade(store, f"hist-{index}", "m1", "ACME", date(2026, 1, 5))
    add_trade(store, "followed", "m1", "ACME", date(2026, 5, 1))

    result = run_backtest(
        settings, store, prices, top_n=3, years=2, min_trades=5, today=date(2026, 8, 1)
    )
    for position in result.positions:
        expected, _ = prices.close_on_or_after(position.symbol, position.disclosure_date)
        assert position.entry_price == expected


def test_backtest_with_no_data_returns_an_empty_result(settings, store, prices):
    result = run_backtest(settings, store, prices, today=date(2026, 8, 1))
    assert result.positions == []
    assert result.summary()["positions"] == 0


def test_summary_reports_median_alongside_mean(settings, store, prices):
    """Mean alone hid that the typical trade lost to the index.

    The top-10 configuration averaged +0.45% excess while its median was
    -1.14%; without the median the strategy looked like it worked.
    """
    add_member(store, "m1", "Ann One")
    for index in range(10):
        add_trade(store, f"hist-{index}", "m1", "ACME", date(2026, 1, 5))
    add_trade(store, "followed", "m1", "ACME", date(2026, 5, 1))
    summary = run_backtest(
        settings, store, prices, top_n=3, years=2, min_trades=5, today=date(2026, 8, 1)
    ).summary()
    if summary["positions"]:
        assert "median_excess_return" in summary
        assert "mean_excess_return" in summary
        assert "beat_benchmark_rate" in summary


def test_the_measurement_is_reused_rather_than_recomputed_daily(settings, store, prices):
    """Two walk-forward runs are the most expensive stage in the pipeline.

    What they measure moves over years, so recomputing twice a day bought
    nothing and cost everything: the scheduled job ran past its time limit,
    and at that duration twice daily it would also have exceeded the free
    monthly minute allowance.
    """
    # A *complete* payload: recent is not enough on its own, because a cache
    # written before a measurement existed would otherwise be served with the
    # new sections silently missing.
    complete = {key: {} for key in PAYLOAD_KEYS}
    complete["summary"] = {"positions": 42}
    complete["computed_at"] = date(2026, 8, 1).isoformat()
    store.set_state("board.backtest", complete)
    reused = refresh_backtest(
        settings, store, prices, today=date(2026, 8, 5), max_age_days=7
    )
    assert reused["positions"] == 42, "a fresh measurement should be reused"


def test_a_stale_measurement_is_recomputed(settings, store, prices):
    from datetime import date

    from trade_bot.backtest import refresh_backtest

    store.set_state(
        "board.backtest",
        {"summary": {"positions": 42}, "computed_at": date(2026, 7, 1).isoformat()},
    )
    recomputed = refresh_backtest(
        settings, store, prices, today=date(2026, 8, 5), max_age_days=7
    )
    assert recomputed.get("positions") != 42, "a stale measurement should be replaced"


# -- saying how much of a measurement this is ------------------------------


def test_an_interval_is_reproducible_and_widens_on_thin_evidence():
    """Stored and rendered, so it must not move when the data has not."""
    steady = [0.05, 0.04, 0.06, 0.05, 0.05, 0.04, 0.06, 0.05, 0.05, 0.05]
    wild = [0.5, -0.4, 0.6, -0.5, 0.4, -0.3, 0.5, -0.45, 0.35, -0.25]
    assert bootstrap_interval(steady) == bootstrap_interval(steady)
    tight_low, tight_high = bootstrap_interval(steady)
    wide_low, wide_high = bootstrap_interval(wild)
    assert (wide_high - wide_low) > (tight_high - tight_low)
    assert tight_low > 0, "ten consistent wins should clear zero"
    assert wide_low < 0 < wide_high, "and a coin-toss spread should not"


def test_too_few_points_get_no_interval_rather_than_a_fake_one():
    assert bootstrap_interval([0.1, 0.2, 0.3]) is None


def test_a_beat_rate_is_measured_against_a_coin():
    """55% over 188 is what a fair coin does about one time in five."""
    assert coin_flip_probability(103, 188) > 0.05
    assert coin_flip_probability(160, 188) < 0.001
    assert coin_flip_probability(94, 188) == 1.0


def test_a_result_whose_interval_contains_zero_is_reported_inconclusive(settings, store, prices):
    """"Inconclusive" is a result, and the summary has to be able to say it."""
    result = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    for index, excess in enumerate([0.4, -0.35, 0.3, -0.4, 0.45, -0.3, 0.25, -0.35, 0.2, -0.2]):
        result.positions.append(
            FollowedTrade(
                symbol="ACME",
                politician_id="m1",
                member_name="Ann One",
                disclosure_date=date(2026, 1, 5 + index),
                entry_price=100.0,
                exit_price=100.0 * (1 + excess),
                trade_return=excess,
                benchmark_return=0.0,
                excess_return=excess,
                rank_at_entry=1,
            )
        )
    summary = result.summary()
    assert summary["excess_ci_low"] < 0 < summary["excess_ci_high"]
    assert summary["conclusive"] is False


def test_the_control_places_the_real_result_among_random_ones():
    beaten_all = _control_summary([-0.02, 0.0, 0.01], actual=0.05)
    assert beaten_all["percentile"] == 1.0
    assert beaten_all["beats_chance"] is True

    middling = _control_summary([-0.02, 0.0, 0.01, 0.06], actual=0.005)
    assert middling["percentile"] == 0.5
    assert middling["beats_chance"] is False, "beating half the coin flips is not skill"


def test_the_control_selector_ignores_rank_but_keeps_the_eligibility_bar():
    """It removes the ranking and nothing else.

    Competing against members who could never have been picked would make the
    control easy to beat and the comparison meaningless.
    """
    rebalance = _Rebalance(
        at=date(2026, 1, 5),
        ranked=[(f"m{index}", 0.1 - index * 0.01, 10) for index in range(8)],
        candidates=[],
    )
    picked = _random_selector(3, random.Random(1))(rebalance)
    assert len(picked) == 3
    assert set(picked) <= {f"m{index}" for index in range(8)}
    # A different seed reaches a different set, or the control is not random.
    others = [
        frozenset(_random_selector(3, random.Random(seed))(rebalance)) for seed in range(12)
    ]
    assert len(set(others)) > 1


def test_pricing_a_trade_twice_asks_the_repository_once(settings, store, prices):
    """The cache is what pays for the control and the extra horizons."""
    add_member(store, "m1", "Ann One")
    add_trade(store, "a", "m1", "ACME", date(2026, 1, 5))
    trade = store.list_trades(actionable_only=True, limit=10)[0]
    pricer = _Pricer(prices, "SPY")
    first = pricer.price(trade, 90, date(2026, 8, 1))
    second = pricer.price(trade, 90, date(2026, 8, 1))
    assert first is second
    assert (pricer.hits, pricer.misses) == (1, 1)


def test_coverage_reports_the_share_dropped_and_its_year_skew(settings, store, prices):
    """The skew is the tell, not the total.

    7% of trades unpriceable sounds tolerable until you see that it is 15% of
    the oldest year and 3% of the newest: the further back a disclosure sits,
    the likelier its company has since disappeared and taken its prices with
    it, so the oldest part of the window is measured only on survivors.
    """
    add_member(store, "m1", "Ann One")
    add_trade(store, "old-priced", "m1", "ACME", date(2024, 1, 8))
    add_trade(store, "old-gone", "m1", "GONE", date(2024, 1, 9))
    add_trade(store, "new-priced", "m1", "ACME", date(2026, 1, 8))

    trades = store.list_trades(actionable_only=True, limit=100)
    coverage = _coverage(trades, prices)
    assert coverage["actionable_trades"] == 3
    assert coverage["unpriced_trades"] == 1
    assert coverage["unpriced_symbols"] == 1
    by_year = {row["year"]: row for row in coverage["by_year"]}
    assert by_year[2024]["unpriced_share"] == 0.5
    assert by_year[2026]["unpriced_share"] == 0.0


# -- measuring the score on every card -------------------------------------


def _position(excess: float, conviction: float | None = None, buyers: int | None = None):
    return FollowedTrade(
        symbol="ACME",
        politician_id="m1",
        member_name="Ann One",
        disclosure_date=date(2026, 1, 5),
        entry_price=100.0,
        exit_price=100.0 * (1 + excess),
        trade_return=excess,
        benchmark_return=0.0,
        excess_return=excess,
        rank_at_entry=1,
        conviction_at_entry=conviction,
        buyers_at_entry=buyers,
    )


def test_conviction_buckets_report_whether_the_score_orders_outcomes():
    """The number on every card, finally asked to justify its bar."""
    ordered = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    for index in range(40):
        # Conviction rises with the outcome: the score works.
        ordered.positions.append(_position(index * 0.01, conviction=index * 0.02))
    report = _by_conviction(ordered)
    assert report["monotone"] is True
    assert report["top_beats_bottom"] is True
    assert report["spread"] > 0

    inverted = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    for index in range(40):
        inverted.positions.append(_position(-index * 0.01, conviction=index * 0.02))
    upside_down = _by_conviction(inverted)
    assert upside_down["monotone"] is False
    assert upside_down["top_beats_bottom"] is False, "the highest scores did worst"


def test_conviction_names_what_it_could_not_reconstruct():
    """A score measured over four of six parts must say which four."""
    result = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    for index in range(40):
        result.positions.append(_position(index * 0.01, conviction=index * 0.02))
    report = _by_conviction(result)
    assert "news sentiment" in report["excluded"]
    assert "recency" in report["constant"]


def test_the_reconstruction_scores_the_formula_the_board_actually_ships(settings):
    """One formula, not two.

    They were separate copies for one afternoon, and in that time removing a
    damper from the board left this measurement grading the old arithmetic and
    reporting that nothing had changed.
    """
    from trade_bot.recommend import damp_unproven, structural_conviction

    expected = damp_unproven(
        settings,
        structural_conviction(
            settings, 1, 3, 50_000, 0, settings.recommendations.disclosure_window_days
        ),
        1,
    )
    assert conviction_as_of(settings, rank=1, buyers=3, dollars=50_000, sellers=0) == expected
    # Opposition no longer moves it; the measurement said the penalty ran the
    # wrong way, so the board stopped applying one.
    assert conviction_as_of(settings, rank=1, buyers=3, dollars=50_000, sellers=4) == expected
    # An unranked buyer is still damped below a top-ranked one.
    assert conviction_as_of(settings, rank=None, buyers=3, dollars=50_000, sellers=0) < expected


def test_the_exit_test_compares_both_legs_over_the_same_window(settings, store, prices):
    """ACME climbs, so exiting early on their sale should cost excess return.

    The comparison is only fair because the proceeds go into the benchmark:
    both legs are excess over the full horizon and the exited one simply stops
    accruing. Measuring a 40-day excess against a 90-day one would hand the
    result to whichever leg was invested longer.
    """
    add_member(store, "m1", "Ann One")
    result = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    entry, _ = prices.close_on_or_after("ACME", date(2026, 2, 2))
    for _index in range(10):
        result.positions.append(
            FollowedTrade(
                symbol="ACME",
                politician_id="m1",
                member_name="Ann One",
                disclosure_date=date(2026, 2, 2),
                entry_price=entry,
                exit_price=entry * 1.2,
                trade_return=0.2,
                benchmark_return=0.0,
                excess_return=0.2,
                rank_at_entry=1,
            )
        )
    sales = [
        {
            "trade_id": f"s{index}",
            "politician_id": "m1",
            "symbol": "ACME",
            "transaction_type": "sale",
            "disclosure_date": date(2026, 3, 2),
        }
        for index in range(10)
    ]
    report = _exit_test(result, sales, prices, "SPY", 90)
    assert report["positions"] == 10
    assert report["exited_mean_excess"] < report["held_mean_excess"]
    assert report["difference"] < 0, "leaving a riser early gives up excess return"


def test_the_exit_test_says_so_when_there_is_nothing_to_measure(settings, store, prices):
    result = BacktestResult(horizon_days=90, top_n=3, rebalance_days=30)
    result.positions.append(_position(0.1))
    assert _exit_test(result, [], prices, "SPY", 90)["positions"] == 0


def test_a_cache_written_before_a_measurement_existed_is_recomputed(settings, store, prices):
    """Recent is not the same as current.

    Every page guards on these keys and renders nothing when they are absent,
    so an age check alone would have shipped a new evidence section invisibly
    until the cache aged out a week later, with no error to say why.
    """
    add_member(store, "m1", "Ann One")
    add_trade(store, "a", "m1", "ACME", date(2026, 1, 5))
    store.set_state(
        "board.backtest",
        {"summary": {"positions": 7}, "computed_at": date(2026, 8, 1).isoformat()},
    )
    refresh_backtest(settings, store, prices, max_age_days=7, today=date(2026, 8, 2))
    stored = store.get_state("board.backtest") or {}
    assert stored["summary"] != {"positions": 7}, "the stale shape should not survive"
    assert set(stored) >= PAYLOAD_KEYS

