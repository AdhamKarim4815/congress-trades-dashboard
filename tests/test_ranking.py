"""Member scoring, and the shrinkage that keeps small samples honest."""

from __future__ import annotations

from datetime import date

import pytest

from trade_bot.domain import (
    AssetClass,
    Chamber,
    DisclosedTrade,
    Politician,
    TradeSide,
    TransactionType,
)
from trade_bot.ranking import compute_member_scores, shrink


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


def add_trades(
    store,
    politician_id: str,
    count: int,
    symbol: str = "ACME",
    kind: TransactionType = TransactionType.PURCHASE,
    prefix: str = "b",
) -> None:
    trades = []
    for index in range(count):
        trades.append(
            DisclosedTrade(
                trade_id=f"{politician_id}-{symbol}-{prefix}{index}",
                filing_id="f1",
                politician_id=politician_id,
                chamber=Chamber.HOUSE,
                symbol=symbol,
                asset_name=symbol,
                asset_class=AssetClass.STOCK,
                transaction_type=kind,
                transaction_date=date(2026, 1, 5),
                disclosure_date=date(2026, 2, 5),
                amount_low=1001,
                amount_high=15000,
            )
        )
    store.add_trades(trades)


def test_shrinkage_pulls_small_samples_toward_zero():
    # Same raw edge, wildly different evidence behind it.
    assert shrink(0.10, 2, 8.0) < shrink(0.10, 40, 8.0) < 0.10
    assert shrink(0.10, 2, 8.0) == pytest.approx(0.10 * 2 / 10)


def test_shrinkage_of_no_trades_is_zero():
    assert shrink(0.5, 0, 8.0) == 0.0


def test_a_lucky_few_trades_do_not_outrank_a_long_record(settings, store, prices):
    """Both members bought the same riser, so their raw edge is identical --
    only sample size separates them, and the ranking must reflect that."""
    add_member(store, "house-tx-lucky-sam", "Sam Lucky")
    add_member(store, "house-tx-steady-pat", "Pat Steady")
    add_trades(store, "house-tx-lucky-sam", 2)
    add_trades(store, "house-tx-steady-pat", 40)

    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    horizon = settings.ranking.primary_horizon_days
    by_member = {
        score.politician_id: score for score in scores if score.horizon_days == horizon
    }
    lucky = by_member["house-tx-lucky-sam"]
    steady = by_member["house-tx-steady-pat"]

    assert lucky.raw_excess_return == pytest.approx(steady.raw_excess_return)
    assert steady.score > lucky.score
    assert not lucky.rankable  # below min_trades_to_rank
    assert steady.rankable


def test_scores_are_written_for_every_configured_horizon(settings, store, prices):
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 6)
    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    horizons = {score.horizon_days for score in scores}
    assert set(settings.ranking.horizons_days).issubset(horizons)


def test_members_with_only_unpriceable_trades_are_omitted(settings, store, prices):
    add_member(store, "house-tx-ghost-lee", "Lee Ghost")
    add_trades(store, "house-tx-ghost-lee", 5, symbol="NOSUCH")
    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    assert all(score.politician_id != "house-tx-ghost-lee" for score in scores)


def test_scores_round_trip_through_the_store(settings, store, prices):
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 10)
    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    store.replace_member_scores(scores)
    horizon = settings.ranking.primary_horizon_days
    stored = store.get_member_score("house-tx-doe-jane", horizon)
    assert stored is not None
    assert stored["trades_scored"] == 10
    assert stored["rankable"] is True


def test_replacing_scores_does_not_accumulate_duplicates(settings, store, prices):
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 10)
    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    store.replace_member_scores(scores)
    store.replace_member_scores(scores)
    # `list_member_scores` returns one side, so compare against that side.
    overall = [score for score in scores if score.side is TradeSide.ALL]
    assert len(store.list_member_scores()) == len(overall)


def test_a_member_is_scored_three_times_and_the_sides_disagree(settings, store, prices):
    """Buying, selling, and both — because they answer different questions.

    One combined number put a member second on the "who to copy" page off the
    back of twenty-five sales and no purchases at all, beneath a backtest that
    only ever buys. A member now has one record per side and the caller has to
    say which it is asking about.
    """
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 6)
    add_trades(store, "house-tx-doe-jane", 6, kind=TransactionType.SALE, prefix="s")
    scores = compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    store.replace_member_scores(scores)

    horizon = settings.ranking.primary_horizon_days
    sides = store.member_score_sides("house-tx-doe-jane", horizon)
    assert set(sides) == {"all", "buy", "sell"}
    assert sides["all"]["trades_scored"] == (
        sides["buy"]["trades_scored"] + sides["sell"]["trades_scored"]
    )
    # ACME climbs, so buying it beat the index and selling it did not. The
    # combined figure sits between the two and describes neither.
    assert sides["buy"]["score"] > 0 > sides["sell"]["score"]
    assert sides["sell"]["score"] < sides["all"]["score"] < sides["buy"]["score"]


def test_a_member_who_only_sells_has_no_buy_record_at_all(settings, store, prices):
    """Not a zero — absent. A zero would rank; nothing to copy must not."""
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 6, kind=TransactionType.SALE)
    store.replace_member_scores(
        compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    )
    horizon = settings.ranking.primary_horizon_days
    assert store.get_member_score("house-tx-doe-jane", horizon, "buy") is None
    assert store.get_member_score("house-tx-doe-jane", horizon, "sell") is not None


def test_the_best_call_records_which_way_they_traded_it(settings, store, prices):
    """A +36% best call that was a sale reads as a pick that rose 36%."""
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 6, kind=TransactionType.SALE)
    store.replace_member_scores(
        compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    )
    horizon = settings.ranking.primary_horizon_days
    score = store.get_member_score("house-tx-doe-jane", horizon, "sell")
    assert score["best_symbol_side"] == "sell"


def test_a_trade_nobody_can_price_is_counted_rather_than_dropped(settings, store, prices):
    """Survivorship, made visible on the member it belongs to.

    A company since acquired or delisted has no series left to fetch, so its
    trades vanish from every figure. Counting them is the difference between
    a record with a known gap and a record that quietly flatters itself.
    """
    add_member(store, "house-tx-doe-jane", "Jane Doe")
    add_trades(store, "house-tx-doe-jane", 6, symbol="ACME")
    add_trades(store, "house-tx-doe-jane", 3, symbol="GONE", prefix="g")
    store.replace_member_scores(
        compute_member_scores(settings, store, prices, today=date(2026, 8, 1))
    )
    horizon = settings.ranking.primary_horizon_days
    score = store.get_member_score("house-tx-doe-jane", horizon, "buy")
    assert score["trades_scored"] == 6, "only the priceable ones are scored"
    assert score["trades_unpriced"] == 3, "and the rest are counted, not forgotten"
