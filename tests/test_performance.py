"""Return measurement, and the member/copier distinction it exists to protect."""

from __future__ import annotations

from datetime import date

import pytest

from trade_bot.performance import evaluate_trade, summarize


def make_trade(**overrides):
    trade = {
        "trade_id": "t1",
        "politician_id": "house-tx-doe-jane",
        "symbol": "ACME",
        "transaction_type": "purchase",
        "transaction_date": date(2026, 1, 5),
        "disclosure_date": date(2026, 2, 5),
        "amount_midpoint": 8000.0,
    }
    trade.update(overrides)
    return trade


def test_member_return_beats_copier_return_when_price_rose_during_the_lag(prices):
    """ACME climbs every session, so a month of reporting lag is a month of
    gains the member captured and a copier could not."""
    result = evaluate_trade(make_trade(), prices, "SPY", horizon_days=90, today=date(2026, 8, 1))
    assert result.priced
    assert result.member_return is not None and result.achievable_return is not None
    assert result.member_return > result.achievable_return
    assert result.entry_price < result.disclosure_price


def test_excess_return_is_net_of_the_benchmark(prices):
    """SPY is flat in the fixture, so excess equals the achievable return."""
    result = evaluate_trade(make_trade(), prices, "SPY", horizon_days=90, today=date(2026, 8, 1))
    assert result.benchmark_return == pytest.approx(0.0, abs=1e-9)
    assert result.excess_return == pytest.approx(result.achievable_return)


def test_a_sale_that_dodged_a_gain_scores_negative(prices):
    """Selling is scored by intent: shedding a riser is a bad decision, so its
    excess is the negation of the price move."""
    buy = evaluate_trade(make_trade(), prices, "SPY", horizon_days=90, today=date(2026, 8, 1))
    sell = evaluate_trade(
        make_trade(transaction_type="sale"), prices, "SPY", horizon_days=90, today=date(2026, 8, 1)
    )
    assert buy.excess_return > 0
    assert sell.excess_return == pytest.approx(-buy.excess_return)


def test_flat_stock_against_flat_benchmark_is_zero_excess(prices):
    result = evaluate_trade(
        make_trade(symbol="FLAT"), prices, "SPY", horizon_days=90, today=date(2026, 8, 1)
    )
    assert result.excess_return == pytest.approx(0.0, abs=1e-9)


def test_unpriced_symbol_reports_rather_than_inventing_zero(prices):
    result = evaluate_trade(
        make_trade(symbol="NOSUCH"), prices, "SPY", horizon_days=90, today=date(2026, 8, 1)
    )
    assert not result.priced
    assert result.excess_return is None
    assert "no price history" in (result.note or "")


def test_line_without_a_ticker_is_not_priced(prices):
    result = evaluate_trade(make_trade(symbol=None), prices, "SPY", today=date(2026, 8, 1))
    assert not result.priced
    assert "no ticker" in (result.note or "")


def test_unelapsed_horizon_is_measured_to_today_and_flagged(prices):
    result = evaluate_trade(
        make_trade(disclosure_date=date(2026, 6, 1)),
        prices,
        "SPY",
        horizon_days=365,
        today=date(2026, 7, 1),
    )
    assert result.priced
    assert "elapsed" in (result.note or "")


def test_weekend_disclosure_prices_at_the_next_session(prices):
    """2026-01-10 is a Saturday; the copier's price is Monday's close."""
    saturday = evaluate_trade(
        make_trade(disclosure_date=date(2026, 1, 10)),
        prices,
        "SPY",
        horizon_days=90,
        today=date(2026, 8, 1),
    )
    monday = evaluate_trade(
        make_trade(disclosure_date=date(2026, 1, 12)),
        prices,
        "SPY",
        horizon_days=90,
        today=date(2026, 8, 1),
    )
    assert saturday.disclosure_price == monday.disclosure_price


def test_summarize_ignores_unpriced_trades(prices):
    priced = evaluate_trade(make_trade(), prices, "SPY", horizon_days=90, today=date(2026, 8, 1))
    unpriced = evaluate_trade(
        make_trade(trade_id="t2", symbol="NOSUCH"), prices, "SPY", today=date(2026, 8, 1)
    )
    summary = summarize([priced, unpriced])
    assert summary["priced"] == 1
    assert summary["win_rate"] == 1.0


def test_summarize_of_nothing_is_zeroed_not_an_error():
    summary = summarize([])
    assert summary["trades"] == 0 and summary["mean_excess"] == 0.0


def test_a_sale_before_a_rise_reads_negative(prices, settings):
    """The verdict on a trade is signed by intent, never by the price.

    A member who sold before a 12% rise did not make 12%; they gave it up.
    `achievable_return` keeps the price move, because "the stock rose" is
    also true and worth showing — but it is not the verdict.
    """
    trade = {
        "trade_id": "s1",
        "politician_id": "m1",
        "symbol": "ACME",
        "transaction_type": "sale",
        "transaction_date": date(2026, 1, 6),
        "disclosure_date": date(2026, 2, 2),
        "amount_midpoint": 8000.0,
    }
    result = evaluate_trade(trade, prices, settings.prices.benchmark_symbol)
    assert result.achievable_return > 0, "ACME climbs, so the price rose after the filing"
    assert result.result_return == -result.achievable_return
    assert result.excess_return < 0, "and the benchmark-relative score agrees"


def test_a_purchase_keeps_the_price_move_as_its_verdict(prices, settings):
    trade = {
        "trade_id": "b1",
        "politician_id": "m1",
        "symbol": "ACME",
        "transaction_type": "purchase",
        "transaction_date": date(2026, 1, 6),
        "disclosure_date": date(2026, 2, 2),
        "amount_midpoint": 8000.0,
    }
    result = evaluate_trade(trade, prices, settings.prices.benchmark_symbol)
    assert result.result_return == result.achievable_return > 0
