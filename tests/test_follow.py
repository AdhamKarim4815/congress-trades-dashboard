"""Mirror portfolios: the rules that stop a watchlist inventing gains."""

from __future__ import annotations

from datetime import date

import pytest

from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, TransactionType
from trade_bot.follow import build_portfolio, create_watchlist, mirror_watchlist


def add_trade(
    store,
    trade_id: str,
    kind: TransactionType,
    politician_id: str = "house-tx-doe-jane",
    symbol: str = "ACME",
    transaction_date: date = date(2026, 1, 5),
    disclosure_date: date = date(2026, 2, 5),
    amount_low: float = 1001,
) -> None:
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
                transaction_type=kind,
                transaction_date=transaction_date,
                disclosure_date=disclosure_date,
                amount_low=amount_low,
                amount_high=amount_low * 15,
            )
        ]
    )


def test_fills_are_priced_at_the_disclosure_date_not_the_trade_date(
    settings, store, prices, member
):
    add_trade(store, "t1", TransactionType.PURCHASE)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)

    fill = store.list_mirror_fills(watchlist.watchlist_id)[0]
    disclosure_price, _ = prices.close_on_or_after("ACME", date(2026, 2, 5))
    transaction_price, _ = prices.close_on_or_after("ACME", date(2026, 1, 5))
    assert fill["price"] == pytest.approx(disclosure_price)
    assert fill["price"] != pytest.approx(transaction_price)
    assert fill["price_basis"] == "disclosure_date"


def test_mirroring_twice_does_not_double_buy(settings, store, prices, member):
    add_trade(store, "t1", TransactionType.PURCHASE)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    first = mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    second = mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    assert first["fills_added"] == 1
    assert second["fills_added"] == 0
    assert len(store.list_mirror_fills(watchlist.watchlist_id)) == 1


def test_a_sale_of_something_never_bought_is_skipped(settings, store, prices, member):
    """Members disclose selling positions opened long before the watchlist
    existed. Mirroring those would fabricate proceeds from thin air."""
    add_trade(store, "sell-only", TransactionType.SALE)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    result = mirror_watchlist(settings, store, prices, watchlist.watchlist_id)

    assert result["fills_added"] == 0
    assert result["skipped"] == 1
    portfolio = build_portfolio(settings, store, prices, watchlist.watchlist_id)
    assert portfolio.positions == []
    assert portfolio.realized_pnl == 0.0


def test_buy_then_full_sale_closes_the_position_and_realises_profit(
    settings, store, prices, member
):
    add_trade(store, "buy", TransactionType.PURCHASE, disclosure_date=date(2026, 2, 5))
    add_trade(
        store,
        "sell",
        TransactionType.SALE,
        transaction_date=date(2026, 5, 1),
        disclosure_date=date(2026, 6, 1),
    )
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)

    portfolio = build_portfolio(settings, store, prices, watchlist.watchlist_id)
    assert portfolio.positions == []
    # ACME rises monotonically, so selling later than buying must realise a gain.
    assert portfolio.realized_pnl > 0
    assert portfolio.total_return > 0


def test_partial_sale_leaves_half_the_position(settings, store, prices, member):
    add_trade(store, "buy", TransactionType.PURCHASE)
    add_trade(
        store,
        "half",
        TransactionType.PARTIAL_SALE,
        transaction_date=date(2026, 5, 1),
        disclosure_date=date(2026, 6, 1),
    )
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)

    portfolio = build_portfolio(settings, store, prices, watchlist.watchlist_id)
    assert len(portfolio.positions) == 1
    buy_fill = next(
        fill
        for fill in store.list_mirror_fills(watchlist.watchlist_id)
        if fill["side"] == "purchase"
    )
    assert portfolio.positions[0].shares == pytest.approx(buy_fill["shares"] / 2)


def test_sells_are_ignored_when_the_watchlist_opts_out(settings, store, prices, member):
    add_trade(store, "buy", TransactionType.PURCHASE)
    add_trade(
        store, "sell", TransactionType.SALE, disclosure_date=date(2026, 6, 1)
    )
    watchlist = create_watchlist(
        settings, store, "Buys only", [member.politician_id], mirror_sells=False
    )
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    portfolio = build_portfolio(settings, store, prices, watchlist.watchlist_id)
    assert len(portfolio.positions) == 1


def test_fixed_allocation_spends_the_configured_amount(settings, store, prices, member):
    add_trade(store, "t1", TransactionType.PURCHASE)
    watchlist = create_watchlist(
        settings, store, "Jane", [member.politician_id], allocation_value=2500.0
    )
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    fill = store.list_mirror_fills(watchlist.watchlist_id)[0]
    assert fill["notional"] == pytest.approx(2500.0)
    assert fill["shares"] * fill["price"] == pytest.approx(2500.0)


def test_small_disclosures_below_the_floor_are_skipped(settings, store, prices, member):
    add_trade(store, "tiny", TransactionType.PURCHASE, amount_low=500)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    result = mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    assert result["fills_added"] == 0
    assert result["skipped"] == 1


def test_unpriceable_symbols_are_counted_not_silently_dropped(
    settings, store, prices, member
):
    add_trade(store, "ghost", TransactionType.PURCHASE, symbol="NOSUCH")
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    result = mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    assert result["unpriced"] == 1
    assert result["fills_added"] == 0


def test_portfolio_equity_reconciles_cash_and_market_value(
    settings, store, prices, member
):
    add_trade(store, "t1", TransactionType.PURCHASE)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    portfolio = build_portfolio(settings, store, prices, watchlist.watchlist_id)
    assert portfolio.equity == pytest.approx(portfolio.cash + portfolio.market_value)
    assert portfolio.total_return == pytest.approx(
        (portfolio.equity - portfolio.starting_equity) / portfolio.starting_equity
    )


def test_deleting_a_watchlist_removes_its_fills(settings, store, prices, member):
    add_trade(store, "t1", TransactionType.PURCHASE)
    watchlist = create_watchlist(settings, store, "Jane", [member.politician_id])
    mirror_watchlist(settings, store, prices, watchlist.watchlist_id)
    store.delete_watchlist(watchlist.watchlist_id)
    assert store.get_watchlist(watchlist.watchlist_id) is None
    assert store.list_mirror_fills(watchlist.watchlist_id) == []
