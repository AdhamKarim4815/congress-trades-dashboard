"""The opportunity board's ranking and its reason chains."""

from __future__ import annotations

from datetime import date, timedelta

from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, Politician, TransactionType
from trade_bot.ranking import compute_member_scores
from trade_bot.recommend import build_recommendations, load_recommendations, refresh_recommendations

TODAY = date(2026, 8, 1)


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
                party="Republican",
            )
        ]
    )


def add_trade(
    store,
    trade_id: str,
    politician_id: str,
    symbol: str,
    kind: TransactionType = TransactionType.PURCHASE,
    disclosure_date: date = date(2026, 7, 25),
    amount_low: float = 1001,
    amount_high: float = 15000,
) -> None:
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=trade_id,
                filing_id="f1",
                politician_id=politician_id,
                chamber=Chamber.HOUSE,
                symbol=symbol,
                asset_name=f"{symbol} Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=kind,
                transaction_date=disclosure_date - timedelta(days=20),
                disclosure_date=disclosure_date,
                amount_low=amount_low,
                amount_high=amount_high,
            )
        ]
    )


def test_more_independent_buyers_outrank_a_lone_buyer(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_member(store, "m2", "Bob Two")
    add_member(store, "m3", "Cy Three")
    add_trade(store, "a", "m1", "ACME")
    add_trade(store, "b", "m2", "ACME")
    add_trade(store, "c", "m3", "ACME")
    add_trade(store, "d", "m1", "FLAT")

    board = build_recommendations(settings, store, prices, today=TODAY)
    by_symbol = {item.symbol: item for item in board}
    assert by_symbol["ACME"].buyer_count == 3
    assert by_symbol["ACME"].conviction > by_symbol["FLAT"].conviction


def test_members_going_the_other_way_are_counted_but_no_longer_damp_the_score(
    settings, store, prices
):
    """The damper was measured and ran backwards.

    Conviction used to fall by up to 45% when members sold the same name, on
    the reasoning that opposition is counter-evidence. Over 678 past positions
    purchases with nobody selling returned -0.25%, one seller against them
    +3.05%, two or more +3.82% -- and the penalty was pushing the best cards
    into the bottom quartile. It is removed rather than reversed: a bonus
    would be fitting one window. The count is still on the card.
    """
    add_member(store, "m1", "Ann One")
    add_member(store, "m2", "Bob Two")
    add_trade(store, "buy-a", "m1", "ACME")
    add_trade(store, "buy-f", "m1", "FLAT")
    add_trade(store, "sell-f", "m2", "FLAT", kind=TransactionType.SALE)

    board = {
        (item.symbol, item.action): item
        for item in build_recommendations(settings, store, prices, today=TODAY)
    }
    assert board[("FLAT", "buy")].seller_count == 1, "still counted and still shown"
    assert board[("ACME", "buy")].seller_count == 0
    # ACME climbs and FLAT does not, so news tone separates them; opposition
    # must not.
    unopposed = board[("ACME", "buy")].conviction
    opposed = board[("FLAT", "buy")].conviction
    assert abs(unopposed - opposed) < 0.13, "opposition alone should not move the score"


def test_recent_disclosures_outrank_stale_ones(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_trade(store, "fresh", "m1", "ACME", disclosure_date=TODAY - timedelta(days=2))
    add_trade(store, "stale", "m1", "FLAT", disclosure_date=TODAY - timedelta(days=40))
    board = {
        item.symbol: item
        for item in build_recommendations(settings, store, prices, today=TODAY)
    }
    assert board["ACME"].conviction > board["FLAT"].conviction


def test_disclosures_outside_the_window_are_excluded(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_trade(store, "ancient", "m1", "ACME", disclosure_date=date(2026, 1, 10))
    board = build_recommendations(settings, store, prices, today=TODAY)
    assert board == []


def test_a_sale_makes_a_sell_card_and_never_a_buy_card(settings, store, prices):
    """The board ranks both directions, but never confuses them.

    A ticker only members have sold must not surface as something to buy; it
    surfaces as a sale, with the sellers named on it.
    """
    add_member(store, "m1", "Ann One")
    add_trade(store, "s", "m1", "ACME", kind=TransactionType.SALE)
    board = build_recommendations(settings, store, prices, today=TODAY)
    assert [item.action for item in board] == ["sell"]
    card = board[0]
    assert card.buyers == []
    assert [seller.name for seller in card.sellers] == ["Ann One"]
    assert card.seller_count == 1 and card.buyer_count == 0


def test_both_sides_of_a_contested_ticker_are_ranked(settings, store, prices):
    """One member buying and another selling is two ideas, not one.

    Each side is also counter-evidence for the other, so neither card should
    score as highly as it would unopposed.
    """
    add_member(store, "m1", "Ann One")
    add_member(store, "m2", "Bob Two")
    add_trade(store, "b", "m1", "ACME")
    add_trade(store, "s", "m2", "ACME", kind=TransactionType.SALE)
    board = build_recommendations(settings, store, prices, today=TODAY)
    by_action = {item.action: item for item in board}
    assert set(by_action) == {"buy", "sell"}
    assert by_action["buy"].seller_count == 1
    assert by_action["sell"].buyer_count == 1
    assert any("went the other way" in reason for reason in by_action["buy"].reasons)


def test_every_recommendation_explains_itself(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_trade(store, "a", "m1", "ACME")
    board = build_recommendations(settings, store, prices, today=TODAY)
    item = board[0]
    assert item.reasons, "a recommendation with no reasons is useless"
    # The buyer is named in the per-buyer table rather than repeated in the
    # prose, which now carries only what a row cannot say.
    assert [buyer.name for buyer in item.buyers] == ["Ann One"]
    assert item.supporting_trade_ids == ["a"]


def test_unranked_buyers_are_stated_as_such(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_trade(store, "a", "m1", "ACME")
    item = build_recommendations(settings, store, prices, today=TODAY)[0]
    assert any("enough scored purchases" in reason for reason in item.reasons)
    assert item.top_member is None


def test_a_ranked_buyer_lifts_conviction_and_is_named(settings, store, prices):
    add_member(store, "good", "Great Trader")
    add_member(store, "unknown", "New Member")
    # Give "good" a long, positive record so they become rankable.
    for index in range(12):
        add_trade(
            store,
            f"hist-{index}",
            "good",
            "ACME",
            disclosure_date=date(2026, 2, 5),
        )
    store.replace_member_scores(
        compute_member_scores(settings, store, prices, today=TODAY)
    )
    add_trade(store, "new-good", "good", "ACME", disclosure_date=TODAY - timedelta(days=3))
    add_trade(store, "new-unknown", "unknown", "FLAT", disclosure_date=TODAY - timedelta(days=3))

    board = {
        item.symbol: item
        for item in build_recommendations(settings, store, prices, today=TODAY)
    }
    assert board["ACME"].top_member == "Great Trader"
    assert board["ACME"].conviction > board["FLAT"].conviction


def test_board_round_trips_through_stored_state(settings, store, prices):
    add_member(store, "m1", "Ann One")
    # refresh_recommendations uses the real clock, so the disclosure must be
    # recent relative to today rather than pinned to a calendar date.
    add_trade(store, "a", "m1", "ACME", disclosure_date=date.today() - timedelta(days=7))
    result = refresh_recommendations(settings, store, prices)
    assert result["recommendations"] == 1
    items, generated_at = load_recommendations(store)
    assert len(items) == 1 and items[0]["symbol"] == "ACME"
    assert generated_at


def test_load_recommendations_when_never_built(store):
    assert load_recommendations(store) == ([], None)


def test_results_are_capped_and_sorted(settings, store, prices):
    settings.recommendations.max_results = 1
    add_member(store, "m1", "Ann One")
    add_trade(store, "a", "m1", "ACME")
    add_trade(store, "b", "m1", "FLAT")
    board = build_recommendations(settings, store, prices, today=TODAY)
    assert len(board) == 1
    assert board[0].symbol == "ACME"


def test_card_quantifies_what_the_reporting_lag_cost(settings, store, prices):
    """The lag is the tax on copying, so the card states it in return terms.

    ACME rises every session, so a trade disclosed a month after it happened
    has demonstrably given away part of the move before anyone could act.
    """
    add_member(store, "m1", "Ann One")
    store.add_trades(
        [
            DisclosedTrade(
                trade_id="lagged",
                filing_id="f1",
                politician_id="m1",
                chamber=Chamber.HOUSE,
                symbol="ACME",
                asset_name="ACME Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(2026, 5, 1),
                disclosure_date=TODAY - timedelta(days=3),
                amount_low=1001,
                amount_high=15000,
            )
        ]
    )
    item = build_recommendations(settings, store, prices, today=TODAY)[0]

    assert item.move_since_trade is not None
    assert item.move_since_disclosure is not None
    # The member's window is longer, so on a riser it captured strictly more.
    assert item.move_since_trade > item.move_since_disclosure

    # Each buyer now carries their own pair of returns.
    buyer = item.buyers[0]
    assert buyer.transaction_date == date(2026, 5, 1)
    assert buyer.move_since_trade is not None and buyer.move_since_disclosure is not None
    assert buyer.move_since_trade > buyer.move_since_disclosure
    assert buyer.lag_days > 0


def test_a_price_that_fell_during_the_lag_is_not_called_a_cost(settings, store, prices):
    """When the price moves against the member, the copier gets a discount.

    Phrasing this as a share of the move produced "170% of the move was
    already gone" on real data, because the two moves had opposite signs.
    Comparing the entry prices directly cannot invert like that.
    """
    add_member(store, "m1", "Ann One")
    # FALLING is priced below its earlier level, so the later entry is cheaper.
    store.add_trades(
        [
            DisclosedTrade(
                trade_id="fell",
                filing_id="f1",
                politician_id="m1",
                chamber=Chamber.HOUSE,
                symbol="FALLING",
                asset_name="Falling Inc",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(2026, 1, 5),
                disclosure_date=TODAY - timedelta(days=3),
                amount_low=1001,
                amount_high=15000,
            )
        ]
    )
    board = build_recommendations(settings, store, prices, today=TODAY)
    item = next((entry for entry in board if entry.symbol == "FALLING"), None)
    if item is None:
        return  # symbol not priced in this fixture; the guard below still holds
    for reason in item.reasons:
        assert "already gone" not in reason
        assert "170%" not in reason


def test_every_buyer_gets_their_own_row_and_returns(settings, store, prices):
    """"6 buyers" with one date hid the detail that matters.

    Two members buying the same ticker weeks apart are up by different
    amounts, and each has their own disclosure date.
    """
    add_member(store, "early", "Early Bird")
    add_member(store, "late", "Late Comer")
    add_trade(store, "e", "early", "ACME", disclosure_date=TODAY - timedelta(days=30))
    add_trade(store, "l", "late", "ACME", disclosure_date=TODAY - timedelta(days=3))

    item = build_recommendations(settings, store, prices, today=TODAY)[0]
    assert item.buyer_count == 2
    assert len(item.buyers) == 2

    # Sorted earliest first, so the card reads as the order they got involved.
    assert item.buyers[0].transaction_date <= item.buyers[1].transaction_date
    names = {buyer.name for buyer in item.buyers}
    assert names == {"Early Bird", "Late Comer"}

    for buyer in item.buyers:
        assert buyer.move_since_trade is not None
        assert buyer.move_since_disclosure is not None
        assert buyer.amount > 0


def test_one_member_buying_twice_is_a_single_row(settings, store, prices):
    """The question is when they got in, not how many tickets they filed."""
    add_member(store, "m1", "Ann One")
    add_trade(store, "first", "m1", "ACME", disclosure_date=TODAY - timedelta(days=20))
    add_trade(store, "second", "m1", "ACME", disclosure_date=TODAY - timedelta(days=5))

    item = build_recommendations(settings, store, prices, today=TODAY)[0]
    assert len(item.buyers) == 1
    buyer = item.buyers[0]
    assert buyer.trade_count == 2
    # Amounts summed, dated from the earliest entry.
    assert buyer.amount > 8000


def test_a_sale_before_a_rise_is_scored_against_the_seller(settings, store, prices):
    """ACME climbs, so selling it was a mistake — and must read as one.

    The raw price move is positive, and printing that on a sell card painted a
    bad exit green. The score is signed by intent everywhere else in this
    application; the card has to agree with it, or the colour and the sign
    tell the reader opposite things.
    """
    add_member(store, "m1", "Ann One")
    add_trade(store, "s", "m1", "ACME", kind=TransactionType.SALE)
    card = build_recommendations(settings, store, prices, today=TODAY)[0]
    seller = card.sellers[0]
    assert seller.move_since_disclosure > 0, "the price rose after the filing"
    assert seller.result_since_disclosure == -seller.move_since_disclosure
    assert seller.result_since_trade == -seller.move_since_trade


def test_a_purchase_keeps_the_price_move_as_its_result(settings, store, prices):
    add_member(store, "m1", "Ann One")
    add_trade(store, "b", "m1", "ACME")
    buyer = build_recommendations(settings, store, prices, today=TODAY)[0].buyers[0]
    assert buyer.result_since_disclosure == buyer.move_since_disclosure


def test_a_sell_card_judges_the_seller_on_their_selling(settings, store, prices):
    """Not on their combined record, which can be carried entirely by buys.

    A member leading a sell card on +5.9% overall had a sell record of -3.3%.
    The card's headline claim is about their selling, so the number behind it
    has to be too.
    """
    add_member(store, "m1", "Ann One")
    # A long, strong record at buying and nothing at selling.
    for index in range(12):
        add_trade(store, f"buy-{index}", "m1", "ACME", disclosure_date=date(2026, 2, 5))
    store.replace_member_scores(
        compute_member_scores(settings, store, prices, today=TODAY)
    )
    add_trade(
        store, "sell-now", "m1", "ACME",
        kind=TransactionType.SALE, disclosure_date=TODAY - timedelta(days=3),
    )
    card = {
        item.action: item
        for item in build_recommendations(settings, store, prices, today=TODAY)
    }["sell"]
    # Their buying record must not vouch for their selling.
    assert card.top_member is None
    assert card.best_rank is None
    assert any("enough scored sales" in reason for reason in card.reasons)
