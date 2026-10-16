"""Running positions built out of bracketed disclosures.

The arithmetic is easy; the honesty is the hard part. Congress reports a range
and never a share count, so these tests are mostly about the ledger refusing
to claim more than the filings support.
"""

from __future__ import annotations

from datetime import date

from trade_bot.ledger import accumulation_summary, build_positions

MEMBERS = {"m1": {"full_name": "Ann One"}, "m2": {"full_name": "Bob Two"}}


def _trade(
    trade_id: str,
    politician_id: str = "m1",
    symbol: str = "ACME",
    kind: str = "purchase",
    traded: date = date(2026, 1, 6),
    filed: date = date(2026, 2, 2),
    low: float = 1001.0,
    high: float = 15000.0,
) -> dict:
    return {
        "trade_id": trade_id,
        "politician_id": politician_id,
        "symbol": symbol,
        "transaction_type": kind,
        "transaction_date": traded,
        "disclosure_date": filed,
        "amount_low": low,
        "amount_high": high,
        "amount_midpoint": (low + high) / 2,
    }


def test_repeated_buying_reads_as_accumulation():
    positions = build_positions(
        [
            _trade("a", traded=date(2026, 1, 6), filed=date(2026, 2, 2)),
            _trade("b", traded=date(2026, 2, 10), filed=date(2026, 3, 2)),
            _trade("c", traded=date(2026, 3, 9), filed=date(2026, 4, 1)),
        ],
        MEMBERS,
    )
    position = positions[0]
    assert position.direction == "accumulating"
    assert position.buy_count == 3 and position.sell_count == 0
    assert position.net_exposure == 3 * 8000.5
    assert position.first_seen == date(2026, 2, 2)
    assert position.days_open == 58


def test_a_single_purchase_is_held_not_accumulating():
    """One trade is not a trend, and the word should not imply one."""
    positions = build_positions([_trade("a")], MEMBERS)
    assert positions[0].direction == "held"


def test_a_member_with_only_sales_is_selling_down_not_trimming():
    """"Trimming" claims we watched them build the position they are leaving.

    For a member whose every disclosure is a sale, we did not. They are
    reducing a holding that predates anything on record.
    """
    positions = build_positions([_trade("a", kind="sale")], MEMBERS)
    assert positions[0].direction == "selling_down"


def test_a_round_trip_is_never_reported_as_closed():
    """`closed` was unreachable: both counts are non-zero by the size test."""
    positions = build_positions(
        [_trade("a"), _trade("b", kind="sale", traded=date(2026, 2, 2))], MEMBERS
    )
    assert positions[0].direction == "round_trip"


def test_selling_after_buying_reads_as_trimming():
    positions = build_positions(
        [
            _trade("a", traded=date(2026, 1, 6), filed=date(2026, 2, 2)),
            _trade("b", traded=date(2026, 1, 7), filed=date(2026, 2, 3)),
            _trade("c", kind="sale", traded=date(2026, 3, 9), filed=date(2026, 4, 1)),
        ],
        MEMBERS,
    )
    assert positions[0].direction == "trimming"


def test_bounds_are_the_bracket_walk_not_the_midpoint():
    """The headline number is an estimate; the range is what the filing said.

    Buying band A and selling band A nets to zero at the midpoint, but the
    filings only bound it between -$14k and +$14k, and saying so is the
    difference between a measurement and a guess.
    """
    positions = build_positions(
        [
            _trade("a", low=1001.0, high=15000.0),
            _trade("b", kind="sale", low=1001.0, high=15000.0, traded=date(2026, 2, 2)),
        ],
        MEMBERS,
    )
    position = positions[0]
    assert position.net_exposure == 0
    assert position.net_low == 1001.0 - 15000.0
    assert position.net_high == 15000.0 - 1001.0


def test_an_exchange_is_recorded_nowhere_rather_than_guessed_at():
    """An exchange moves exposure between two names and sizes neither leg."""
    assert build_positions([_trade("a", kind="exchange")], MEMBERS) == []


def test_positions_are_kept_apart_by_member_and_symbol():
    positions = build_positions(
        [
            _trade("a", politician_id="m1", symbol="ACME"),
            _trade("b", politician_id="m2", symbol="ACME"),
            _trade("c", politician_id="m1", symbol="FLAT"),
        ],
        MEMBERS,
    )
    assert len(positions) == 3
    assert {(p.politician_id, p.symbol) for p in positions} == {
        ("m1", "ACME"),
        ("m2", "ACME"),
        ("m1", "FLAT"),
    }


def test_the_copier_basis_uses_the_disclosure_price(prices):
    """Never the member's own entry, which was not available to anyone else."""
    positions = build_positions(
        [_trade("a", symbol="ACME", traded=date(2026, 1, 6), filed=date(2026, 2, 2))],
        MEMBERS,
        prices,
    )
    position = positions[0]
    price_at_trade, _ = prices.close_on_or_after("ACME", date(2026, 1, 6))
    price_at_filing, _ = prices.close_on_or_after("ACME", date(2026, 2, 2))
    assert position.copier_basis == price_at_filing
    assert position.copier_basis != price_at_trade
    assert position.unrealized_return is not None


def test_the_summary_counts_who_is_building_and_who_is_leaving():
    positions = build_positions(
        [
            _trade("a", politician_id="m1", traded=date(2026, 1, 6), filed=date(2026, 2, 2)),
            _trade("b", politician_id="m1", traded=date(2026, 1, 8), filed=date(2026, 2, 4)),
            _trade("c", politician_id="m2", kind="sale"),
        ],
        MEMBERS,
    )
    summary = accumulation_summary(positions)
    assert summary["members"] == 2
    assert summary["accumulating"] == 1
    # Sales only: nothing here watched them build the position they are
    # leaving, which is a weaker fact than trimming one we did watch.
    assert summary["selling_down"] == 1
    assert summary["trimming"] == 0
