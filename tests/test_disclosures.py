"""Parsing House periodic transaction reports.

The fixtures below are lightly-trimmed real PTR extractions, including the
NUL bytes that the form's small-caps labels produce, because every awkward
case this parser handles came from a real filing.
"""

from __future__ import annotations

from datetime import date

from trade_bot.disclosures import (
    SYMBOL_SHAPE_RE,
    ParsedTransaction,
    build_trades,
    classify_asset,
    corroborate_transaction_dates,
    paren_ticker,
    parse_amount,
    parse_ptr_text,
    politician_id,
    repair_transaction_date,
    split_state_district,
)
from trade_bot.domain import AssetClass, Chamber, Filing, Owner, TransactionType, build_trade_id

# Ticker in parentheses, asset name wrapping onto the code's own line, and a
# multi-line description that must not bleed into the next row's asset name.
WRAPPED_PTR = """P\x00\x00\x00 T\x00\x00\x00\x00 R\x00\x00
Name: Hon. Mark Alford
State/District:MO04
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
Amazon.com, Inc. - Common Stock
(AMZN) [ST]
S (partial) 03/16/202603/16/2026$1,001 - $15,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00: New
S\x00\x00\x00\x00\x00 O\x00: Putnam Investments
D\x00\x00\x00\x00\x00\x00: The full transaction included the following sales: T - 37.426 shares
sold @ $27.645/share BRK/B - 3 shares sold @ $493.42/share
Apple Inc. - Common Stock (AAPL)
[ST]
S (partial) 03/16/202603/16/2026$1,001 - $15,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00: New
"""

# Owner prefix, and the transaction type butting straight against "]".
TIGHT_PTR = """ID Owner Asset Transaction
Type
Date Notification
Date
Procter & Gamble Company (PG) [ST]P 05/15/202606/04/2026$1,001 - $15,000
F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00: New
SP Atlas Energy Solutions Inc. Common
Stock (AESI) [RS]
P 04/30/202604/30/2026$5,000,001 - $25,000,000
"""

BOND_PTR = """ID Owner Asset Transaction
Type
JPMORGAN CHASE &CO NOTE
CALL MAKE WHOLE 5.57200%
04/22/2036 [CS]
P 05/08/202605/18/2026$1,001 - $15,000
"""


def test_parses_wrapped_asset_names_and_tickers():
    rows = parse_ptr_text(WRAPPED_PTR)
    assert len(rows) == 2
    assert [row.symbol for row in rows] == ["AMZN", "AAPL"]
    assert all(row.transaction_type is TransactionType.PARTIAL_SALE for row in rows)
    assert all(row.asset_class is AssetClass.STOCK for row in rows)
    assert rows[0].transaction_date == date(2026, 3, 16)
    assert rows[0].amount_low == 1001 and rows[0].amount_high == 15000


def test_multiline_description_does_not_leak_into_next_asset_name():
    rows = parse_ptr_text(WRAPPED_PTR)
    # The AAPL row follows a two-line description full of share counts.
    assert "shares" not in rows[1].asset_name
    assert "Putnam" not in rows[1].asset_name
    assert rows[1].asset_name.startswith("Apple Inc.")


def test_transaction_type_directly_after_asset_code():
    rows = parse_ptr_text(TIGHT_PTR)
    assert rows[0].symbol == "PG"
    assert rows[0].transaction_type is TransactionType.PURCHASE
    assert rows[0].asset_class is AssetClass.STOCK


def test_owner_prefix_is_recognised():
    rows = parse_ptr_text(TIGHT_PTR)
    spouse = rows[1]
    assert spouse.owner is Owner.SPOUSE
    assert spouse.symbol == "AESI"
    assert not spouse.asset_name.startswith("SP ")
    assert spouse.amount_high == 25_000_000


def test_bonds_are_not_treated_as_tradeable_equity():
    rows = parse_ptr_text(BOND_PTR)
    assert len(rows) == 1
    assert rows[0].asset_class is AssetClass.BOND
    assert rows[0].symbol is None


def test_scanned_filing_yields_nothing_rather_than_garbage():
    assert parse_ptr_text("") == []
    assert parse_ptr_text("\x0c \n ") == []


def test_parse_amount_handles_ranges_and_singles():
    assert parse_amount("$1,001 - $15,000") == (1001.0, 15000.0)
    assert parse_amount("$50,001 -\n$100,000") == (50001.0, 100000.0)
    assert parse_amount("$1,000") == (1000.0, 1000.0)
    assert parse_amount("") == (0.0, 0.0)


def test_classify_asset_treats_other_with_ticker_as_a_fund():
    # ETFs are routinely filed as "OT"; a real ticker is the signal.
    assert classify_asset("OT", "QQQ") is AssetClass.FUND
    assert classify_asset("OT", None) is AssetClass.OTHER
    assert classify_asset("ST", "AAPL") is AssetClass.STOCK
    assert classify_asset("CT", "BTC") is AssetClass.CRYPTO


def test_split_state_district():
    assert split_state_district("MO04") == ("MO", "04")
    assert split_state_district("PR00") == ("PR", None)
    assert split_state_district("TX") == ("TX", None)


def test_politician_id_is_stable_and_slugged():
    first = politician_id(Chamber.HOUSE, "MO", "Alford", "Mark")
    second = politician_id(Chamber.HOUSE, "MO", "Alford", "Mark")
    assert first == second == "house-mo-alford-mark"


def test_trade_ids_are_deterministic_but_distinguish_repeated_rows():
    args = ("house:2026:1", "AAPL", "purchase", date(2026, 1, 5), "self", 1001.0)
    assert build_trade_id(*args, 0) == build_trade_id(*args, 0)
    # A filing legitimately repeats an identical line; the ordinal keeps them apart.
    assert build_trade_id(*args, 0) != build_trade_id(*args, 1)


def test_build_trades_uses_filing_date_as_the_disclosure_date():
    """The notification date is when the broker told the member.

    Only the filing date is when the public could know, so that is what a
    copier's returns must be measured from.
    """
    filing = Filing(
        filing_id="house:2026:99",
        politician_id="house-mo-alford-mark",
        chamber=Chamber.HOUSE,
        doc_id="99",
        year=2026,
        filed_at=date(2026, 6, 4),
        source_url="https://example.invalid/99.pdf",
    )
    rows = parse_ptr_text(TIGHT_PTR)
    trades = build_trades(filing, rows)
    pg = trades[0]
    assert pg.transaction_date == date(2026, 5, 15)
    assert pg.disclosure_date == date(2026, 6, 4)
    assert pg.reporting_lag_days == 20


def test_a_mistyped_future_transaction_date_is_clamped_and_recorded():
    """Filers do mistype years.

    Rep. Cohen's 2026 PTR reports a purchase dated 12/26/2026 with a
    notification date of 01/21/2026 -- a trade after it was reported, which
    cannot happen. Left alone it poisons return windows and the "latest
    disclosure" figure, so it is clamped to the filing date with the original
    kept in the comment.
    """
    filing = Filing(
        filing_id="house:2026:1",
        politician_id="house-mo-alford-mark",
        chamber=Chamber.HOUSE,
        doc_id="1",
        year=2026,
        filed_at=date(2020, 1, 1),
        source_url="",
    )
    trades = build_trades(filing, parse_ptr_text(TIGHT_PTR))
    assert all(trade.disclosure_date >= trade.transaction_date for trade in trades)
    assert all(trade.transaction_date <= filing.filed_at for trade in trades)
    flagged = [trade for trade in trades if trade.comment]
    assert flagged, "the impossible date should be recorded, not silently altered"
    assert "after the filing date" in flagged[0].comment


def test_a_wrapped_description_does_not_become_the_next_asset_name():
    """A description bleeding downwards produced a company called a sentence.

    The real filing behind this stored ZTS as "the stocks be divested. The new
    investment firm sold the stock and instead invested in Exchange Traded
    Funds. SP Zoetis Inc. Class A". Only the *first* line of a description
    carries its `D:` label, so the unlabelled continuation lines were invisible
    to the metadata guard, and the wording ("sold the stock", not "shares
    sold") slipped past the descriptive one too.
    """
    text = (
        "ID Owner Asset Transaction\n"
        "Type\n"
        "Date Notification\n"
        "Date\n"
        "Merck & Co Inc. Common Stock (MRK) [ST]\n"
        "S 05/02/202605/20/2026$1,001 - $15,000\n"
        "D\x00\x00\x00\x00\x00\x00: The trustee recommended that\n"
        "the stocks be divested. The new investment firm\n"
        "sold the stock and instead invested in Exchange\n"
        "Traded Funds.\n"
        "SP Zoetis Inc. Class A [ST]\n"
        "P 06/09/202607/15/2026$1,001 - $15,000\n"
    )
    rows = parse_ptr_text(text)
    assert rows, "the transaction row should still parse"
    row = rows[-1]
    assert row.asset_name == "Zoetis Inc. Class A"
    # The same bleed pushed the owner code off position 0, so OWNER_RE never
    # matched and the trade was stored as UNKNOWN. Both must stay fixed.
    assert row.owner is Owner.SPOUSE


def test_an_asset_name_that_merely_contains_an_owner_letter_pair_is_kept():
    """`SP` inside a name is not the owner column. Do not cut on it."""
    text = (
        "ID Owner Asset Transaction\n"
        "Type\n"
        "Date Notification\n"
        "Date\n"
        "SPDR S&P 500 ETF Trust (SPY) [ST]\n"
        "P 06/09/202607/15/2026$1,001 - $15,000\n"
    )
    rows = parse_ptr_text(text)
    assert rows
    assert "SPDR" in rows[-1].asset_name


def _filing(filed_at: date, filing_id: str = "house:2025:1") -> Filing:
    return Filing(
        filing_id=filing_id,
        politician_id="house-ca-khanna-rohit",
        chamber=Chamber.HOUSE,
        doc_id=filing_id.rsplit(":", 1)[-1],
        year=filed_at.year,
        filed_at=filed_at,
        source_url="",
    )


def _row(transaction_date: date, name: str = "ACME CORP CMN") -> ParsedTransaction:
    return ParsedTransaction(
        asset_name=name,
        symbol=None,
        asset_class=AssetClass.STOCK,
        transaction_type=TransactionType.PURCHASE,
        owner=Owner.UNKNOWN,
        transaction_date=transaction_date,
        notification_date=transaction_date,
        amount_low=1001.0,
        amount_high=15000.0,
    )


def test_a_year_read_a_century_early_is_corrected():
    """The 36,535-day reporting lag.

    A vision model reading a scanned filing wrote 1925-02-25 for 2025-02-25.
    Only the century is wrong, so correcting it invents no digit.
    """
    repaired, comment = repair_transaction_date(date(1925, 2, 25), date(2025, 3, 6))
    assert repaired == date(2025, 2, 25)
    assert comment and "a century before the filing" in comment


def test_a_genuinely_late_filing_is_left_alone():
    """Three years late is a fact about the member, not a parse error."""
    repaired, comment = repair_transaction_date(date(2023, 6, 30), date(2026, 7, 14))
    assert repaired == date(2023, 6, 30)
    assert comment is None


def test_an_absurd_date_is_corroborated_by_the_rest_of_its_filing():
    """1975-02-04 alongside 2025-02-04 in the same filing.

    The scanner read the day and month correctly on other rows of the same
    page, so the year those rows agree on is evidence rather than a guess.
    """
    filed = date(2025, 3, 6)
    dates = [date(1975, 2, 4), date(2025, 2, 4), date(1975, 2, 14)]
    repaired = corroborate_transaction_dates(dates, filed)
    assert repaired[0][0] == date(2025, 2, 4)
    assert repaired[0][1] and "other rows of the same filing" in repaired[0][1]
    assert repaired[1] == (date(2025, 2, 4), None)
    # No row of this filing dates 14 February, so nothing vouches for it and it
    # is left exactly as it was read.
    assert repaired[2][0] == date(1975, 2, 14)


def test_corroboration_refuses_when_the_whole_filing_is_unreadable():
    """With no sound row on the page there is nothing to corroborate against."""
    filed = date(2025, 3, 6)
    dates = [date(1975, 2, 4), date(1975, 2, 5)]
    repaired = corroborate_transaction_dates(dates, filed)
    assert [value for value, _ in repaired] == dates
    assert all(comment is None for _, comment in repaired)


def test_build_trades_keeps_ids_stable_across_a_date_repair():
    """A repaired row must not be re-inserted as a new trade.

    The id hashes the date as it was *read*, so re-parsing a filing after the
    repair lands on the same id and `add_trades` filters it out.
    """
    filing = _filing(date(2025, 3, 6))
    trades = build_trades(filing, [_row(date(1925, 2, 25))])
    assert trades[0].transaction_date == date(2025, 2, 25)
    assert trades[0].trade_id == build_trade_id(
        filing.filing_id, "ACME CORP CMN", "purchase", date(1925, 2, 25), "unknown", 1001.0, 0
    )


def test_the_symbol_is_read_from_the_last_parenthesis_not_the_first():
    """`(DE)` is the state of incorporation, and it priced UnitedHealth as Deere."""
    assert paren_ticker("UnitedHealth Group Incorporated Common Stock (DE) (UNH)") == "UNH"
    assert paren_ticker("American Tower Corporation (REIT) (AMT)") == "AMT"
    assert paren_ticker("TORONTO-DOMINION BANK (THE)") is None
    assert paren_ticker("Amazon.com, Inc. - Common Stock (AMZN)") == "AMZN"


def test_a_symbol_that_is_not_a_ticker_is_cleared_but_the_row_is_kept():
    """`-- AMCR` reached the price fetcher every run and could only fail.

    The asset name is what the filing actually said and stays untouched; only
    the claim that the row is tradeable goes away. Clearing beats guessing:
    there is no way to know which company `-- AMCR` was meant to be.
    """
    assert SYMBOL_SHAPE_RE.fullmatch("AAPL")
    assert SYMBOL_SHAPE_RE.fullmatch("BRK.B")
    assert SYMBOL_SHAPE_RE.fullmatch("AHL-C")
    assert not SYMBOL_SHAPE_RE.fullmatch("-- AMCR")
    assert not SYMBOL_SHAPE_RE.fullmatch("ETRN EQT")
    assert not SYMBOL_SHAPE_RE.fullmatch("")
