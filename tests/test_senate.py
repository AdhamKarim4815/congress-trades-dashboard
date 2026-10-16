"""Parsing Senate periodic transaction reports.

Unlike the House's PDFs these arrive as a real HTML table, so the parsing is
simpler — but the column meanings still have to be mapped exactly, and a
"paper" filing carries no table at all.
"""

from __future__ import annotations

from datetime import date

from trade_bot.domain import AssetClass, Owner, TransactionType
from trade_bot.senate import (
    SenateFiling,
    _filing_from_row,
    parse_senate_filing,
    senate_politician,
)

FILING_HTML = """
<table class="table">
  <tr class="header">
    <th>#</th><th>Transaction Date</th><th>Owner</th><th>Ticker</th>
    <th>Asset Name</th><th>Asset Type</th><th>Type</th><th>Amount</th><th>Comment</th>
  </tr>
  <tr>
    <td>1</td><td>07/23/2026</td><td>Self</td>
    <td><a href="/x">O</a></td>
    <td>Realty Income Corporation Common Stock</td>
    <td>Stock</td><td>Purchase</td><td>$1,001 - $15,000</td><td>--</td>
  </tr>
  <tr>
    <td>2</td><td>07/24/2026</td><td>Spouse</td><td>--</td>
    <td>US Treasury Note</td>
    <td>Corporate Bond</td><td>Sale (Partial)</td><td>$15,001 - $50,000</td><td>note</td>
  </tr>
  <tr>
    <td>3</td><td>07/25/2026</td><td>Joint</td><td>XLU</td>
    <td>SPDR Select Sector Fund - Utilities</td>
    <td>Exchange Traded Fund</td><td>Sale (Full)</td><td>$50,001 - $100,000</td><td>--</td>
  </tr>
</table>
"""

FILED_AT = date(2026, 8, 7)


def test_parses_every_transaction_row():
    rows = parse_senate_filing(FILING_HTML, FILED_AT)
    assert len(rows) == 3
    assert [row.symbol for row in rows] == ["O", None, "XLU"]


def test_maps_owners_types_and_amounts():
    first, second, third = parse_senate_filing(FILING_HTML, FILED_AT)

    assert first.owner is Owner.SELF
    assert first.transaction_type is TransactionType.PURCHASE
    assert first.asset_class is AssetClass.STOCK
    assert first.transaction_date == date(2026, 7, 23)
    assert (first.amount_low, first.amount_high) == (1001, 15000)

    assert second.owner is Owner.SPOUSE
    assert second.transaction_type is TransactionType.PARTIAL_SALE
    assert second.asset_class is AssetClass.BOND
    assert second.symbol is None, "a '--' ticker means there is none"

    assert third.owner is Owner.JOINT
    assert third.transaction_type is TransactionType.SALE
    assert third.asset_class is AssetClass.FUND


def test_ticker_links_are_stripped_to_the_symbol():
    first = parse_senate_filing(FILING_HTML, FILED_AT)[0]
    assert first.symbol == "O"
    assert "<" not in first.asset_name


def test_a_filing_with_no_table_yields_nothing():
    assert parse_senate_filing("<html><body>Paper filing</body></html>", FILED_AT) == []


def test_search_row_is_read_into_a_filing():
    row = [
        "Gary C",
        "Peters",
        "Peters, Gary (Senator)",
        '<a href="/search/view/ptr/328a9b36-1205-4117-bdc4-cd7c3ccbcbbc/" '
        'target="_blank">Periodic Transaction Report for 08/07/2026</a>',
        "08/07/2026",
    ]
    filing = _filing_from_row(row)
    assert filing is not None
    assert filing.uuid == "328a9b36-1205-4117-bdc4-cd7c3ccbcbbc"
    assert filing.kind == "ptr"
    assert filing.filed_at == date(2026, 8, 7)
    assert filing.full_name == "Gary C Peters"
    assert filing.url.endswith("/search/view/ptr/328a9b36-1205-4117-bdc4-cd7c3ccbcbbc/")


def test_paper_filings_are_recognised_as_a_different_kind():
    row = [
        "Tommy",
        "Tuberville",
        "Tuberville, Tommy (Senator)",
        '<a href="/search/view/paper/abc12345-0000-0000-0000-000000000000/">Report</a>',
        "01/15/2026",
    ]
    filing = _filing_from_row(row)
    assert filing is not None and filing.kind == "paper"


def test_malformed_search_rows_are_skipped():
    assert _filing_from_row(["too", "short"]) is None
    assert _filing_from_row(["a", "b", "c", "no link here", "08/07/2026"]) is None


def test_politician_id_omits_state_because_search_does_not_supply_one():
    """The House index carries a district; Senate search results do not.

    The id must still be stable, so it is built from the name alone and
    `sync_legislators` fills in state and party afterwards.
    """
    filing = SenateFiling(
        uuid="u", kind="ptr", first_name="Gary C", last_name="Peters",
        office="Peters, Gary (Senator)", filed_at=FILED_AT,
    )
    politician = senate_politician(filing)
    assert politician.politician_id == "senate-peters-gary-c"
    assert politician.chamber.value == "senate"
    assert politician.full_name == "Gary C Peters"
