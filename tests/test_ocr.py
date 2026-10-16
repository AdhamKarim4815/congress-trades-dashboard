"""Reading scanned paper filings, and the guards that keep it honest.

Nothing here calls the API. The tests cover the parts that decide what gets
believed: name-to-ticker resolution, which rows survive, and the orientation
fix that the accuracy of the whole path depends on.
"""

from __future__ import annotations

import io
from datetime import date

import pytest

from trade_bot.ocr import OcrRow, _normalise, to_transactions
from trade_bot.symbols import build_index, normalise, resolve


@pytest.fixture
def index() -> dict[str, str]:
    return build_index(
        [
            ("KEYSIGHT TECHNOLOGIES INC. COMMON STOCK (KEYS)", "KEYS"),
            ("DANAHER CORPORATION COMMON STOCK", "DHR"),
            ("WORKDAY, INC. - CLASS A COMMON STOCK", "WDAY"),
            ("ZIONS BANCORPORATION N.A. - COMMON STOCK (ZION)", "ZION"),
        ]
    )


def test_paper_names_resolve_to_the_electronic_corpus(index):
    """The paper form forbids tickers, so names must map back to symbols."""
    assert resolve("KEYSIGHT TECHNOLOGIES INC", index) == "KEYS"
    assert resolve("DANAHER CORPORATION", index) == "DHR"
    assert resolve("WORKDAY INC CL A", index) == "WDAY"
    assert resolve("ZIONS BANCORPORATION", index) == "ZION"


def test_an_unknown_company_resolves_to_nothing_rather_than_the_nearest_one(index):
    """A wrong ticker invents a trade the member never made.

    Leaving a row unpriced costs one disclosure; guessing puts a fabricated
    position on the board under a real person's name.
    """
    assert resolve("CAVCO INDUSTRIES INC", index) is None
    assert resolve("VALVOLINE INC", index) is None


def test_a_name_seen_against_two_symbols_is_dropped():
    """Ambiguity is not resolved by picking one."""
    index = build_index([("ACME CORP", "ACM"), ("ACME CORP", "ACME")])
    assert resolve("ACME CORP", index) is None


def test_short_names_never_match(index):
    """A one-word key matches far too much to be evidence of anything."""
    assert resolve("INC", index) is None
    assert resolve("THE", index) is None


def test_normalise_strips_only_the_uninformative_parts():
    assert normalise("DANAHER CORPORATION COMMON STOCK") == "DANAHER"
    assert normalise("KEYSIGHT TECHNOLOGIES INC. (KEYS)") == "KEYSIGHT TECHNOLOGIES"


def test_rows_without_a_date_or_direction_are_discarded(index):
    """Neither field has a safe default: a guess would be a fabricated trade."""
    rows = [
        OcrRow("DANAHER CORPORATION", None, date(2026, 6, 1), "purchase", "$1,001 - $15,000", True),
        OcrRow("DANAHER CORPORATION", None, None, "purchase", "$1,001 - $15,000", True),
        OcrRow("DANAHER CORPORATION", None, date(2026, 6, 1), "unknown", "$1,001 - $15,000", True),
    ]
    out = to_transactions(rows, index, date(2026, 7, 5))
    assert len(out) == 1
    assert out[0].symbol == "DHR"
    assert out[0].amount_low == 1001
    assert out[0].amount_high == 15000


def test_unresolved_rows_are_kept_but_not_classified_as_stock(index):
    """Most rows on these filings are private LPs, munis and funds.

    They are real disclosures and worth storing, but they are not copyable
    positions and must not reach the board as though they were.
    """
    rows = [
        OcrRow("Spruce Holdings LP", None, date(2026, 6, 1), "purchase", "$1,001 - $15,000", True)
    ]
    out = to_transactions(rows, index, date(2026, 7, 5))
    assert len(out) == 1
    assert out[0].symbol is None
    assert out[0].asset_class == "other"


def test_sideways_pages_are_turned_upright():
    """The single largest accuracy win in the scanned path, and it is free.

    Some pages are scanned into a portrait frame with the landscape form on
    its side. Measured on a real filing, a vision model read 20% of the rows
    off the sideways page and 100% off the same page rotated.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (1696, 2200), color=255).save(buffer, format="PNG")
    turned = _normalise(buffer.getvalue())

    assert turned is not None
    assert Image.open(io.BytesIO(turned)).size == (2200, 1696)


def test_upright_pages_are_left_alone():
    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (2200, 1696), color=255).save(buffer, format="PNG")
    kept = _normalise(buffer.getvalue())

    assert kept is not None
    assert Image.open(io.BytesIO(kept)).size == (2200, 1696)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("CMN", "instrument suffix: 'BROWN & BROWN, INC. CMN' is common stock"),
        ("COM", "instrument suffix"),
        ("SP", "owner column: spouse"),
        ("JT", "owner column: joint"),
        ("DC", "owner column: dependent child"),
        ("NULL", "the model writing out its own null"),
        ("TOOLONGX", "not shaped like a symbol"),
        ("12", "not shaped like a symbol"),
    ],
)
def test_ticker_shaped_noise_is_never_believed(raw, reason):
    """Each of these collapses many companies onto one symbol if trusted.

    They all sit in a ticker-shaped position on the form -- trailing the asset
    name, or in the owner column at the far left of every row -- so shape
    alone cannot tell them apart from a real symbol.
    """
    from trade_bot.ocr import _parse_row

    row = _parse_row(
        {
            "asset_name": "BROWN & BROWN, INC.",
            "ticker": raw,
            "transaction_date": "2026-06-01",
            "transaction_type": "purchase",
            "amount_range": "$1,001 - $15,000",
            "confident": True,
        }
    )
    assert row is not None
    assert row.ticker is None, reason


def test_a_real_ticker_still_survives():
    from trade_bot.ocr import _parse_row

    row = _parse_row(
        {
            "asset_name": "MICROSOFT CORP",
            "ticker": "msft",
            "transaction_date": "2026-06-01",
            "transaction_type": "purchase",
            "amount_range": "$1,001 - $15,000",
            "confident": True,
        }
    )
    assert row is not None and row.ticker == "MSFT"


def test_the_forms_own_example_row_is_not_a_transaction():
    """The blank PTR form carries a worked example above the first real row.

    A scan includes it, and read as data it invents transactions under a real
    member's name -- fifty of them in one backfill, in a stock that does not
    exist.
    """
    from trade_bot.ocr import _parse_row

    for name in (
        "Example: Mega Corp. Common Stock",
        "Example Mega Corp, Common Stock",
        "SAMPLE ASSET NAME",
    ):
        assert (
            _parse_row(
                {
                    "asset_name": name,
                    "ticker": "X",
                    "transaction_date": "2026-01-01",
                    "transaction_type": "purchase",
                    "amount_range": "$1,001 - $15,000",
                    "confident": True,
                }
            )
            is None
        ), name


def test_instrument_suffixes_do_not_drive_a_fuzzy_match():
    """`CMN` is scanned variously as CMT, CNA or CAN.

    Left in the name it dominates the comparison and pulls unrelated companies
    onto whichever ticker it resembles -- twenty REITs arrived under CMT this
    way, and eighteen unrelated names under CNA.
    """
    assert normalise("DIGITAL REALTY TRUST, INC. CMN") == "DIGITAL REALTY TRUST"
    assert normalise("JPMORGAN CHASE & CO. HYBRID PERPETUAL USD") == "JPMORGAN CHASE"
    assert normalise("EQUINIX, INC. REIT") == "EQUINIX"


def test_pages_already_paid_for_are_kept_when_a_later_page_fails(monkeypatch, tmp_path):
    """Discarding the filing meant paying for the same pages twice.

    The first backfill cost roughly three times its estimate, and this was
    the largest reason: a rate-limited page threw away the five pages already
    read and billed, and the retry then bought them again.
    """
    import trade_bot.ocr as ocr

    calls = {"n": 0}

    def flaky(settings, image, hint, number, count):
        calls["n"] += 1
        if number == 2:
            raise RuntimeError("rate_limit")
        return [
            {
                "asset_name": f"COMPANY {number}",
                "ticker": None,
                "transaction_date": "2026-06-01",
                "transaction_type": "purchase",
                "amount_range": "$1,001 - $15,000",
                "confident": True,
            }
        ]

    monkeypatch.setattr(ocr, "_ask_vision", flaky)
    monkeypatch.setattr(ocr, "is_available", lambda settings: True)
    monkeypatch.setattr(ocr, "page_images", lambda data: [b"a", b"b", b"c"])

    settings = type("S", (), {"project_root": tmp_path, "openai_model": "test"})()
    result = ocr.read_scanned_filing(settings, "doc-1", b"pdf")

    assert len(result.rows) == 2, "the two readable pages must survive the failed one"
    assert {row.asset_name for row in result.rows} == {"COMPANY 1", "COMPANY 3"}


def test_a_run_cannot_exceed_its_request_ceiling(monkeypatch):
    """The filing cap does not bound spend: a filing is many pages.

    This is the number that actually decides the bill, so it is enforced
    where the requests are made.
    """
    import trade_bot.ocr as ocr

    monkeypatch.setattr(ocr, "_MAX_REQUESTS_PER_PROCESS", 0)
    monkeypatch.setattr(ocr, "_requests_made", 0)
    with pytest.raises(ocr.BudgetExhausted):
        ocr._ask_vision(None, b"image", None, 1, 1)
