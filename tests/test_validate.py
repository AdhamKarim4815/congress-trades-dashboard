"""The stage that judges the data rather than the code.

Each case below is a defect that actually shipped while the rest of the suite
was green, which is the whole argument for the module existing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from trade_bot.database import Store
from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, Politician, TransactionType
from trade_bot.prices import PriceRepository
from trade_bot.validate import (
    check_asset_names,
    check_duplicates,
    check_inert_sources,
    check_malformed_symbols,
    check_member_enrichment,
    check_price_adjustment,
    check_reporting_lag,
    check_symbol_names,
    duplicate_groups,
    load_report,
    run_validation,
    scan_series_for_splits,
)


def _trade(**overrides) -> dict:
    base = {
        "trade_id": "t1",
        "filing_id": "house:2025:1",
        "politician_id": "house-ca-khanna-rohit",
        "symbol": "AAPL",
        "asset_name": "Apple Inc. - Common Stock (AAPL)",
        "asset_class": "stock",
        "transaction_type": "purchase",
        "transaction_date": date(2025, 2, 25),
        "disclosure_date": date(2025, 3, 6),
        "amount_low": 1001.0,
        "amount_high": 15000.0,
        "amount_midpoint": 8000.5,
        "source_url": "",
    }
    base.update(overrides)
    base["reporting_lag_days"] = (base["disclosure_date"] - base["transaction_date"]).days
    return base


def _series(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    opens = opens or list(closes)
    start = pd.Timestamp("2025-01-06")
    return pd.DataFrame(
        {
            "date": [start + pd.Timedelta(days=index) for index in range(len(closes))],
            "open": opens,
            "high": [max(a, b) for a, b in zip(opens, closes, strict=True)],
            "low": [min(a, b) for a, b in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )


# -- trades ----------------------------------------------------------------


def test_a_century_scale_lag_is_reported_as_an_error():
    finding = check_reporting_lag([_trade(transaction_date=date(1925, 2, 25))])
    assert finding is not None
    assert finding.severity == "error"
    assert finding.count == 1


def test_an_ordinary_lag_produces_no_finding():
    assert check_reporting_lag([_trade()]) is None


def test_a_bled_description_is_reported_but_a_long_real_name_is_not():
    """Length alone cannot separate them, and neither can punctuation.

    "AT&T Inc. Depositary Shares, each representing a 1/1,000th interest in a
    share of 5.000% Perpetual Preferred Stock" is a real name, is long, and
    ends in a full stop's worth of prose. It must not be flagged.
    """
    real = (
        "AT&T Inc. Depositary Shares, each representing a 1/1,000th interest "
        "in a share of 5.000% Perpetual Preferred Stock"
    )
    bled = (
        "the stocks be divested. The new investment firm sold the stock and "
        "instead invested in Exchange Traded Funds, which the member says was "
        "done without their knowledge or direction at any point."
    )
    finding = check_asset_names([_trade(asset_name=real), _trade(asset_name=bled)])
    assert finding is not None
    assert finding.count == 1
    assert finding.samples[0]["asset_name"].startswith("the stocks be divested")


def test_a_symbol_contradicted_by_its_own_name_is_reported():
    """`(DE)` is the state of incorporation; the row was priced as Deere."""
    finding = check_symbol_names(
        [_trade(symbol="DE", asset_name="UnitedHealth Group Incorporated Common Stock (UNH)")]
    )
    assert finding is not None
    assert finding.samples[0]["printed_symbol"] == "UNH"


def test_a_parenthesis_that_is_not_a_ticker_is_not_a_contradiction():
    """Filers write plenty of things where the ticker goes.

    The state of incorporation, the instrument type, an article, a country.
    Reading those as tickers reported a contradiction on every one, which is
    how a check earns its way into being ignored.
    """
    for name in (
        "American Tower Corporation Common Stock (REIT)",
        "TORONTO-DOMINION BANK (THE)",
        "SCHNEIDER ELECTRIC SE UNSPONSORED ADR (FRANCE)",
    ):
        assert check_symbol_names([_trade(symbol="AMT", asset_name=name)]) is None


def test_identical_disclosure_lines_group_as_lots():
    trades = [_trade(trade_id="a"), _trade(trade_id="b"), _trade(trade_id="c", symbol="MSFT")]
    groups = duplicate_groups(trades)
    assert len(groups) == 1
    assert len(groups[0]) == 2
    finding = check_duplicates(trades)
    assert finding is not None
    assert finding.severity == "info", "two lots is a fact about the filing, not a defect"
    assert finding.samples[0]["lots"] == 2


# -- prices ----------------------------------------------------------------


def test_a_back_adjusted_split_leaves_no_trace():
    """What the providers actually return today, and what must keep being true."""
    assert scan_series_for_splits(_series([100.0, 101.0, 99.0, 100.5, 102.0])) == []


def test_an_unadjusted_four_for_one_split_is_caught():
    closes = [400.0, 404.0, 101.0, 102.0, 100.0]
    opens = [400.0, 402.0, 101.0, 101.5, 102.0]
    hits = scan_series_for_splits(_series(closes, opens))
    assert len(hits) == 1
    assert hits[0]["ratio"] == 4
    assert hits[0]["reverted"] is False


def test_a_real_gap_that_keeps_moving_is_not_called_a_split():
    """Agios halved on trial data and opened 2.4% off a clean two-for-one.

    A split is arithmetic and lands exactly; a collapse gaps and then keeps
    going. Treating the second as the first would report a defect on every
    bad day in biotech.
    """
    closes = [45.49, 22.34, 21.10, 20.80]
    opens = [45.00, 23.29, 22.20, 21.00]
    assert scan_series_for_splits(_series(closes, opens)) == []


def test_a_bar_that_jumps_and_comes_straight_back_is_a_spike():
    closes = [20.60, 103.00, 20.44, 20.50]
    opens = [20.55, 103.00, 20.37, 20.44]
    hits = scan_series_for_splits(_series(closes, opens))
    assert hits, "a 5x bar that reverts the next session must be reported"
    assert all(hit["reverted"] for hit in hits)


def test_price_adjustment_says_so_explicitly_when_nothing_is_wrong(settings, tmp_path):
    """Silence is not the same as a clean bill of health.

    The backtest rests on these series being adjusted, so the check reports
    that it looked even when it found nothing.
    """
    from trade_bot.activity import NullReporter

    repository = PriceRepository(settings)
    repository.save("AAPL", _series([100.0, 101.0, 102.0, 103.0]))
    findings = check_price_adjustment(repository, ["AAPL"], NullReporter())
    codes = {finding.code for finding in findings}
    assert "split_adjustment" in codes
    assert all(finding.severity == "info" for finding in findings)


# -- sources ---------------------------------------------------------------


def test_a_source_that_reports_ok_while_doing_nothing_is_flagged(settings):
    """How the senator enrichment failed silently for all 35 senators."""
    store = Store(settings.database_url)
    store.initialize()
    store.set_state(
        "sync.disclosures",
        {
            "sources": {
                "congress-legislators": {
                    "ok": True,
                    "filings_seen": 0,
                    "filings_parsed": 0,
                    "trades_added": 0,
                    "politicians_added": 0,
                    "details": {"enriched": 0, "roster_size": 537},
                }
            }
        },
    )
    finding = check_inert_sources(store)
    assert finding is not None
    assert finding.samples[0]["source"] == "congress-legislators"


def test_a_source_that_looked_and_found_nothing_new_is_healthy(settings):
    store = Store(settings.database_url)
    store.initialize()
    store.set_state(
        "sync.disclosures",
        {
            "sources": {
                "house-clerk": {
                    "ok": True,
                    "filings_seen": 1754,
                    "filings_parsed": 0,
                    "trades_added": 0,
                    "politicians_added": 0,
                    "details": {"queued": 0},
                }
            }
        },
    )
    assert check_inert_sources(store) is None


def test_members_the_roster_never_matched_are_named(settings):
    store = Store(settings.database_url)
    store.initialize()
    store.upsert_politicians(
        [
            Politician(
                politician_id="senate-carper-thomas-r",
                full_name="Thomas R Carper",
                last_name="Carper",
                chamber=Chamber.SENATE,
            )
        ]
    )
    store.add_trades(
        [
            DisclosedTrade(
                trade_id="t1",
                filing_id="senate:2025:1",
                politician_id="senate-carper-thomas-r",
                chamber=Chamber.SENATE,
                symbol="AAPL",
                asset_name="Apple Inc.",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(2025, 1, 2),
                disclosure_date=date(2025, 1, 20),
            )
        ]
    )
    finding = check_member_enrichment(store)
    assert finding is not None
    assert finding.samples[0]["full_name"] == "Thomas R Carper"


# -- the stage --------------------------------------------------------------


def test_the_stage_stores_a_report_the_dashboard_can_read(settings):
    store = Store(settings.database_url)
    store.initialize()
    repository = PriceRepository(settings)
    repository.save("AAPL", _series([100.0, 101.0, 102.0]))
    store.add_trades(
        [
            DisclosedTrade(
                trade_id="t1",
                filing_id="house:2025:1",
                politician_id="house-ca-khanna-rohit",
                chamber=Chamber.HOUSE,
                symbol="AAPL",
                asset_name="Apple Inc. - Common Stock (AAPL)",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(1925, 2, 25),
                disclosure_date=date(2025, 3, 6),
            )
        ]
    )
    result = run_validation(settings, store, repository)
    assert result["errors"] >= 1

    report = load_report(store)
    assert report is not None
    assert datetime.fromisoformat(report["generated_at"]).tzinfo is not None
    codes = {finding["code"] for finding in report["findings"]}
    assert "reporting_lag" in codes
    # Findings are ordered worst-first, because the top of the list is the
    # only part anyone reliably reads.
    assert report["findings"][0]["severity"] == "error"


def test_a_report_is_written_even_when_there_is_nothing_to_say(settings):
    store = Store(settings.database_url)
    store.initialize()
    run_validation(settings, store, PriceRepository(settings))
    report = load_report(store)
    assert report is not None
    assert report["errors"] == 0
    assert report["generated_at"].endswith("+00:00") or report["generated_at"].endswith("Z")
    assert datetime.now(UTC) is not None


def test_a_symbol_that_is_not_a_ticker_is_an_error():
    """`-- AMCR` reached the price fetcher, where it could only ever fail."""
    finding = check_malformed_symbols([_trade(symbol="-- AMCR"), _trade(symbol="BRK.B")])
    assert finding is not None
    assert finding.severity == "error"
    assert finding.count == 1
    assert finding.samples[0]["symbol"] == "-- AMCR"


def test_ordinary_share_classes_are_not_malformed():
    for symbol in ("AAPL", "BRK.B", "AHL-C", "F"):
        assert check_malformed_symbols([_trade(symbol=symbol)]) is None
