"""Store behaviour: idempotency, enrichment and timestamp handling."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from trade_bot.domain import (
    AssetClass,
    Chamber,
    DisclosedTrade,
    Filing,
    PoliticalMention,
    Politician,
    Stance,
    TransactionType,
)


def a_trade(trade_id: str = "t1", symbol: str | None = "ACME", asset_class=AssetClass.STOCK):
    return DisclosedTrade(
        trade_id=trade_id,
        filing_id="house:2026:1",
        politician_id="house-tx-doe-jane",
        chamber=Chamber.HOUSE,
        symbol=symbol,
        asset_name="ACME Common Stock",
        asset_class=asset_class,
        transaction_type=TransactionType.PURCHASE,
        transaction_date=date(2026, 1, 5),
        disclosure_date=date(2026, 2, 5),
        amount_low=1001,
        amount_high=15000,
    )


def test_adding_the_same_trade_twice_is_a_no_op(store):
    assert store.add_trades([a_trade()]) == 1
    assert store.add_trades([a_trade()]) == 0
    assert store.trade_counts()["total"] == 1


def test_actionable_filter_excludes_bonds_and_untickered_lines(store):
    store.add_trades(
        [
            a_trade("stock"),
            a_trade("bond", symbol=None, asset_class=AssetClass.BOND),
            a_trade("odd", symbol="XYZ", asset_class=AssetClass.OTHER),
        ]
    )
    assert store.trade_counts()["total"] == 3
    assert store.trade_counts()["actionable"] == 1
    assert store.distinct_trade_symbols() == ["ACME"]


def test_trade_dict_derives_midpoint_and_lag(store):
    store.add_trades([a_trade()])
    trade = store.list_trades()[0]
    assert trade["amount_midpoint"] == 8000.5
    assert trade["reporting_lag_days"] == 31


def test_politician_enrichment_is_additive(store):
    """The filing index knows district but not party; the roster knows party
    but nothing about filings. Neither may erase the other's contribution."""
    store.upsert_politicians(
        [
            Politician(
                politician_id="house-tx-doe-jane",
                full_name="Jane Doe",
                first_name="Jane",
                last_name="Doe",
                chamber=Chamber.HOUSE,
                state="TX",
                district="21",
            )
        ]
    )
    store.upsert_politicians(
        [
            Politician(
                politician_id="house-tx-doe-jane",
                full_name="Jane Doe",
                first_name="Jane",
                last_name="Doe",
                chamber=Chamber.HOUSE,
                state="TX",
                party="Republican",
                bioguide_id="D000123",
            )
        ]
    )
    member = store.get_politician("house-tx-doe-jane")
    assert member["district"] == "21"  # not blanked by the second write
    assert member["party"] == "Republican"
    assert member["display_state"] == "TX-21"


def test_upsert_reports_only_genuinely_new_members(store):
    politician = Politician(
        politician_id="house-tx-doe-jane",
        full_name="Jane Doe",
        chamber=Chamber.HOUSE,
        state="TX",
    )
    assert store.upsert_politicians([politician]) == 1
    assert store.upsert_politicians([politician]) == 0


def test_filing_upsert_records_parse_failures(store):
    filing = Filing(
        filing_id="house:2026:9",
        politician_id="house-tx-doe-jane",
        chamber=Chamber.HOUSE,
        doc_id="9",
        year=2026,
        filed_at=date(2026, 3, 1),
        source_url="https://example.invalid/9.pdf",
    )
    store.upsert_filing(filing.model_copy(update={"parse_error": "scanned paper filing"}))
    counts = store.filing_counts()
    assert counts["total"] == 1 and counts["failed"] == 1 and counts["parsed"] == 0
    assert store.known_filing_ids(Chamber.HOUSE.value) == {"house:2026:9"}


def test_news_is_deduplicated_and_aggregated(store):
    rows = [
        {
            "item_id": "n1",
            "category": "ticker",
            "subject": "ACME",
            "title": "ACME surges",
            "url": "https://example.invalid/1",
            "published_at": datetime(2026, 7, 1, tzinfo=UTC),
            "sentiment_score": 0.6,
            "sentiment_label": "bullish",
        },
        {
            "item_id": "n2",
            "category": "ticker",
            "subject": "ACME",
            "title": "ACME slips",
            "url": "https://example.invalid/2",
            "published_at": datetime(2026, 7, 2, tzinfo=UTC),
            "sentiment_score": -0.2,
            "sentiment_label": "bearish",
        },
    ]
    assert store.add_news(rows) == 2
    assert store.add_news(rows) == 0
    summary = store.news_sentiment_by_subject("ticker", datetime(2026, 1, 1, tzinfo=UTC))
    assert summary["ACME"]["items"] == 2
    assert summary["ACME"]["mean_score"] == pytest.approx(0.2)


def test_timestamps_come_back_timezone_aware(store):
    """SQLite hands back naive datetimes; domain records reject those."""
    store.add_mentions(
        [
            PoliticalMention(
                mention_id="m1",
                figure="Donald Trump",
                figure_slug="donald-trump",
                symbol="ACME",
                stance=Stance.BULLISH,
                headline="Trump praises ACME",
                url="https://example.invalid/m1",
                published_at=datetime(2026, 7, 1, tzinfo=UTC),
                confidence=0.8,
            )
        ]
    )
    mention = store.list_mentions()[0]
    assert mention["published_at"].tzinfo is not None


def test_state_round_trips_json(store):
    store.set_state("sync.test", {"ok": True, "count": 3})
    assert store.get_state("sync.test") == {"ok": True, "count": 3}
    assert store.get_state("missing") is None


def test_disclosed_since_filter(store):
    store.add_trades([a_trade("old"), a_trade("new")])
    assert len(store.list_trades(disclosed_since=date(2026, 1, 1))) == 2
    assert store.list_trades(disclosed_since=date(2026, 6, 1)) == []
