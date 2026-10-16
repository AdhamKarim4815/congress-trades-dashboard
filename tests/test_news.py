"""RSS parsing and the query construction that feeds it."""

from __future__ import annotations

from trade_bot.news import company_names, figure_query, member_query, parse_rss, ticker_query

FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Test feed</title>
<item>
  <title>ACME shares surge on record profit</title>
  <link>https://example.invalid/a</link>
  <pubDate>Fri, 07 Aug 2026 10:23:51 GMT</pubDate>
  <source url="https://cnbc.com">CNBC</source>
  <description>&lt;p&gt;Shares climbed after earnings.&lt;/p&gt;</description>
</item>
<item>
  <title>Second story</title>
  <link>https://example.invalid/b</link>
  <pubDate>Thu, 06 Aug 2026 13:31:40 GMT</pubDate>
</item>
</channel></rss>"""


def test_parses_items_with_metadata():
    items = parse_rss(FEED)
    assert len(items) == 2
    first = items[0]
    assert first.title == "ACME shares surge on record profit"
    assert first.url == "https://example.invalid/a"
    assert first.source == "CNBC"
    assert first.published_at.tzinfo is not None
    assert first.published_at.year == 2026


def test_html_is_stripped_from_descriptions():
    item = parse_rss(FEED)[0]
    assert item.summary == "Shares climbed after earnings."
    assert "<p>" not in (item.summary or "")


def test_items_without_a_link_or_title_are_dropped():
    feed = """<rss><channel>
      <item><title>No link</title></item>
      <item><link>https://example.invalid/x</link></item>
      <item><title>Good</title><link>https://example.invalid/y</link></item>
    </channel></rss>"""
    assert [item.title for item in parse_rss(feed)] == ["Good"]


def test_malformed_feed_yields_nothing_rather_than_raising():
    """A dead or garbled feed must not break a whole news sync."""
    assert parse_rss("<not xml") == []
    assert parse_rss("") == []


def test_limit_is_respected():
    assert len(parse_rss(FEED, limit=1)) == 1


def test_ids_are_stable_across_parses():
    assert [item.item_id for item in parse_rss(FEED)] == [
        item.item_id for item in parse_rss(FEED)
    ]


def test_missing_pubdate_defaults_to_now_and_stays_aware():
    feed = """<rss><channel><item>
      <title>T</title><link>https://example.invalid/z</link>
    </item></channel></rss>"""
    assert parse_rss(feed)[0].published_at.tzinfo is not None


def test_member_query_targets_the_right_chamber():
    assert "senator" in member_query("Jane Doe", "senate")
    assert "representative" in member_query("Jane Doe", "house")
    assert '"Jane Doe"' in member_query("Jane Doe", "house")


def test_ticker_query_prefers_a_cleaned_company_name():
    query = ticker_query("AAPL", "Apple Inc. - Common Stock")
    assert "Apple" in query
    assert "Common Stock" not in query
    # With no name, fall back to the symbol.
    assert "AAPL" in ticker_query("AAPL", None)


def test_ticker_query_strips_an_embedded_ticker():
    assert "(AAPL)" not in ticker_query("AAPL", "Apple Inc. (AAPL)")


def test_figure_query_names_the_figure():
    assert '"Donald Trump"' in figure_query("Donald Trump")


def test_company_names_prefers_the_longest_seen_spelling(store):
    from datetime import date

    from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, TransactionType

    def trade(trade_id, name):
        return DisclosedTrade(
            trade_id=trade_id,
            filing_id="f",
            politician_id="p",
            chamber=Chamber.HOUSE,
            symbol="AAPL",
            asset_name=name,
            asset_class=AssetClass.STOCK,
            transaction_type=TransactionType.PURCHASE,
            transaction_date=date(2026, 1, 5),
            disclosure_date=date(2026, 2, 5),
            amount_low=1001,
            amount_high=15000,
        )

    store.add_trades([trade("a", "Apple Inc"), trade("b", "Apple Inc. - Common Stock (AAPL)")])
    names = company_names(store)
    assert names["AAPL"] == "Apple Inc. - Common Stock"
