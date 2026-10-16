"""News ingest from free RSS, scored for sentiment on the way in.

Google News' RSS endpoints are the backbone: they take an arbitrary search
query or a topic section, need no key or account, and return up to a hundred
items with a title, publisher and timestamp. That covers every feed this
dashboard wants — a member's coverage, a ticker's coverage, a political
figure's statements, and general world/business headlines — from one free
source.

Every item is sentiment-scored at ingest (see `trade_bot.sentiment`) and
stored with the terms that drove the score, so a headline's contribution to
a ranking can always be traced back.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from defusedxml import ElementTree as DefusedET

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.sentiment import score_text

GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"
GOOGLE_NEWS_TOPIC = "https://news.google.com/rss/headlines/section/topic/{topic}"
LOCALE = {"hl": "en-US", "gl": "US", "ceid": "US:en"}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsItem:
    item_id: str
    title: str
    url: str
    source: str
    summary: str | None
    published_at: datetime

    @property
    def age_hours(self) -> float:
        return (datetime.now(UTC) - self.published_at).total_seconds() / 3600


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", TAG_RE.sub(" ", text)).strip()


def _item_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode()).hexdigest()[:32]


def _parse_published(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return datetime.now(UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_rss(payload: str, limit: int = 100) -> list[NewsItem]:
    """Parse an RSS document into items, tolerating partial/odd feeds."""
    try:
        root = DefusedET.fromstring(payload)
    except Exception:
        return []
    items: list[NewsItem] = []
    for node in root.iter("item"):
        title = _clean(node.findtext("title"))
        url = (node.findtext("link") or "").strip()
        if not title or not url:
            continue
        source_node = node.find("source")
        source = _clean(source_node.text) if source_node is not None else ""
        items.append(
            NewsItem(
                item_id=_item_id(url, title),
                title=title,
                url=url,
                source=source,
                summary=_clean(node.findtext("description")) or None,
                published_at=_parse_published(node.findtext("pubDate")),
            )
        )
        if len(items) >= limit:
            break
    return items


class GoogleNewsClient:
    """Thin RSS reader for Google News search and topic feeds."""

    name = "google-news"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def _get(self, url: str, params: dict[str, str]) -> str:
        with httpx.Client(
            timeout=self.timeout, headers=BROWSER_HEADERS, follow_redirects=True
        ) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text

    def search(self, query: str, limit: int = 40) -> list[NewsItem]:
        payload = self._get(GOOGLE_NEWS_SEARCH, {"q": query, **LOCALE})
        return parse_rss(payload, limit)

    def topic(self, topic: str, limit: int = 40) -> list[NewsItem]:
        payload = self._get(GOOGLE_NEWS_TOPIC.format(topic=topic.upper()), dict(LOCALE))
        return parse_rss(payload, limit)


# -- query construction ----------------------------------------------------


def member_query(full_name: str, chamber: str) -> str:
    role = "senator" if chamber == "senate" else "congressman OR congresswoman OR representative"
    return f'"{full_name}" ({role} OR congress)'


def ticker_query(symbol: str, company_name: str | None = None) -> str:
    if company_name:
        cleaned = re.sub(
            r"\s*[-\u2013]?\s*(common stock|class [a-z]|inc\.?|corporation|corp\.?|co\.?|ltd\.?)"
            r"\s*$",
            "",
            company_name,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\([A-Z.\-]{1,6}\)", "", cleaned).strip(" -,")
        if cleaned:
            return f'"{cleaned}" (stock OR shares OR earnings)'
    return f'"{symbol}" stock'


def figure_query(name: str) -> str:
    return f'"{name}" (stock OR stocks OR shares OR company OR tariff OR market)'


# -- ingest ----------------------------------------------------------------


def _to_row(item: NewsItem, category: str, subject: str) -> dict[str, Any]:
    scored = score_text(f"{item.title}. {item.summary or ''}")
    return {
        "item_id": _item_id(item.url, f"{category}:{subject}:{item.title}"),
        "category": category,
        "subject": subject,
        "title": item.title,
        "url": item.url,
        "source": item.source,
        "summary": item.summary,
        "published_at": item.published_at,
        "sentiment_score": scored.score,
        "sentiment_label": scored.label,
        "sentiment_terms": {
            "positive": scored.positive_terms,
            "negative": scored.negative_terms,
        },
    }


@dataclass(frozen=True)
class FeedRequest:
    category: str
    subject: str
    query: str | None = None
    topic: str | None = None
    limit: int = 40


def fetch_feeds(
    requests: list[FeedRequest],
    reporter: JobReporter | None = None,
    workers: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch many feeds concurrently, returning storable rows and failures."""
    reporter = reporter or NullReporter()
    client = GoogleNewsClient()
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    total = len(requests)
    if not total:
        return rows, failures

    def _run(request: FeedRequest) -> tuple[FeedRequest, list[NewsItem]]:
        if request.topic:
            return request, client.topic(request.topic, request.limit)
        return request, client.search(request.query or "", request.limit)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as pool:
        futures = {pool.submit(_run, request): request for request in requests}
        for done, future in enumerate(as_completed(futures), start=1):
            request = futures[future]
            label = request.subject or request.category
            reporter.step(f"reading news {done}/{total}: {label}", done, total)
            try:
                _, items = future.result()
            except Exception as exc:
                failures[f"{request.category}:{label}"] = f"{type(exc).__name__}: {exc}"
                continue
            rows.extend(_to_row(item, request.category, request.subject) for item in items)
    return rows, failures


def sync_world_news(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """General world and business headlines for the market-context panel."""
    reporter = reporter or NullReporter()
    requests = [
        FeedRequest(category="world", subject="world", topic="WORLD", limit=40),
        FeedRequest(category="business", subject="business", topic="BUSINESS", limit=40),
        FeedRequest(
            category="business",
            subject="markets",
            query="stock market (rally OR selloff OR outlook OR fed OR inflation)",
            limit=30,
        ),
    ]
    rows, failures = fetch_feeds(requests, reporter)
    added = store.add_news(rows)
    reporter.note(f"world/business news: {added} new of {len(rows)} fetched")
    return {"fetched": len(rows), "added": added, "failures": failures}


def sync_figure_news(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Coverage of each tracked political figure (Trump and anyone else configured)."""
    reporter = reporter or NullReporter()
    figures = settings.politics.figures
    if not figures:
        return {"fetched": 0, "added": 0, "failures": {}}
    requests = [
        FeedRequest(
            category="figure",
            subject=figure.slug,
            query=feed_query,
            limit=settings.politics.max_items_per_feed,
        )
        for figure in figures
        for feed_query in [figure_query(figure.name)]
    ]
    rows, failures = fetch_feeds(requests, reporter)
    added = store.add_news(rows)
    reporter.note(f"figure news: {added} new of {len(rows)} fetched")
    return {"fetched": len(rows), "added": added, "failures": failures}


def sync_member_news(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    limit_members: int = 40,
) -> dict[str, Any]:
    """Recent coverage for the members who trade most actively.

    Capped rather than exhaustive: a few hundred members would be a few
    hundred feed requests per sync, and the ones worth reading about are the
    ones actually filing transactions.
    """
    reporter = reporter or NullReporter()
    members = store.list_politicians(with_trades_only=True)
    if not members:
        return {"fetched": 0, "added": 0, "failures": {}, "members": 0}

    ranked = {
        score["politician_id"]: score["trades_scored"]
        for score in store.list_member_scores(
            horizon_days=settings.ranking.primary_horizon_days
        )
    }
    members.sort(key=lambda member: ranked.get(member["politician_id"], 0), reverse=True)
    selected = members[:limit_members]

    requests = [
        FeedRequest(
            category="member",
            subject=member["politician_id"],
            query=member_query(member["full_name"], member["chamber"]),
            limit=12,
        )
        for member in selected
    ]
    rows, failures = fetch_feeds(requests, reporter)
    added = store.add_news(rows)
    reporter.note(f"member news: {added} new of {len(rows)} across {len(selected)} members")
    return {
        "fetched": len(rows),
        "added": added,
        "failures": failures,
        "members": len(selected),
    }


def sync_ticker_news(
    settings: Settings,
    store: Store,
    symbols: list[str],
    reporter: JobReporter | None = None,
    limit_symbols: int = 40,
) -> dict[str, Any]:
    """Coverage for the tickers currently in play on the opportunity board."""
    reporter = reporter or NullReporter()
    wanted = [symbol.upper() for symbol in symbols][:limit_symbols]
    if not wanted:
        return {"fetched": 0, "added": 0, "failures": {}, "symbols": 0}

    names = company_names(store)
    requests = [
        FeedRequest(
            category="ticker",
            subject=symbol,
            query=ticker_query(symbol, names.get(symbol)),
            limit=12,
        )
        for symbol in wanted
    ]
    rows, failures = fetch_feeds(requests, reporter)
    added = store.add_news(rows)
    reporter.note(f"ticker news: {added} new of {len(rows)} across {len(wanted)} symbols")
    return {
        "fetched": len(rows),
        "added": added,
        "failures": failures,
        "symbols": len(wanted),
    }


def company_names(store: Store) -> dict[str, str]:
    """Best-known company name per ticker, harvested from disclosure lines.

    Filings spell the same company several ways ("Apple Inc. - Common
    Stock", "Apple Inc"); the longest seen is kept because it carries the
    most matchable text for news queries and mention detection.
    """
    names: dict[str, str] = {}
    for trade in store.list_trades(actionable_only=True, limit=100_000):
        symbol = trade.get("symbol")
        if not symbol:
            continue
        candidate = re.sub(r"\([A-Z.\-]{1,6}\)", "", trade.get("asset_name") or "").strip(" -,")
        if not candidate:
            continue
        existing = names.get(symbol)
        if existing is None or len(candidate) > len(existing):
            names[symbol] = candidate
    return names


def sentiment_window(
    store: Store, category: str, days: int = 14
) -> dict[str, dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    return store.news_sentiment_by_subject(category, since)
