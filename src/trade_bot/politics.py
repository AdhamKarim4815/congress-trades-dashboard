"""Turning what political figures say into ticker-level signals.

There is no free, reliable API for a given politician's posts, so this works
from news coverage instead: the figure's headlines are fetched (see
`trade_bot.news`), and each is scanned for companies it names — by cashtag
(`$TSLA`), by bare ticker, or by company name.

Every mention carries a `confidence` and the `matched_terms` that produced
it, because this is materially weaker evidence than a disclosed trade and
the dashboard must present it that way. A company-name match is trusted more
than a bare three-letter ticker, which collides with ordinary English
constantly ("ALL", "IT", "ON", "GO" are all real tickers).
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import PoliticalMention, Stance
from trade_bot.news import company_names
from trade_bot.sentiment import score_text

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Confidence by how the ticker was found.
CONFIDENCE_CASHTAG = 0.9
CONFIDENCE_COMPANY = 0.8
CONFIDENCE_BARE_TICKER = 0.35

# Common English words and abbreviations that are also valid tickers. A bare
# match on any of these is worthless without more context.
TICKER_STOPWORDS = {
    "A", "ALL", "AN", "AND", "ARE", "AS", "AT", "BE", "BIG", "BY", "CEO", "CFO", "DO",
    "EU", "FOR", "GO", "HAS", "HE", "IT", "IN", "IS", "ITS", "NEW", "NOW", "OF", "ON",
    "ONE", "OR", "OUT", "SO", "THE", "TO", "UP", "US", "USA", "WAR", "WAS", "WHO", "WHY",
    "AI", "CNN", "FBI", "GDP", "IPO", "NYSE", "SEC", "UK", "UN", "TV", "PM", "AM", "ET",
    "NEWS", "SAYS", "WILL", "MAY", "CAN", "GET", "END", "TOP", "KEY", "OLD", "OWN", "RUN",
}

# Household names worth matching even when nobody in Congress has disclosed
# trading them, so the political feed is not limited to the disclosure
# universe. Extended at runtime by names harvested from filings.
BASELINE_COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "META": "Meta",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "INTC": "Intel",
    "AMD": "AMD",
    "BA": "Boeing",
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon",
    "NOC": "Northrop Grumman",
    "GD": "General Dynamics",
    "F": "Ford",
    "GM": "General Motors",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "BAC": "Bank of America",
    "WMT": "Walmart",
    "TGT": "Target Corporation",
    "COST": "Costco",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "PFE": "Pfizer",
    "MRNA": "Moderna",
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "MCD": "McDonald's",
    "DIS": "Disney",
    "UBER": "Uber",
    "COIN": "Coinbase",
    "PLTR": "Palantir",
    "DJT": "Trump Media",
    "TSM": "Taiwan Semiconductor",
    "BABA": "Alibaba",
    "SMCI": "Super Micro",
    "CAT": "Caterpillar",
    "DE": "Deere",
    "USS": "US Steel",
    "X": "United States Steel",
    "NKE": "Nike",
    "SBUX": "Starbucks",
}


def build_company_index(store: Store) -> dict[str, str]:
    """Ticker -> best company name, from filings plus the baseline list."""
    index = dict(BASELINE_COMPANIES)
    for symbol, name in company_names(store).items():
        index.setdefault(symbol, name)
    return index


def _name_pattern(name: str) -> re.Pattern[str] | None:
    """A word-boundary matcher for a company name, ignoring corporate suffixes."""
    cleaned = re.sub(
        r"\b(inc|incorporated|corp|corporation|company|co|ltd|plc|holdings|group|"
        r"common stock|class [a-z])\b\.?",
        "",
        name,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^\w&'\- ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Stripping suffix words out of a filing name ("Apple Inc. - Common
    # Stock") leaves dangling separators behind; without trimming them the
    # compiled pattern is "Apple -" and never matches real prose.
    cleaned = cleaned.strip(" -&',")
    if len(cleaned) < 3:
        return None
    return re.compile(rf"\b{re.escape(cleaned)}\b", re.IGNORECASE)


def extract_symbols(
    text: str, company_index: dict[str, str]
) -> list[tuple[str, float, list[str]]]:
    """Find tickers named in a piece of text.

    Returns `(symbol, confidence, matched_terms)`, keeping the strongest
    evidence per symbol so a headline that says both "Tesla" and "$TSLA"
    yields one high-confidence mention rather than two.
    """
    found: dict[str, tuple[float, list[str]]] = {}

    def _record(symbol: str, confidence: float, term: str) -> None:
        existing = found.get(symbol)
        if existing is None:
            found[symbol] = (confidence, [term])
        else:
            best = max(existing[0], confidence)
            terms = existing[1] + [term] if term not in existing[1] else existing[1]
            found[symbol] = (best, terms)

    for match in CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        if symbol in company_index or symbol not in TICKER_STOPWORDS:
            _record(symbol, CONFIDENCE_CASHTAG, f"${symbol}")

    for symbol, name in company_index.items():
        pattern = _name_pattern(name)
        if pattern is not None and pattern.search(text):
            _record(symbol, CONFIDENCE_COMPANY, name)

    for match in BARE_TICKER_RE.finditer(text):
        symbol = match.group(1).upper()
        if symbol in TICKER_STOPWORDS or symbol not in company_index:
            continue
        _record(symbol, CONFIDENCE_BARE_TICKER, symbol)

    return [
        (symbol, confidence, terms) for symbol, (confidence, terms) in sorted(found.items())
    ]


def _mention_id(figure_slug: str, symbol: str, url: str) -> str:
    return hashlib.sha256(f"{figure_slug}|{symbol}|{url}".encode()).hexdigest()[:32]


def extract_mentions(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Scan stored figure news for company mentions and record them."""
    reporter = reporter or NullReporter()
    window = days or settings.politics.lookback_days
    since = datetime.now(UTC) - timedelta(days=window)
    company_index = build_company_index(store)
    figures = {figure.slug: figure for figure in settings.politics.figures}
    if not figures:
        return {"mentions_added": 0, "headlines_scanned": 0, "reason": "no figures configured"}

    mentions: list[PoliticalMention] = []
    scanned = 0
    for slug, figure in figures.items():
        items = store.list_news(category="figure", subject=slug, since=since, limit=400)
        scanned += len(items)
        reporter.step(f"scanning {len(items)} headlines for {figure.name}")
        for item in items:
            text = f"{item['title']}. {item.get('summary') or ''}"
            # Only attribute a mention when the figure is actually named --
            # Google News search is fuzzy and returns adjacent coverage.
            if not any(term.lower() in text.lower() for term in figure.match_terms):
                continue
            scored = score_text(text)
            stance = (
                Stance.BULLISH
                if scored.is_bullish
                else Stance.BEARISH
                if scored.is_bearish
                else Stance.NEUTRAL
            )
            for symbol, confidence, terms in extract_symbols(text, company_index):
                mentions.append(
                    PoliticalMention(
                        mention_id=_mention_id(slug, symbol, item["url"]),
                        figure=figure.name,
                        figure_slug=slug,
                        symbol=symbol,
                        company_name=company_index.get(symbol),
                        stance=stance,
                        headline=item["title"],
                        url=item["url"],
                        source=item.get("source", ""),
                        published_at=item["published_at"],
                        confidence=confidence,
                        matched_terms=terms,
                    )
                )

    added = store.add_mentions(mentions)
    reporter.note(
        f"political mentions: {added} new from {scanned} headlines "
        f"across {len(figures)} figures"
    )
    return {
        "mentions_added": added,
        "mentions_found": len(mentions),
        "headlines_scanned": scanned,
        "figures": len(figures),
    }


def mention_summary(
    store: Store, days: int = 30, min_confidence: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Per-symbol rollup of recent, trustworthy mentions, for the ranking."""
    since = datetime.now(UTC) - timedelta(days=days)
    summary: dict[str, dict[str, Any]] = {}
    for mention in store.list_mentions(since=since, limit=2000):
        if mention["confidence"] < min_confidence:
            continue
        entry = summary.setdefault(
            mention["symbol"],
            {"count": 0, "bullish": 0, "bearish": 0, "figures": set(), "latest": None},
        )
        entry["count"] += 1
        if mention["stance"] == Stance.BULLISH.value:
            entry["bullish"] += 1
        elif mention["stance"] == Stance.BEARISH.value:
            entry["bearish"] += 1
        entry["figures"].add(mention["figure"])
        published = mention["published_at"]
        if entry["latest"] is None or published > entry["latest"]:
            entry["latest"] = published
    for entry in summary.values():
        entry["figures"] = sorted(entry["figures"])
    return summary
