"""Official ticker symbols, from the SEC's own registry.

`symbols.py` resolves a written-out company name by matching it against names
seen in earlier electronic filings. That corpus is free and shares the filers'
vocabulary, but it only knows companies somebody already traded, which is why
roughly 40% of rows read off paper filings resolve to nothing.

The SEC publishes every registrant's CIK, ticker and legal name as a single
JSON file, with no key and no rate limit. It is the authoritative mapping --
if a company has a ticker, it is in here -- so it belongs ahead of the
corpus rather than beside it.

Cached on disk. The file changes when companies list or delist, which is not
something worth re-fetching on every run.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from trade_bot.config import Settings

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# The SEC asks for a contact address in the User-Agent and rejects requests
# without one. This is a condition of use, not a nicety.
_MAX_AGE_DAYS = 30


def _cache(settings: Settings) -> Path:
    directory = settings.project_root / "var" / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "sec_company_tickers.json"


def fetch(settings: Settings, force: bool = False) -> dict[str, Any]:
    """The SEC registry, from disk when it is recent enough."""
    path = _cache(settings)
    if path.exists() and not force:
        age = (datetime.now(UTC).timestamp() - path.stat().st_mtime) / 86_400
        if age < _MAX_AGE_DAYS:
            try:
                return dict(json.loads(path.read_text()))
            except Exception:
                path.unlink(missing_ok=True)

    response = httpx.get(
        SEC_TICKERS_URL,
        timeout=settings.disclosures.request_timeout_seconds,
        headers={"User-Agent": settings.disclosures.user_agent},
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = dict(response.json())
    path.write_text(json.dumps(payload))
    return payload


def build_index(settings: Settings, force: bool = False) -> dict[str, str]:
    """A normalised-name to ticker index over every SEC registrant."""
    from trade_bot.symbols import normalise

    try:
        payload = fetch(settings, force=force)
    except Exception as exc:
        # Absent the registry the corpus index still works, so this degrades
        # rather than failing a sync.
        logger.warning("SEC ticker registry unavailable: %s", exc)
        return {}

    seen: dict[str, set[str]] = {}
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        title, ticker = row.get("title"), row.get("ticker")
        if not title or not ticker:
            continue
        key = normalise(str(title))
        if len(key) >= 5:
            seen.setdefault(key, set()).add(str(ticker).upper())
    # A name registered against two tickers is usually share classes, which is
    # exactly the case where guessing picks the wrong one.
    return {key: next(iter(t)) for key, t in seen.items() if len(t) == 1}


def combined_index(settings: Settings, store: Any) -> dict[str, str]:
    """The SEC registry, overlaid with names seen in real filings.

    The registry is broader; the corpus is closer to how filers actually write
    a company down, so where both know a name the corpus wins. Either alone is
    usable, which matters because the registry needs the network and the
    corpus needs prior filings.
    """
    from trade_bot.symbols import build_index as corpus_index

    corpus = corpus_index(
        [
            (trade["asset_name"], trade["symbol"])
            for trade in store.list_trades(limit=200_000)
            if trade.get("symbol") and trade.get("asset_name")
        ]
    )
    return {**build_index(settings), **corpus}
