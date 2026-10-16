"""Daily price history, from sources that cost nothing and need no key.

Provider order matters and is configurable. Nasdaq's public quote API is the
default primary because it answers without an API key, covers both stocks
and ETFs, and reaches back several years. Stooq and Yahoo follow as
independent fallbacks; Twelve Data sits last and is only tried when the user
has actually stored a key.

Everything is cached to Parquet under `var/prices/`, keyed by symbol, and
refreshed on an age check rather than on every call -- a member's trade from
eight months ago does not need a fresh download to be scored.

The lookups exposed here are date-based rather than index-based on purpose.
Disclosure and transaction dates land on weekends and holidays constantly,
so `close_on_or_after` is what callers actually need: "the first price you
could have transacted at, on or after this date".
"""

from __future__ import annotations

import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.market import last_completed_session
from trade_bot.secrets import get_api_key

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_MONEY_RE = re.compile(r"[^0-9.\-]")


class PriceUnavailableError(RuntimeError):
    """No configured provider could supply a series for this symbol."""


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = _MONEY_RE.sub("", str(value))
    if not text or text in {"-", "."}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _empty_frame() -> pd.DataFrame:
    columns = {column: pd.Series(dtype="float64") for column in PRICE_COLUMNS[1:]}
    frame = pd.DataFrame(columns).assign(date=pd.Series(dtype="datetime64[ns]"))
    return frame[list(PRICE_COLUMNS)]


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame()
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    return frame.reset_index(drop=True)


class DailyPriceProvider(Protocol):
    name: str

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


@dataclass
class NasdaqDailyProvider:
    """Nasdaq's public quote API. No key, no account, stocks and ETFs."""

    timeout: float = 30.0
    name: str = "nasdaq"
    base_url: str = "https://api.nasdaq.com/api/quote"

    def _request(
        self, symbol: str, asset_class: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        params = {
            "assetclass": asset_class,
            "fromdate": start.isoformat(),
            "todate": end.isoformat(),
            "limit": "9999",
        }
        with httpx.Client(timeout=self.timeout, headers=BROWSER_HEADERS) as client:
            response = client.get(f"{self.base_url}/{symbol}/historical", params=params)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data") or {}
        table = data.get("tradesTable") or {}
        return table.get("rows") or []

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        # The endpoint demands the right asset class and simply returns
        # nothing for the wrong one, so an ETF has to be retried as such.
        for asset_class in ("stocks", "etf"):
            try:
                rows = self._request(symbol, asset_class, start, end)
            except httpx.HTTPError:
                rows = []
            if rows:
                break
        if not rows:
            return _empty_frame()
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    pd.Series([str(row.get("date") or "") for row in rows]),
                    format="%m/%d/%Y",
                    errors="coerce",
                ),
                "open": [_to_float(row.get("open")) for row in rows],
                "high": [_to_float(row.get("high")) for row in rows],
                "low": [_to_float(row.get("low")) for row in rows],
                "close": [_to_float(row.get("close")) for row in rows],
                "volume": [_to_float(row.get("volume")) for row in rows],
            }
        )
        return _tidy(frame)


@dataclass
class StooqDailyProvider:
    """Stooq's CSV export. No key; occasionally gated behind a JS check."""

    timeout: float = 30.0
    name: str = "stooq"
    base_url: str = "https://stooq.com/q/d/l/"

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "s": f"{symbol.lower().replace('.', '-')}.us",
            "i": "d",
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
        }
        with httpx.Client(timeout=self.timeout, headers=BROWSER_HEADERS) as client:
            response = client.get(self.base_url, params=params)
            response.raise_for_status()
            text = response.text
        if not text.startswith("Date"):
            return _empty_frame()
        raw = pd.read_csv(io.StringIO(text))

        def column(name: str) -> pd.Series:
            # Stooq omits Volume for some instruments.
            if name not in raw.columns:
                return pd.Series([float("nan")] * len(raw))
            return pd.to_numeric(raw[name], errors="coerce")

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["Date"], errors="coerce"),
                "open": column("Open"),
                "high": column("High"),
                "low": column("Low"),
                "close": column("Close"),
                "volume": column("Volume"),
            }
        )
        return _tidy(frame)


@dataclass
class YahooDailyProvider:
    """Yahoo's chart endpoint. No key; rate-limits aggressively by IP."""

    timeout: float = 30.0
    name: str = "yahoo"
    base_url: str = "https://query1.finance.yahoo.com/v8/finance/chart"

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "period1": str(int(datetime.combine(start, datetime.min.time()).timestamp())),
            "period2": str(int(datetime.combine(end, datetime.max.time()).timestamp())),
            "interval": "1d",
        }
        with httpx.Client(timeout=self.timeout, headers=BROWSER_HEADERS) as client:
            response = client.get(f"{self.base_url}/{symbol}", params=params)
            response.raise_for_status()
            payload = response.json()
        results = ((payload.get("chart") or {}).get("result")) or []
        if not results:
            return _empty_frame()
        result = results[0]
        stamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(pd.Series(stamps, dtype="float64"), unit="s").dt.normalize(),
                "open": pd.to_numeric(pd.Series(quote.get("open") or []), errors="coerce"),
                "high": pd.to_numeric(pd.Series(quote.get("high") or []), errors="coerce"),
                "low": pd.to_numeric(pd.Series(quote.get("low") or []), errors="coerce"),
                "close": pd.to_numeric(pd.Series(quote.get("close") or []), errors="coerce"),
                "volume": pd.to_numeric(pd.Series(quote.get("volume") or []), errors="coerce"),
            }
        )
        return _tidy(frame)


@dataclass
class TwelveDataDailyProvider:
    """Twelve Data. Needs a key, so it is only ever tried when one is stored."""

    api_key: str
    timeout: float = 30.0
    name: str = "twelve-data"
    base_url: str = "https://api.twelvedata.com/time_series"

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": "1day",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "outputsize": "5000",
            "apikey": self.api_key,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.base_url, params=params)
            response.raise_for_status()
            payload = response.json()
        if payload.get("status") == "error":
            raise PriceUnavailableError(payload.get("message", "Twelve Data rejected the request"))
        values = payload.get("values") or []
        if not values:
            return _empty_frame()
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime([row.get("datetime") for row in values], errors="coerce"),
                "open": [_to_float(row.get("open")) for row in values],
                "high": [_to_float(row.get("high")) for row in values],
                "low": [_to_float(row.get("low")) for row in values],
                "close": [_to_float(row.get("close")) for row in values],
                "volume": [_to_float(row.get("volume")) for row in values],
            }
        )
        return _tidy(frame)


def build_providers(settings: Settings) -> list[DailyPriceProvider]:
    """Assemble the provider chain named in configuration.

    Keyed providers are skipped silently when no key is stored, which is what
    keeps the whole application working at zero cost out of the box.
    """
    timeout = 30.0
    available: dict[str, DailyPriceProvider] = {
        "nasdaq": NasdaqDailyProvider(timeout=timeout),
        "stooq": StooqDailyProvider(timeout=timeout),
        "yahoo": YahooDailyProvider(timeout=timeout),
    }
    key = get_api_key(settings.twelve_data_keyring_name)
    if key:
        available["twelve-data"] = TwelveDataDailyProvider(api_key=key, timeout=timeout)
    chain = [
        available[name] for name in settings.prices.provider_priority if name in available
    ]
    return chain or [available["nasdaq"]]


class PriceRepository:
    """Parquet-backed cache of daily bars, with date-aware lookups."""

    def __init__(
        self, settings: Settings, providers: list[DailyPriceProvider] | None = None
    ) -> None:
        self.settings = settings
        self.root = settings.project_root / "var" / "prices"
        self.root.mkdir(parents=True, exist_ok=True)
        self._providers = providers
        # Frames are read by concurrent web requests while a sync writes them.
        self._memory: dict[str, pd.DataFrame] = {}
        self._memory_lock = threading.Lock()

    @property
    def providers(self) -> list[DailyPriceProvider]:
        if self._providers is None:
            self._providers = build_providers(self.settings)
        return self._providers

    def path_for(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "-")
        return self.root / f"{safe}.parquet"

    def is_fresh(self, symbol: str) -> bool:
        """Whether the cache already holds the last final close.

        Freshness used to mean "written recently", which cannot tell stale
        apart from unchanged. On a daily schedule every file is always older
        than the twelve-hour window, so the setting could never prevent a
        fetch, and on weekends and holidays two thousand symbols were re-read
        to receive exactly what was already on disk.
        """
        path = self.path_for(symbol)
        if not path.exists():
            return False
        frame = self.load(symbol)
        if frame.empty:
            return False
        last_held = pd.Timestamp(frame["date"].iloc[-1]).date()
        return last_held >= last_completed_session()

    def load(self, symbol: str) -> pd.DataFrame:
        key = symbol.upper()
        with self._memory_lock:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        path = self.path_for(key)
        frame = pd.read_parquet(path) if path.exists() else _empty_frame()
        with self._memory_lock:
            self._memory[key] = frame
        return frame

    def save(self, symbol: str, frame: pd.DataFrame) -> None:
        key = symbol.upper()
        frame.to_parquet(self.path_for(key), index=False)
        with self._memory_lock:
            self._memory[key] = frame

    def fetch(self, symbol: str) -> tuple[pd.DataFrame, str]:
        """Fetch a symbol from the first provider that returns anything."""
        end = date.today()
        start = end - timedelta(days=self.settings.prices.history_days)
        errors: list[str] = []
        for provider in self.providers:
            try:
                frame = provider.daily(symbol.upper(), start, end)
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}")
                continue
            if not frame.empty:
                return frame, provider.name
            errors.append(f"{provider.name}: no rows")
        raise PriceUnavailableError(
            f"no price provider returned data for {symbol.upper()} ({'; '.join(errors)})"
        )

    def ensure(
        self,
        symbols: list[str],
        reporter: JobReporter | None = None,
        force: bool = False,
        workers: int = 8,
    ) -> dict[str, Any]:
        """Refresh the cache for `symbols`, skipping anything already fresh.

        Fetches run on a small thread pool: a sync touches a few hundred
        tickers and each provider round-trip is a second or two, which is
        minutes of pure waiting done serially. Writes happen on this thread
        as results arrive, so the Parquet cache is only ever touched here.
        """
        reporter = reporter or NullReporter()
        wanted = sorted({symbol.upper() for symbol in symbols if symbol})
        stale = [
            symbol for symbol in wanted if force or not self.is_fresh(symbol)
        ]
        skipped = [symbol for symbol in wanted if symbol not in set(stale)]
        total = len(stale)
        fetched: list[str] = []
        failed: dict[str, str] = {}
        providers_used: dict[str, int] = {}

        if total:
            reporter.step(f"fetching prices for {total} symbols", 0, total)
            with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as pool:
                futures = {
                    pool.submit(self.fetch, symbol): symbol for symbol in stale
                }
                for done, future in enumerate(as_completed(futures), start=1):
                    symbol = futures[future]
                    reporter.step(f"prices {done}/{total}: {symbol}", done, total)
                    try:
                        frame, provider_name = future.result()
                    except PriceUnavailableError as exc:
                        failed[symbol] = str(exc)
                        continue
                    except Exception as exc:
                        failed[symbol] = f"{type(exc).__name__}: {exc}"
                        continue
                    self.save(symbol, frame)
                    fetched.append(symbol)
                    providers_used[provider_name] = providers_used.get(provider_name, 0) + 1

        if failed:
            reporter.note(
                f"{len(failed)} symbols had no price data available",
                level="warn",
                detail=", ".join(sorted(failed)[:20]),
            )
        return {
            "requested": len(wanted),
            "fetched": len(fetched),
            "skipped": len(skipped),
            "failed": len(failed),
            "failures": failed,
            "providers": providers_used,
        }

    # -- lookups -----------------------------------------------------------

    def close_on_or_after(self, symbol: str, when: date) -> tuple[float | None, date | None]:
        """First close on or after `when`.

        Disclosure and transaction dates land on weekends and holidays all the
        time; this answers the question that actually matters, which is what
        price was available once the market next opened.
        """
        frame = self.load(symbol)
        if frame.empty:
            return None, None
        stamp = pd.Timestamp(when)
        eligible = frame[frame["date"] >= stamp]
        if eligible.empty:
            return None, None
        row = eligible.iloc[0]
        return float(row["close"]), row["date"].date()

    def close_on_or_before(self, symbol: str, when: date) -> tuple[float | None, date | None]:
        frame = self.load(symbol)
        if frame.empty:
            return None, None
        eligible = frame[frame["date"] <= pd.Timestamp(when)]
        if eligible.empty:
            return None, None
        row = eligible.iloc[-1]
        return float(row["close"]), row["date"].date()

    def latest_close(self, symbol: str) -> tuple[float | None, date | None]:
        frame = self.load(symbol)
        if frame.empty:
            return None, None
        row = frame.iloc[-1]
        return float(row["close"]), row["date"].date()

    def return_between(
        self, symbol: str, start: date, end: date | None = None
    ) -> tuple[float | None, float | None, float | None]:
        """Simple return from the first close on/after `start` to `end`.

        Returns `(pct_return, entry_price, exit_price)`; any leg that cannot be
        priced yields `None` rather than a silently wrong zero.
        """
        entry, _ = self.close_on_or_after(symbol, start)
        if entry is None or entry <= 0:
            return None, None, None
        if end is None:
            exit_price, _ = self.latest_close(symbol)
        else:
            exit_price, _ = self.close_on_or_before(symbol, end)
            if exit_price is None:
                exit_price, _ = self.latest_close(symbol)
        if exit_price is None:
            return None, entry, None
        return (exit_price - entry) / entry, entry, exit_price

    def coverage(self) -> dict[str, Any]:
        files = sorted(self.root.glob("*.parquet"))
        newest = max((path.stat().st_mtime for path in files), default=None)
        return {
            "symbols": len(files),
            "last_updated": (
                datetime.fromtimestamp(newest, tz=UTC).isoformat() if newest else None
            ),
        }

    def cached_symbols(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.parquet"))
