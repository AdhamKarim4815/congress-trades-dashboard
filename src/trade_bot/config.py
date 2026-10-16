"""Runtime configuration.

Settings come from three layers, in increasing precedence: model defaults
here, `config/default.toml`, then `TRADE_BOT_`-prefixed environment
variables (`__` separates nesting, e.g. `TRADE_BOT_PRICES__BENCHMARK_SYMBOL`).
Each section of the TOML file maps onto one sub-model below; a missing
section just leaves that sub-model on its defaults, so the app runs with no
config file at all.
"""

from __future__ import annotations

import secrets
import tomllib
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"

AllocationMode = Literal["fixed_dollars", "proportional_to_amount", "percent_of_equity"]


@lru_cache(maxsize=1)
def config_document(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _section(name: str) -> dict[str, Any]:
    section = config_document().get(name, {})
    return section if isinstance(section, dict) else {}


class DisclosureSettings(BaseModel):
    """Which filing years to track, and how hard to hit the public archives."""

    house_years: list[int] = Field(default_factory=list)
    senate_years: list[int] = Field(default_factory=list)
    max_filings_per_sync: int = Field(default=250, ge=1, le=10_000)
    request_timeout_seconds: float = Field(default=45.0, ge=5.0, le=300.0)
    max_concurrent_downloads: int = Field(default=4, ge=1, le=16)
    user_agent: str = "trade-bot/0.2 (personal congressional-disclosure dashboard)"

    @staticmethod
    def _default_years() -> list[int]:
        year = datetime.now(UTC).year
        return [year - 1, year]

    @property
    def resolved_house_years(self) -> list[int]:
        return sorted(self.house_years) or self._default_years()

    @property
    def resolved_senate_years(self) -> list[int]:
        return sorted(self.senate_years) or self._default_years()


class PriceSettings(BaseModel):
    benchmark_symbol: str = "SPY"
    provider_priority: list[str] = Field(
        default_factory=lambda: ["stooq", "yahoo", "twelve-data"]
    )
    history_days: int = Field(default=1825, ge=90, le=10_000)
    # Retained for `ensure(force=...)` callers and ad-hoc scripts. It no
    # longer decides a scheduled refresh: freshness is now whether the cache
    # holds the last final close, which a clock cannot tell you.
    cache_max_age_hours: float = Field(default=12.0, ge=0.0, le=720.0)


class RankingSettings(BaseModel):
    lookback_days: int = Field(default=1095, ge=180, le=7300)
    horizons_days: list[int] = Field(default_factory=lambda: [30, 90, 180, 365])
    primary_horizon_days: int = Field(default=90, ge=1, le=3650)
    min_trades_to_rank: int = Field(default=4, ge=1, le=1000)
    shrinkage_prior_trades: float = Field(default=8.0, ge=0.0, le=1000.0)


class FollowSettings(BaseModel):
    default_allocation_mode: AllocationMode = "fixed_dollars"
    default_allocation_dollars: float = Field(default=1000.0, gt=0)
    mirror_sells: bool = True
    min_amount_low: float = Field(default=1001.0, ge=0)


class RecommendationSettings(BaseModel):
    disclosure_window_days: int = Field(default=45, ge=1, le=365)
    max_results: int = Field(default=25, ge=1, le=500)
    # Walk-forward testing (see `trade_bot.backtest`) finds the edge in the
    # very top of the ranking and nowhere reliable below it: ranks 1-3
    # returned +3.43% mean excess at a 55% beat rate, ranks 4-6 -0.11%, and
    # ranks 7 and lower +1.83% — better than the band above them. Conviction
    # therefore keys off a member's *rank*, and treats everything below the
    # top band as one undifferentiated group, because that is what the
    # measurement supports. The live figures are on `/evidence`.
    trusted_rank: int = Field(default=3, ge=1, le=50)
    credible_rank: int = Field(default=5, ge=1, le=100)
    cluster_weight: float = Field(default=0.35, ge=0, le=1)
    politics_weight: float = Field(default=0.20, ge=0, le=1)


class PoliticalFigure(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    feeds: list[str] = Field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-").replace(".", "")

    @property
    def match_terms(self) -> list[str]:
        return [self.name, *self.aliases]


class PoliticsSettings(BaseModel):
    figures: list[PoliticalFigure] = Field(default_factory=list)
    lookback_days: int = Field(default=30, ge=1, le=365)
    max_items_per_feed: int = Field(default=60, ge=1, le=500)
    request_timeout_seconds: float = Field(default=30.0, ge=5.0, le=300.0)

    @classmethod
    def from_section(cls) -> PoliticsSettings:
        section = _section("politics")
        figures = [PoliticalFigure(**figure) for figure in section.get("figures", [])]
        rest = {key: value for key, value in section.items() if key != "figures"}
        return cls(figures=figures, **rest)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRADE_BOT_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: str = "development"
    database_url: str = "sqlite:///var/trade_bot.sqlite3"
    host: str = "127.0.0.1"
    port: int = 8765
    # Notional equity the mirror portfolios are sized against. No real money
    # or broker is involved anywhere in this application.
    starting_equity: float = Field(default=10_000.0, gt=0)
    session_secret: str = ""
    twelve_data_keyring_name: str = "twelve-data"
    quiver_keyring_name: str = "quiver"
    # Optional. Used only for work no free path handles: reading the scanned
    # paper filings that carry no text layer, and briefing the few tickers at
    # the top of the board. Everything else stays keyless.
    openai_keyring_name: str = "openai"
    openai_model: str = "gpt-4o-mini"

    disclosures: DisclosureSettings = Field(
        default_factory=lambda: DisclosureSettings(**_section("disclosures"))
    )
    prices: PriceSettings = Field(default_factory=lambda: PriceSettings(**_section("prices")))
    ranking: RankingSettings = Field(default_factory=lambda: RankingSettings(**_section("ranking")))
    follow: FollowSettings = Field(default_factory=lambda: FollowSettings(**_section("follow")))
    recommendations: RecommendationSettings = Field(
        default_factory=lambda: RecommendationSettings(**_section("recommendations"))
    )
    politics: PoliticsSettings = Field(default_factory=PoliticsSettings.from_section)

    @property
    def effective_session_secret(self) -> str:
        return self.session_secret or secrets.token_urlsafe(32)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def ensure_runtime_dirs(self) -> None:
        for relative in (
            "var",
            "var/filings",
            "var/prices",
            "var/models",
            "var/reports",
            "artifacts",
        ):
            (self.project_root / relative).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
