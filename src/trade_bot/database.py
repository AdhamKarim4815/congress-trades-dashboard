"""Durable storage for disclosures, watchlists and derived scores.

The event log (`events`) and key/value `app_state` carry over from the
previous design and still serve the same purpose: an append-only audit trail
of everything ingested, plus small pieces of mutable status (last sync time,
per-source health) that any process can read while another writes, over
SQLite's WAL mode.

Everything else is new and models the copy-trading domain directly.
Disclosed trades are keyed by the deterministic `trade_id` from
`trade_bot.domain`, so re-parsing the same filing is idempotent -- inserts
are filtered against the ids already present rather than relying on a
dialect-specific upsert.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    inspect,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from trade_bot.domain import (
    DisclosedTrade,
    DomainRecord,
    Filing,
    MemberScore,
    MirrorFill,
    PoliticalMention,
    Politician,
    TradeSide,
    Watchlist,
)

logger = logging.getLogger(__name__)

# Alembic ships as a dependency but is deliberately not wired up: this schema
# is still settling, and `Base.metadata.create_all` is fine for a local
# single-user database that is routinely re-created from scratch. Switch to
# migrations once the disclosure schema stops moving.


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """Re-attach UTC to a timestamp read back from storage.

    SQLite has no native timestamp type and hands back naive datetimes even
    for `DateTime(timezone=True)` columns. Domain records reject naive
    timestamps on purpose, so every datetime leaving the store is normalised
    here rather than at each of the dozen call sites.
    """
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def normalize_database_url(url: str) -> str:
    """Repair a Postgres URL whose password was pasted raw.

    Connection strings are handed out with a `[YOUR-PASSWORD]` placeholder,
    and real passwords routinely contain characters that are structural in a
    URL. `@` is the dangerous one: standard parsers split credentials from the
    host on the *first* `@`, so `...://user:pa@ss@host/db` is read as host
    `ss@host` and the connection dies with "failed to resolve host" — naming a
    host nobody typed, which makes the error look like a DNS problem rather
    than a quoting one.

    Splitting on the *last* `@` recovers the intended boundary, since the host
    section can never contain one. The password is then percent-encoded, and
    surrounding whitespace and quotes are stripped, because a value copied
    into a CI secret box frequently carries a trailing newline.

    A password already containing `%` is left alone: it is assumed to be
    encoded already, and re-encoding would corrupt it.
    """
    url = url.strip().strip("'\"").replace("\n", "").replace("\r", "")
    if not url.startswith(("postgresql", "postgres:")):
        return url

    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    # Split on the last `@` *before* splitting off the path: a password may
    # contain `/` as legitimately as `@`, and taking the path first would cut
    # the credentials in half.
    credentials, at_sign, host_and_path = rest.rpartition("@")
    if not at_sign:
        return url
    host, slash, tail = host_and_path.partition("/")

    user, colon, password = credentials.partition(":")
    if colon and password and "%" not in password:
        password = quote(password, safe="")
    credentials = f"{user}:{password}" if colon else user
    return f"{scheme}://{credentials}@{host}{slash}{tail}"


def describe_database_url(url: str) -> str:
    """A connection target safe to print: never the password.

    Failures here are almost always in the host, port or user, and a log line
    that shows them turns a mystifying error into an obvious one.
    """
    url = normalize_database_url(url)
    if not url.startswith(("postgresql", "postgres:")):
        return url
    _, _, rest = url.partition("://")
    credentials, at_sign, host_and_path = rest.rpartition("@")
    host, _, tail = host_and_path.partition("/")
    user = credentials.partition(":")[0] if at_sign else ""
    database = tail.partition("?")[0] or "(default)"
    return f"host={host} user={user or '(none)'} database={database}"


def _engine_options(database_url: str) -> dict[str, Any]:
    """Connection settings appropriate to the backend.

    Local SQLite needs nothing. A hosted Postgres (Supabase) sits behind a
    connection pooler that drops idle connections, so connections are
    pre-pinged and recycled well inside that window; a scheduled ingest job
    that runs for minutes would otherwise fail on a stale socket. Supabase's
    transaction-mode pooler also cannot hold server-side prepared statements,
    so those are disabled.
    """
    if not database_url.startswith(("postgresql", "postgres:")):
        return {}
    return {
        "pool_pre_ping": True,
        "pool_recycle": 240,
        # Deliberately small. This database is shared with several other
        # applications and the instance allows 60 connections in total, of
        # which the others already hold about twenty. Every database write
        # here happens on one thread anyway -- the concurrency in this app is
        # in HTTP fetching, not SQL -- so a large pool would reserve capacity
        # from its neighbours for nothing.
        "pool_size": 3,
        "max_overflow": 2,
        "connect_args": {"prepare_threshold": None},
    }


class Base(DeclarativeBase):
    pass


class EventRow(Base):
    __tablename__ = "ct_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    subject: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), index=True)
    causation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)


class StateRow(Base):
    __tablename__ = "ct_app_state"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PoliticianRow(Base):
    __tablename__ = "ct_politicians"

    politician_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), index=True)
    first_name: Mapped[str] = mapped_column(String(120), default="")
    last_name: Mapped[str] = mapped_column(String(120), default="", index=True)
    chamber: Mapped[str] = mapped_column(String(16), index=True)
    state: Mapped[str] = mapped_column(String(8), default="")
    district: Mapped[str | None] = mapped_column(String(8), nullable=True)
    party: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    bioguide_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    photo_url: Mapped[str | None] = mapped_column(String(400), nullable=True)


class CommitteeSeatRow(Base):
    """One member's seat on one committee.

    Replaced wholesale on every sync rather than merged: committee membership
    changes with the Congress, and a member who has left a committee must stop
    being credited with its jurisdiction immediately.
    """

    __tablename__ = "ct_committee_seats"

    politician_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    committee_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    committee_name: Mapped[str] = mapped_column(String(240), default="")
    chamber: Mapped[str] = mapped_column(String(16), default="", index=True)
    title: Mapped[str] = mapped_column(String(80), default="")
    rank: Mapped[int] = mapped_column(Integer, default=0)
    jurisdiction: Mapped[str] = mapped_column(String(120), default="", index=True)


class FilingRow(Base):
    __tablename__ = "ct_filings"

    filing_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    politician_id: Mapped[str] = mapped_column(String(120), index=True)
    chamber: Mapped[str] = mapped_column(String(16), index=True)
    doc_id: Mapped[str] = mapped_column(String(64), index=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    filing_type: Mapped[str] = mapped_column(String(8), default="P")
    filed_at: Mapped[date] = mapped_column(Date, index=True)
    source_url: Mapped[str] = mapped_column(String(400), default="")
    parsed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    parse_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)


class TradeRow(Base):
    __tablename__ = "ct_disclosed_trades"

    trade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filing_id: Mapped[str] = mapped_column(String(120), index=True)
    politician_id: Mapped[str] = mapped_column(String(120), index=True)
    chamber: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    asset_name: Mapped[str] = mapped_column(String(400), default="")
    asset_class: Mapped[str] = mapped_column(String(16), default="other", index=True)
    transaction_type: Mapped[str] = mapped_column(String(20), index=True)
    owner: Mapped[str] = mapped_column(String(16), default="unknown")
    transaction_date: Mapped[date] = mapped_column(Date, index=True)
    disclosure_date: Mapped[date] = mapped_column(Date, index=True)
    amount_low: Mapped[float] = mapped_column(Float, default=0)
    amount_high: Mapped[float] = mapped_column(Float, default=0)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(String(400), default="")


class WatchlistRow(Base):
    __tablename__ = "ct_watchlists"

    watchlist_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    politician_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    allocation_mode: Mapped[str] = mapped_column(String(32), default="fixed_dollars")
    allocation_value: Mapped[float] = mapped_column(Float, default=1000.0)
    mirror_sells: Mapped[bool] = mapped_column(Boolean, default=True)
    min_amount_low: Mapped[float] = mapped_column(Float, default=0)
    starting_equity: Mapped[float] = mapped_column(Float, default=10_000.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MirrorFillRow(Base):
    __tablename__ = "ct_mirror_fills"

    fill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    watchlist_id: Mapped[str] = mapped_column(String(64), index=True)
    trade_id: Mapped[str] = mapped_column(String(64), index=True)
    politician_id: Mapped[str] = mapped_column(String(120), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(20))
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    notional: Mapped[float] = mapped_column(Float)
    priced_on: Mapped[date] = mapped_column(Date, index=True)
    price_basis: Mapped[str] = mapped_column(String(32), default="disclosure_date")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class MentionRow(Base):
    __tablename__ = "ct_political_mentions"

    mention_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    figure: Mapped[str] = mapped_column(String(120), index=True)
    figure_slug: Mapped[str] = mapped_column(String(120), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    stance: Mapped[str] = mapped_column(String(16), default="neutral", index=True)
    headline: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(600))
    source: Mapped[str] = mapped_column(String(200), default="")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0)
    matched_terms_json: Mapped[str] = mapped_column(Text, default="[]")


class NewsRow(Base):
    """Headlines, scored for sentiment at ingest time.

    `category` says what kind of subject this was fetched for (world,
    business, member, ticker, figure) and `subject` is that subject's id, so
    one table serves the world feed, a member's page and a ticker's page.
    """

    __tablename__ = "ct_news_items"

    item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(24), index=True)
    subject: Mapped[str] = mapped_column(String(160), default="", index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(800))
    source: Mapped[str] = mapped_column(String(200), default="")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0, index=True)
    sentiment_label: Mapped[str] = mapped_column(String(16), default="neutral", index=True)
    sentiment_terms_json: Mapped[str] = mapped_column(Text, default="{}")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemberScoreRow(Base):
    """One member's record over one horizon and one side of the book.

    `side` is part of the key because a member has three records, not one:
    everything, their buying, and their selling. They routinely disagree --
    one member ranked second overall on twenty-five sales and no purchases at
    all -- so a single row per horizon could not answer "whose buying should
    I copy" without guessing.
    """

    __tablename__ = "ct_member_scores"

    politician_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    horizon_days: Mapped[int] = mapped_column(Integer, primary_key=True)
    side: Mapped[str] = mapped_column(String(8), primary_key=True, default="all")
    trades_scored: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0)
    raw_excess_return: Mapped[float] = mapped_column(Float, default=0)
    median_excess_return: Mapped[float] = mapped_column(Float, default=0)
    score: Mapped[float] = mapped_column(Float, default=0, index=True)
    total_disclosed_dollars: Mapped[float] = mapped_column(Float, default=0)
    average_reporting_lag_days: Mapped[float] = mapped_column(Float, default=0)
    best_symbol: Mapped[str | None] = mapped_column(String(24), nullable=True)
    best_symbol_excess: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_symbol_side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Actionable trades of theirs that no provider could price, so they are in
    # none of the figures beside them. Usually a company since acquired,
    # delisted or renamed -- which is exactly the kind of omission a record
    # should not be able to hide.
    trades_unpriced: Mapped[int] = mapped_column(Integer, default=0)
    model_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    rankable: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def _politician_dict(row: PoliticianRow) -> dict[str, Any]:
    # Names arrive from the disclosure portals with stray punctuation -- the
    # Senate index lists one sitting senator as "Moran," -- which then reads as
    # "Jerry Moran,, ranked #6" wherever a name is followed by a clause.
    # Cleaned on the way out rather than on the way in, so the stored value
    # still matches the source and no backfill is needed to fix the display.
    return {
        "politician_id": row.politician_id,
        "full_name": row.full_name.strip(" ,;"),
        "first_name": row.first_name,
        "last_name": row.last_name.strip(" ,;"),
        "chamber": row.chamber,
        "state": row.state,
        "district": row.district,
        "party": row.party,
        "bioguide_id": row.bioguide_id,
        "photo_url": row.photo_url,
        "display_state": f"{row.state}-{row.district}" if row.district else row.state,
    }


def _trade_dict(row: TradeRow) -> dict[str, Any]:
    midpoint = (row.amount_low + row.amount_high) / 2 if row.amount_high else row.amount_low
    return {
        "trade_id": row.trade_id,
        "filing_id": row.filing_id,
        "politician_id": row.politician_id,
        "chamber": row.chamber,
        "symbol": row.symbol,
        "asset_name": row.asset_name,
        "asset_class": row.asset_class,
        "transaction_type": row.transaction_type,
        "owner": row.owner,
        "transaction_date": row.transaction_date,
        "disclosure_date": row.disclosure_date,
        "amount_low": row.amount_low,
        "amount_high": row.amount_high,
        "amount_midpoint": midpoint,
        "reporting_lag_days": (row.disclosure_date - row.transaction_date).days,
        "comment": row.comment,
        "source_url": row.source_url,
    }


def _filing_dict(row: FilingRow) -> dict[str, Any]:
    return {
        "filing_id": row.filing_id,
        "politician_id": row.politician_id,
        "chamber": row.chamber,
        "doc_id": row.doc_id,
        "year": row.year,
        "filing_type": row.filing_type,
        "filed_at": row.filed_at,
        "source_url": row.source_url,
        "parsed": row.parsed,
        "parse_error": row.parse_error,
        "transaction_count": row.transaction_count,
    }


def _watchlist_dict(row: WatchlistRow) -> dict[str, Any]:
    return {
        "watchlist_id": row.watchlist_id,
        "name": row.name,
        "politician_ids": json.loads(row.politician_ids_json),
        "allocation_mode": row.allocation_mode,
        "allocation_value": row.allocation_value,
        "mirror_sells": row.mirror_sells,
        "min_amount_low": row.min_amount_low,
        "starting_equity": row.starting_equity,
        "active": row.active,
        "created_at": as_utc(row.created_at),
    }


def _mention_dict(row: MentionRow) -> dict[str, Any]:
    return {
        "mention_id": row.mention_id,
        "figure": row.figure,
        "figure_slug": row.figure_slug,
        "symbol": row.symbol,
        "company_name": row.company_name,
        "stance": row.stance,
        "headline": row.headline,
        "url": row.url,
        "source": row.source,
        "published_at": as_utc(row.published_at),
        "confidence": row.confidence,
        "matched_terms": json.loads(row.matched_terms_json),
    }


def _score_dict(row: MemberScoreRow) -> dict[str, Any]:
    return {
        "politician_id": row.politician_id,
        "horizon_days": row.horizon_days,
        "side": row.side,
        "trades_scored": row.trades_scored,
        "win_rate": row.win_rate,
        "raw_excess_return": row.raw_excess_return,
        "median_excess_return": row.median_excess_return,
        "score": row.score,
        "total_disclosed_dollars": row.total_disclosed_dollars,
        "average_reporting_lag_days": row.average_reporting_lag_days,
        "best_symbol": row.best_symbol,
        "best_symbol_excess": row.best_symbol_excess,
        "best_symbol_side": row.best_symbol_side,
        "trades_unpriced": row.trades_unpriced,
        "model_probability": row.model_probability,
        "rankable": row.rankable,
        "computed_at": as_utc(row.computed_at),
    }


def _fill_dict(row: MirrorFillRow) -> dict[str, Any]:
    return {
        "fill_id": row.fill_id,
        "watchlist_id": row.watchlist_id,
        "trade_id": row.trade_id,
        "politician_id": row.politician_id,
        "symbol": row.symbol,
        "side": row.side,
        "shares": row.shares,
        "price": row.price,
        "notional": row.notional,
        "priced_on": row.priced_on,
        "price_basis": row.price_basis,
        "note": row.note,
    }


class Store:
    def __init__(self, database_url: str) -> None:
        database_url = normalize_database_url(database_url)
        self.database_url = database_url
        if database_url.startswith("sqlite:///"):
            path = Path(database_url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True, **_engine_options(database_url))
        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._configure_sqlite)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    def initialize(self) -> None:
        self._migrate_member_scores()
        Base.metadata.create_all(self.engine)

    def _migrate_member_scores(self) -> None:
        """Rebuild `ct_member_scores` whenever the model has a column it lacks.

        `create_all` creates missing tables and never alters an existing one,
        so a schema change that joins the primary key needs saying out loud.

        Dropping is the right move here and nowhere else in this schema: every
        row is derived, the `ranking` stage rewrites the whole table on each
        run, and there is no source of truth being discarded. The pages handle
        an empty score table by saying members are unscored, so the worst case
        between this and the next ranking run is a page that admits it does not
        know yet -- which is the failure mode this project prefers.

        Filings and trades are never touched by anything like this.
        """
        inspector = inspect(self.engine)
        if not inspector.has_table(MemberScoreRow.__tablename__):
            return
        columns = {column["name"] for column in inspector.get_columns(MemberScoreRow.__tablename__)}
        expected = {column.name for column in MemberScoreRow.__table__.columns}
        missing = expected - columns
        if not missing:
            return
        logger.warning(
            "rebuilding %s: it is missing %s; scores are derived and the next "
            "`trade-bot refresh --stage ranking` restores them",
            MemberScoreRow.__tablename__,
            ", ".join(sorted(missing)),
        )
        Base.metadata.tables[MemberScoreRow.__tablename__].drop(self.engine)

    # -- audit log ---------------------------------------------------------

    def append_event(self, record: DomainRecord, subject: str | None = None) -> None:
        with self.sessions.begin() as session:
            session.add(
                EventRow(
                    id=record.event_id,
                    event_type=record.__class__.__name__,
                    subject=subject,
                    occurred_at=record.occurred_at,
                    correlation_id=record.correlation_id,
                    causation_id=record.causation_id,
                    payload_json=record.model_dump_json(),
                )
            )

    def audit(self, subject: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(EventRow)
            if subject is not None:
                statement = statement.where(EventRow.subject == subject)
            rows = session.scalars(
                statement.order_by(EventRow.occurred_at.desc()).limit(limit)
            ).all()
        return [
            {
                "id": row.id,
                "type": row.event_type,
                "subject": row.subject,
                "occurred_at": as_utc(row.occurred_at),
                "payload": json.loads(row.payload_json),
            }
            for row in rows
        ]

    # -- key/value state ---------------------------------------------------

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            row = session.get(StateRow, key)
            payload = json.dumps(value, default=str)
            if row is None:
                session.add(StateRow(key=key, value_json=payload, updated_at=utc_now()))
            else:
                row.value_json = payload
                row.updated_at = utc_now()

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(StateRow, key)
            return json.loads(row.value_json) if row is not None else None

    def list_states(self, prefix: str) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(StateRow)
                .where(StateRow.key.like(f"{prefix}%"))
                .order_by(StateRow.updated_at.desc())
            ).all()
        return [
            {"key": row.key, "updated_at": as_utc(row.updated_at), **json.loads(row.value_json)}
            for row in rows
        ]

    # -- politicians -------------------------------------------------------

    def upsert_politicians(self, politicians: Iterable[Politician]) -> int:
        """Insert new members and enrich existing ones.

        Enrichment is deliberately additive: the House filing index knows a
        member's name and district but never their party or bioguide id, and
        the congress-legislators roster knows those but nothing about
        filings. Whichever arrives second must not blank out what the first
        already established.
        """
        added = 0
        with self.sessions.begin() as session:
            for politician in politicians:
                row = session.get(PoliticianRow, politician.politician_id)
                if row is None:
                    session.add(
                        PoliticianRow(
                            politician_id=politician.politician_id,
                            full_name=politician.full_name,
                            first_name=politician.first_name,
                            last_name=politician.last_name,
                            chamber=politician.chamber.value,
                            state=politician.state,
                            district=politician.district,
                            party=politician.party,
                            bioguide_id=politician.bioguide_id,
                            photo_url=politician.photo_url,
                        )
                    )
                    added += 1
                    continue
                row.full_name = politician.full_name or row.full_name
                row.first_name = politician.first_name or row.first_name
                row.last_name = politician.last_name or row.last_name
                row.state = politician.state or row.state
                row.district = politician.district or row.district
                row.party = politician.party or row.party
                row.bioguide_id = politician.bioguide_id or row.bioguide_id
                row.photo_url = politician.photo_url or row.photo_url
        return added

    def list_politicians(
        self, chamber: str | None = None, with_trades_only: bool = False
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(PoliticianRow)
            if chamber is not None:
                statement = statement.where(PoliticianRow.chamber == chamber)
            if with_trades_only:
                statement = statement.where(
                    PoliticianRow.politician_id.in_(select(TradeRow.politician_id).distinct())
                )
            rows = session.scalars(
                statement.order_by(PoliticianRow.last_name, PoliticianRow.first_name)
            ).all()
        return [_politician_dict(row) for row in rows]

    def get_politician(self, politician_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(PoliticianRow, politician_id)
            return _politician_dict(row) if row is not None else None

    # -- committee seats ---------------------------------------------------

    def replace_committee_memberships(self, seats: Sequence[dict[str, Any]]) -> int:
        with self.sessions.begin() as session:
            session.execute(delete(CommitteeSeatRow))
            session.add_all(
                [
                    CommitteeSeatRow(
                        politician_id=seat["politician_id"],
                        committee_id=seat["committee_id"],
                        committee_name=seat["committee_name"],
                        chamber=seat.get("chamber", ""),
                        title=seat.get("title", ""),
                        rank=seat.get("rank", 0),
                        jurisdiction=seat.get("jurisdiction", ""),
                    )
                    for seat in seats
                ]
            )
        return len(seats)

    def list_committee_memberships(
        self, politician_id: str | None = None, mapped_only: bool = False
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(CommitteeSeatRow)
            if politician_id is not None:
                statement = statement.where(CommitteeSeatRow.politician_id == politician_id)
            if mapped_only:
                statement = statement.where(CommitteeSeatRow.jurisdiction != "")
            rows = session.scalars(statement.order_by(CommitteeSeatRow.rank)).all()
        return [
            {
                "politician_id": row.politician_id,
                "committee_id": row.committee_id,
                "committee_name": row.committee_name,
                "chamber": row.chamber,
                "title": row.title,
                "rank": row.rank,
                "jurisdiction": row.jurisdiction,
            }
            for row in rows
        ]

    # -- filings -----------------------------------------------------------

    def known_filing_ids(self, chamber: str | None = None) -> set[str]:
        with self.sessions() as session:
            statement = select(FilingRow.filing_id)
            if chamber is not None:
                statement = statement.where(FilingRow.chamber == chamber)
            return set(session.scalars(statement).all())

    def upsert_filing(self, filing: Filing) -> None:
        with self.sessions.begin() as session:
            row = session.get(FilingRow, filing.filing_id)
            if row is None:
                session.add(
                    FilingRow(
                        filing_id=filing.filing_id,
                        politician_id=filing.politician_id,
                        chamber=filing.chamber.value,
                        doc_id=filing.doc_id,
                        year=filing.year,
                        filing_type=filing.filing_type,
                        filed_at=filing.filed_at,
                        source_url=filing.source_url,
                        parsed=filing.parsed,
                        parse_error=filing.parse_error,
                        transaction_count=filing.transaction_count,
                    )
                )
                return
            row.parsed = filing.parsed
            row.parse_error = filing.parse_error
            row.transaction_count = filing.transaction_count
            row.source_url = filing.source_url or row.source_url

    def list_filings(
        self,
        politician_id: str | None = None,
        parsed: bool | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(FilingRow)
            if politician_id is not None:
                statement = statement.where(FilingRow.politician_id == politician_id)
            if parsed is not None:
                statement = statement.where(FilingRow.parsed.is_(parsed))
            rows = session.scalars(
                statement.order_by(FilingRow.filed_at.desc()).limit(limit)
            ).all()
        return [_filing_dict(row) for row in rows]

    def filing_counts(self) -> dict[str, int]:
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(FilingRow)) or 0
            parsed = (
                session.scalar(
                    select(func.count()).select_from(FilingRow).where(FilingRow.parsed.is_(True))
                )
                or 0
            )
            failed = (
                session.scalar(
                    select(func.count())
                    .select_from(FilingRow)
                    .where(FilingRow.parse_error.is_not(None))
                )
                or 0
            )
        return {"total": total, "parsed": parsed, "failed": failed, "pending": total - parsed}

    # -- disclosed trades --------------------------------------------------

    def add_trades(self, trades: Sequence[DisclosedTrade]) -> int:
        """Insert only trades whose deterministic id is not already stored."""
        if not trades:
            return 0
        incoming = {trade.trade_id: trade for trade in trades}
        with self.sessions.begin() as session:
            existing = set(
                session.scalars(
                    select(TradeRow.trade_id).where(TradeRow.trade_id.in_(list(incoming)))
                ).all()
            )
            fresh = [trade for key, trade in incoming.items() if key not in existing]
            session.add_all(
                [
                    TradeRow(
                        trade_id=trade.trade_id,
                        filing_id=trade.filing_id,
                        politician_id=trade.politician_id,
                        chamber=trade.chamber.value,
                        symbol=trade.symbol,
                        asset_name=trade.asset_name,
                        asset_class=trade.asset_class.value,
                        transaction_type=trade.transaction_type.value,
                        owner=trade.owner.value,
                        transaction_date=trade.transaction_date,
                        disclosure_date=trade.disclosure_date,
                        amount_low=trade.amount_low,
                        amount_high=trade.amount_high,
                        comment=trade.comment,
                        source_url=trade.source_url,
                    )
                    for trade in fresh
                ]
            )
            return len(fresh)

    def list_trades(
        self,
        politician_id: str | None = None,
        symbol: str | None = None,
        chamber: str | None = None,
        transaction_type: str | None = None,
        actionable_only: bool = False,
        disclosed_since: date | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(TradeRow)
            if politician_id is not None:
                statement = statement.where(TradeRow.politician_id == politician_id)
            if symbol is not None:
                statement = statement.where(TradeRow.symbol == symbol.upper())
            if chamber is not None:
                statement = statement.where(TradeRow.chamber == chamber)
            if transaction_type is not None:
                statement = statement.where(TradeRow.transaction_type == transaction_type)
            if actionable_only:
                statement = statement.where(
                    TradeRow.symbol.is_not(None),
                    TradeRow.asset_class.in_(("stock", "fund")),
                )
            if disclosed_since is not None:
                statement = statement.where(TradeRow.disclosure_date >= disclosed_since)
            rows = session.scalars(
                statement.order_by(
                    TradeRow.disclosure_date.desc(), TradeRow.transaction_date.desc()
                ).limit(limit)
            ).all()
        return [_trade_dict(row) for row in rows]

    def update_asset_names(self, updates: Sequence[tuple[str, str]]) -> int:
        """Rewrite stored asset names in place, by trade id."""
        if not updates:
            return 0
        with self.sessions() as session:
            for trade_id, name in updates:
                session.query(TradeRow).filter(TradeRow.trade_id == trade_id).update(
                    {"asset_name": name}
                )
            session.commit()
        return len(updates)

    def update_transaction_dates(self, updates: Sequence[tuple[str, date, str]]) -> int:
        """Rewrite a stored transaction date, keeping the original in the comment.

        Used only by `disclosures.repair_stored_trades`, which repairs dates
        that are impossible rather than merely surprising. The comment is
        overwritten deliberately: it exists to record exactly this kind of
        correction, and a row can only carry one.
        """
        if not updates:
            return 0
        with self.sessions() as session:
            for trade_id, transaction_date, comment in updates:
                session.query(TradeRow).filter(TradeRow.trade_id == trade_id).update(
                    {"transaction_date": transaction_date, "comment": comment}
                )
            session.commit()
        return len(updates)

    def update_symbols(self, updates: Sequence[tuple[str, str]]) -> int:
        """Rewrite stored symbols in place, by trade id."""
        if not updates:
            return 0
        with self.sessions() as session:
            for trade_id, symbol in updates:
                session.query(TradeRow).filter(TradeRow.trade_id == trade_id).update(
                    {"symbol": symbol.upper()}
                )
            session.commit()
        return len(updates)

    def clear_symbols(self, trade_ids: Sequence[str]) -> int:
        """Drop a symbol that was never a ticker, keeping the row.

        The asset name is what the filing said and stays; only the claim that
        the row is tradeable goes away, because a symbol like `-- AMCR` can
        never be priced and counted as a permanent coverage failure.
        """
        if not trade_ids:
            return 0
        with self.sessions() as session:
            for trade_id in trade_ids:
                session.query(TradeRow).filter(TradeRow.trade_id == trade_id).update(
                    {"symbol": None, "asset_class": "other"}
                )
            session.commit()
        return len(trade_ids)

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(TradeRow, trade_id)
            return _trade_dict(row) if row is not None else None

    def distinct_trade_symbols(self, actionable_only: bool = True) -> list[str]:
        with self.sessions() as session:
            statement = select(TradeRow.symbol).where(TradeRow.symbol.is_not(None)).distinct()
            if actionable_only:
                statement = statement.where(TradeRow.asset_class.in_(("stock", "fund")))
            return sorted({symbol for symbol in session.scalars(statement).all() if symbol})

    def trade_counts(self) -> dict[str, int]:
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(TradeRow)) or 0
            actionable = (
                session.scalar(
                    select(func.count())
                    .select_from(TradeRow)
                    .where(
                        TradeRow.symbol.is_not(None),
                        TradeRow.asset_class.in_(("stock", "fund")),
                    )
                )
                or 0
            )
            members = (
                session.scalar(select(func.count(func.distinct(TradeRow.politician_id)))) or 0
            )
        return {"total": total, "actionable": actionable, "members": members}

    def latest_disclosure_date(self) -> date | None:
        with self.sessions() as session:
            return session.scalar(select(func.max(TradeRow.disclosure_date)))

    # -- member scores -----------------------------------------------------

    def replace_member_scores(self, scores: Sequence[MemberScore]) -> int:
        with self.sessions.begin() as session:
            session.execute(delete(MemberScoreRow))
            session.add_all(
                [
                    MemberScoreRow(
                        politician_id=score.politician_id,
                        horizon_days=score.horizon_days,
                        side=score.side.value,
                        trades_scored=score.trades_scored,
                        win_rate=score.win_rate,
                        raw_excess_return=score.raw_excess_return,
                        median_excess_return=score.median_excess_return,
                        score=score.score,
                        total_disclosed_dollars=score.total_disclosed_dollars,
                        average_reporting_lag_days=score.average_reporting_lag_days,
                        best_symbol=score.best_symbol,
                        best_symbol_excess=score.best_symbol_excess,
                        best_symbol_side=(
                            score.best_symbol_side.value if score.best_symbol_side else None
                        ),
                        trades_unpriced=score.trades_unpriced,
                        model_probability=score.model_probability,
                        rankable=score.rankable,
                        computed_at=score.computed_at,
                    )
                    for score in scores
                ]
            )
        return len(scores)

    def list_member_scores(
        self,
        horizon_days: int | None = None,
        rankable_only: bool = False,
        side: str = TradeSide.ALL.value,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Scores for one side of the book, defaulting to the whole record.

        `side` defaults to `all` so an existing caller keeps the behaviour it
        had, but it is never optional in meaning: asking for a ranking without
        saying which direction it is for is how a member with no purchases was
        recommended off the back of a buys-only backtest.
        """
        with self.sessions() as session:
            statement = select(MemberScoreRow).where(MemberScoreRow.side == side)
            if horizon_days is not None:
                statement = statement.where(MemberScoreRow.horizon_days == horizon_days)
            if rankable_only:
                statement = statement.where(MemberScoreRow.rankable.is_(True))
            rows = session.scalars(
                statement.order_by(MemberScoreRow.score.desc()).limit(limit)
            ).all()
        return [_score_dict(row) for row in rows]

    def get_member_score(
        self, politician_id: str, horizon_days: int, side: str = TradeSide.ALL.value
    ) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(MemberScoreRow, (politician_id, horizon_days, side))
            return _score_dict(row) if row is not None else None

    def member_score_sides(self, politician_id: str, horizon_days: int) -> dict[str, Any]:
        """Every side of one member's record at one horizon, keyed by side.

        The three routinely disagree, and a page that shows one without the
        others invites the reader to assume it covers both.
        """
        with self.sessions() as session:
            rows = session.scalars(
                select(MemberScoreRow).where(
                    MemberScoreRow.politician_id == politician_id,
                    MemberScoreRow.horizon_days == horizon_days,
                )
            ).all()
        return {row.side: _score_dict(row) for row in rows}

    # -- watchlists --------------------------------------------------------

    def create_watchlist(self, watchlist: Watchlist) -> None:
        with self.sessions.begin() as session:
            session.merge(
                WatchlistRow(
                    watchlist_id=watchlist.watchlist_id,
                    name=watchlist.name,
                    politician_ids_json=json.dumps(watchlist.politician_ids),
                    allocation_mode=watchlist.allocation_mode,
                    allocation_value=watchlist.allocation_value,
                    mirror_sells=watchlist.mirror_sells,
                    min_amount_low=watchlist.min_amount_low,
                    starting_equity=watchlist.starting_equity,
                    active=watchlist.active,
                    created_at=watchlist.created_at,
                )
            )

    def list_watchlists(self, active_only: bool = False) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(WatchlistRow)
            if active_only:
                statement = statement.where(WatchlistRow.active.is_(True))
            rows = session.scalars(statement.order_by(WatchlistRow.created_at.desc())).all()
        return [_watchlist_dict(row) for row in rows]

    def get_watchlist(self, watchlist_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(WatchlistRow, watchlist_id)
            return _watchlist_dict(row) if row is not None else None

    def set_watchlist_active(self, watchlist_id: str, active: bool) -> None:
        with self.sessions.begin() as session:
            row = session.get(WatchlistRow, watchlist_id)
            if row is not None:
                row.active = active

    def delete_watchlist(self, watchlist_id: str) -> None:
        with self.sessions.begin() as session:
            session.execute(delete(MirrorFillRow).where(MirrorFillRow.watchlist_id == watchlist_id))
            session.execute(delete(WatchlistRow).where(WatchlistRow.watchlist_id == watchlist_id))

    # -- mirror fills ------------------------------------------------------

    def add_mirror_fills(self, fills: Sequence[MirrorFill]) -> int:
        if not fills:
            return 0
        incoming = {fill.fill_id: fill for fill in fills}
        with self.sessions.begin() as session:
            existing = set(
                session.scalars(
                    select(MirrorFillRow.fill_id).where(MirrorFillRow.fill_id.in_(list(incoming)))
                ).all()
            )
            fresh = [fill for key, fill in incoming.items() if key not in existing]
            session.add_all(
                [
                    MirrorFillRow(
                        fill_id=fill.fill_id,
                        watchlist_id=fill.watchlist_id,
                        trade_id=fill.trade_id,
                        politician_id=fill.politician_id,
                        symbol=fill.symbol,
                        side=fill.side.value,
                        shares=fill.shares,
                        price=fill.price,
                        notional=fill.notional,
                        priced_on=fill.priced_on,
                        price_basis=fill.price_basis,
                        note=fill.note,
                    )
                    for fill in fresh
                ]
            )
            return len(fresh)

    def list_mirror_fills(self, watchlist_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(MirrorFillRow)
                .where(MirrorFillRow.watchlist_id == watchlist_id)
                .order_by(MirrorFillRow.priced_on.asc(), MirrorFillRow.fill_id.asc())
                .limit(limit)
            ).all()
        return [_fill_dict(row) for row in rows]

    def mirrored_trade_ids(self, watchlist_id: str) -> set[str]:
        with self.sessions() as session:
            return set(
                session.scalars(
                    select(MirrorFillRow.trade_id).where(
                        MirrorFillRow.watchlist_id == watchlist_id
                    )
                ).all()
            )

    # -- political mentions ------------------------------------------------

    def add_mentions(self, mentions: Sequence[PoliticalMention]) -> int:
        if not mentions:
            return 0
        incoming = {mention.mention_id: mention for mention in mentions}
        with self.sessions.begin() as session:
            existing = set(
                session.scalars(
                    select(MentionRow.mention_id).where(MentionRow.mention_id.in_(list(incoming)))
                ).all()
            )
            fresh = [mention for key, mention in incoming.items() if key not in existing]
            session.add_all(
                [
                    MentionRow(
                        mention_id=mention.mention_id,
                        figure=mention.figure,
                        figure_slug=mention.figure_slug,
                        symbol=mention.symbol,
                        company_name=mention.company_name,
                        stance=mention.stance.value,
                        headline=mention.headline,
                        url=mention.url,
                        source=mention.source,
                        published_at=mention.published_at,
                        confidence=mention.confidence,
                        matched_terms_json=json.dumps(mention.matched_terms),
                    )
                    for mention in fresh
                ]
            )
            return len(fresh)

    # -- news --------------------------------------------------------------

    def add_news(self, items: Sequence[dict[str, Any]]) -> int:
        if not items:
            return 0
        incoming = {item["item_id"]: item for item in items}
        with self.sessions.begin() as session:
            existing = set(
                session.scalars(
                    select(NewsRow.item_id).where(NewsRow.item_id.in_(list(incoming)))
                ).all()
            )
            fresh = [item for key, item in incoming.items() if key not in existing]
            session.add_all(
                [
                    NewsRow(
                        item_id=item["item_id"],
                        category=item["category"],
                        subject=item.get("subject", ""),
                        title=item["title"],
                        url=item["url"],
                        source=item.get("source", ""),
                        summary=item.get("summary"),
                        published_at=item["published_at"],
                        sentiment_score=item.get("sentiment_score", 0.0),
                        sentiment_label=item.get("sentiment_label", "neutral"),
                        sentiment_terms_json=json.dumps(item.get("sentiment_terms", {})),
                    )
                    for item in fresh
                ]
            )
            return len(fresh)

    def list_news(
        self,
        category: str | None = None,
        subject: str | None = None,
        since: datetime | None = None,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(NewsRow)
            if category is not None:
                statement = statement.where(NewsRow.category == category)
            if subject is not None:
                statement = statement.where(NewsRow.subject == subject)
            if since is not None:
                statement = statement.where(NewsRow.published_at >= since)
            rows = session.scalars(
                statement.order_by(NewsRow.published_at.desc()).limit(limit)
            ).all()
        return [
            {
                "item_id": row.item_id,
                "category": row.category,
                "subject": row.subject,
                "title": row.title,
                "url": row.url,
                "source": row.source,
                "summary": row.summary,
                "published_at": as_utc(row.published_at),
                "sentiment_score": row.sentiment_score,
                "sentiment_label": row.sentiment_label,
                "sentiment_terms": json.loads(row.sentiment_terms_json),
            }
            for row in rows
        ]

    def news_sentiment_by_subject(
        self, category: str, since: datetime, limit: int = 500
    ) -> dict[str, dict[str, Any]]:
        """Mean sentiment and headline count per subject, for ranking inputs."""
        with self.sessions() as session:
            rows = session.execute(
                select(
                    NewsRow.subject,
                    func.count().label("items"),
                    func.avg(NewsRow.sentiment_score).label("mean_score"),
                )
                .where(NewsRow.category == category, NewsRow.published_at >= since)
                .group_by(NewsRow.subject)
                .limit(limit)
            ).all()
        return {
            row.subject: {"items": int(row.items), "mean_score": float(row.mean_score or 0.0)}
            for row in rows
        }

    def news_counts(self) -> dict[str, int]:
        with self.sessions() as session:
            rows = session.execute(
                select(NewsRow.category, func.count()).group_by(NewsRow.category)
            ).all()
        return {row[0]: int(row[1]) for row in rows}

    def list_mentions(
        self,
        symbol: str | None = None,
        figure_slug: str | None = None,
        since: datetime | None = None,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            statement = select(MentionRow)
            if symbol is not None:
                statement = statement.where(MentionRow.symbol == symbol.upper())
            if figure_slug is not None:
                statement = statement.where(MentionRow.figure_slug == figure_slug)
            if since is not None:
                statement = statement.where(MentionRow.published_at >= since)
            rows = session.scalars(
                statement.order_by(MentionRow.published_at.desc()).limit(limit)
            ).all()
        return [_mention_dict(row) for row in rows]
