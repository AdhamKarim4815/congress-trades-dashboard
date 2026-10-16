"""Ingest of Senate periodic transaction reports.

`efdsearch.senate.gov` sits behind an edge WAF that fingerprints the TLS
handshake, not just the headers. httpx and curl are refused with a blanket
403 from every network -- residential included -- while Chrome on the same
machine and the same IP loads the site fine. No amount of header forgery
changes that, because the tell is in the handshake. `curl_cffi` replays
Chrome's TLS and HTTP/2 fingerprint, which is what makes this module
possible at all.

The reward for getting in is that Senate filings are far better structured
than the House's: an HTML table with one row per transaction, already
carrying a ticker column, rather than a PDF that has to be reconstructed
from extracted text.

Reading the site requires posting an acknowledgement of the Ethics in
Government Act's restrictions on the *use* of these reports -- not for a
commercial purpose, not for credit rating, not for soliciting money. A
personal dashboard is squarely within what the Act permits; anything else is
the operator's responsibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.disclosures import ParsedTransaction, build_trades, parse_amount, politician_id
from trade_bot.domain import (
    AssetClass,
    Chamber,
    Filing,
    Owner,
    Politician,
    SyncReport,
    TransactionType,
)

BASE = "https://efdsearch.senate.gov"
HOME_URL = f"{BASE}/search/home/"
SEARCH_URL = f"{BASE}/search/report/data/"
# Report type 11 is the periodic transaction report.
PTR_REPORT_TYPE = "[11]"

CSRF_RE = re.compile(r"name=['\"]csrfmiddlewaretoken['\"]\s+value=['\"]([^'\"]+)")
LINK_RE = re.compile(r'href="(/search/view/([a-z]+)/([0-9a-f-]+)/?)"', re.IGNORECASE)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")

OWNERS = {
    "self": Owner.SELF,
    "spouse": Owner.SPOUSE,
    "joint": Owner.JOINT,
    "child": Owner.CHILD,
    "dependent child": Owner.CHILD,
}
TRANSACTION_TYPES = {
    "purchase": TransactionType.PURCHASE,
    "sale": TransactionType.SALE,
    "sale (full)": TransactionType.SALE,
    "sale (partial)": TransactionType.PARTIAL_SALE,
    "exchange": TransactionType.EXCHANGE,
}
ASSET_CLASSES = {
    "stock": AssetClass.STOCK,
    "stock option": AssetClass.OPTION,
    "corporate bond": AssetClass.BOND,
    "municipal security": AssetClass.BOND,
    "non-public stock": AssetClass.OTHER,
    "other securities": AssetClass.OTHER,
    "cryptocurrency": AssetClass.CRYPTO,
    "exchange traded fund": AssetClass.FUND,
    "etf": AssetClass.FUND,
    "mutual fund": AssetClass.FUND,
}


class SenateBlockedError(RuntimeError):
    """The portal refused the client outright, rather than returning no data.

    Kept distinct from an empty result so the dashboard never reports "the
    Senate disclosed nothing" when the truth is "we could not look".
    """


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", html)).replace("&#35;", "#").strip()


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class SenateFiling:
    uuid: str
    kind: str  # "ptr" for electronic, "paper" for a scanned upload
    first_name: str
    last_name: str
    office: str
    filed_at: date

    @property
    def url(self) -> str:
        return f"{BASE}/search/view/{self.kind}/{self.uuid}/"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class SenateEfdClient:
    """A Chrome-shaped session against the Senate disclosure portal."""

    def __init__(self, timeout: float = 45.0, impersonate: str = "chrome") -> None:
        self.timeout = timeout
        self.impersonate = impersonate
        self._session: Any = None
        self._token: str = ""

    def open(self) -> None:
        try:
            from curl_cffi import requests as curl_requests
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise SenateBlockedError(
                "curl_cffi is required to reach the Senate portal; run `uv sync`"
            ) from exc

        session: Any = curl_requests.Session(impersonate=self.impersonate)
        response = session.get(HOME_URL, timeout=self.timeout)
        if response.status_code == 403:
            raise SenateBlockedError(
                "the Senate portal refused this client (HTTP 403) even with TLS "
                "impersonation; it may be blocking this network"
            )
        match = CSRF_RE.search(response.text)
        if not match:
            raise SenateBlockedError("no CSRF token on the Senate search page")
        self._token = match.group(1)
        # Acknowledge the Ethics in Government Act notice. This is a condition
        # of viewing the reports at all; see the module docstring.
        session.post(
            HOME_URL,
            data={"prohibition_agreement": "1", "csrfmiddlewaretoken": self._token},
            headers={"Referer": HOME_URL},
            timeout=self.timeout,
        )
        self._session = session

    def search(self, since: date, page_size: int = 100, max_pages: int = 40) -> list[SenateFiling]:
        """Every periodic transaction report filed on or after `since`."""
        if self._session is None:
            self.open()
        assert self._session is not None

        filings: list[SenateFiling] = []
        offset = 0
        for _ in range(max_pages):
            response = self._session.post(
                SEARCH_URL,
                data={
                    "start": str(offset),
                    "length": str(page_size),
                    "report_types": PTR_REPORT_TYPE,
                    "filer_types": "[]",
                    "submitted_start_date": since.strftime("%m/%d/%Y 00:00:00"),
                    "submitted_end_date": "",
                    "candidate_state": "",
                    "senator_state": "",
                    "office_id": "",
                    "first_name": "",
                    "last_name": "",
                    "csrfmiddlewaretoken": self._token,
                },
                headers={"Referer": HOME_URL, "X-CSRFToken": self._token},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise SenateBlockedError(
                    f"Senate search returned HTTP {response.status_code}"
                )
            payload = response.json()
            rows = payload.get("data") or []
            if not rows:
                break
            for row in rows:
                filing = _filing_from_row(row)
                if filing is not None:
                    filings.append(filing)
            offset += len(rows)
            if offset >= int(payload.get("recordsTotal") or 0):
                break
        return filings

    def transactions(self, filing: SenateFiling) -> list[ParsedTransaction]:
        if self._session is None:
            self.open()
        assert self._session is not None
        response = self._session.get(filing.url, timeout=self.timeout)
        if response.status_code != 200:
            raise SenateBlockedError(f"filing fetch returned HTTP {response.status_code}")
        return parse_senate_filing(response.text, filing.filed_at)


def _filing_from_row(row: list[str]) -> SenateFiling | None:
    """Search rows are `[first, last, office, link_html, filed_date]`."""
    if len(row) < 5:
        return None
    link = LINK_RE.search(row[3] or "")
    filed_at = _parse_date(_text(row[4] or ""))
    if link is None or filed_at is None:
        return None
    return SenateFiling(
        uuid=link.group(3),
        kind=link.group(2).lower(),
        first_name=_text(row[0] or ""),
        last_name=_text(row[1] or ""),
        office=_text(row[2] or ""),
        filed_at=filed_at,
    )


def parse_senate_filing(html: str, filed_at: date) -> list[ParsedTransaction]:
    """Read the transaction table out of an electronic Senate PTR.

    Columns are `# | Transaction Date | Owner | Ticker | Asset Name |
    Asset Type | Type | Amount | Comment`. A paper filing has no table at
    all, which is why callers check the filing kind first.
    """
    transactions: list[ParsedTransaction] = []
    for raw_row in ROW_RE.findall(html):
        cells = [_text(cell) for cell in CELL_RE.findall(raw_row)]
        if len(cells) < 8:
            continue
        transaction_date = _parse_date(cells[1])
        if transaction_date is None:
            continue

        ticker = cells[3].strip().upper()
        if ticker in {"--", "N/A", ""}:
            ticker = ""
        asset_name = cells[4] or "(unnamed asset)"
        asset_type = cells[5].strip().lower()
        transaction_type = TRANSACTION_TYPES.get(
            cells[6].strip().lower(), TransactionType.UNKNOWN
        )
        amount_low, amount_high = parse_amount(cells[7])
        asset_class = ASSET_CLASSES.get(asset_type, AssetClass.OTHER)
        # An exchange-traded ticker on an otherwise unclassified line is the
        # same signal the House parser leans on.
        if asset_class is AssetClass.OTHER and ticker:
            asset_class = AssetClass.FUND

        transactions.append(
            ParsedTransaction(
                asset_name=asset_name[:300],
                symbol=ticker or None,
                asset_class=asset_class,
                transaction_type=transaction_type,
                owner=OWNERS.get(cells[2].strip().lower(), Owner.UNKNOWN),
                transaction_date=transaction_date,
                notification_date=filed_at,
                amount_low=amount_low,
                amount_high=amount_high,
            )
        )
    return transactions


def senate_politician(filing: SenateFiling) -> Politician:
    """A member record keyed on name.

    Unlike the House index, the search results carry no state, so the id
    omits it; `sync_legislators` fills state and party in afterwards.
    """
    return Politician(
        politician_id=politician_id(Chamber.SENATE, "", filing.last_name, filing.first_name),
        full_name=filing.full_name,
        first_name=filing.first_name,
        last_name=filing.last_name,
        chamber=Chamber.SENATE,
    )


def sync_senate(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    since: date | None = None,
    limit: int | None = None,
) -> SyncReport:
    """Download and parse every new Senate periodic transaction report."""
    reporter = reporter or NullReporter()
    started = datetime.now(UTC)
    years = settings.disclosures.resolved_senate_years
    since = since or date(min(years), 1, 1)
    cap = limit or settings.disclosures.max_filings_per_sync

    client = SenateEfdClient(timeout=settings.disclosures.request_timeout_seconds)
    try:
        reporter.step("opening a Senate eFD session")
        client.open()
        reporter.step(f"searching Senate filings since {since}")
        filings = client.search(since)
    except SenateBlockedError as exc:
        reporter.note(f"Senate portal unavailable: {exc}", level="warn")
        return SyncReport(
            source="senate-efd",
            ok=False,
            error=(
                f"{exc}. House coverage is unaffected; the Senate portal "
                "fingerprints clients and sometimes refuses even a faithful one."
            ),
            started_at=started,
            finished_at=datetime.now(UTC),
        )
    except Exception as exc:
        return SyncReport(
            source="senate-efd",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    known = store.known_filing_ids(Chamber.SENATE.value)
    pending = [
        filing
        for filing in filings
        if f"senate:{filing.filed_at.year}:{filing.uuid}" not in known
    ]
    pending.sort(key=lambda filing: filing.filed_at, reverse=True)
    pending = pending[:cap]

    politicians_added = store.upsert_politicians(
        senate_politician(filing) for filing in filings
    )

    parsed = failed = trades_added = 0
    total = len(pending)
    for index, filing in enumerate(pending, start=1):
        reporter.step(
            f"parsing Senate filing {index}/{total}: {filing.full_name} ({filing.filed_at})",
            index,
            total,
        )
        record = Filing(
            filing_id=f"senate:{filing.filed_at.year}:{filing.uuid}",
            politician_id=politician_id(
                Chamber.SENATE, "", filing.last_name, filing.first_name
            ),
            chamber=Chamber.SENATE,
            doc_id=filing.uuid,
            year=filing.filed_at.year,
            filing_type="P",
            filed_at=filing.filed_at,
            source_url=filing.url,
        )
        if filing.kind != "ptr":
            failed += 1
            store.upsert_filing(
                record.model_copy(
                    update={
                        "parse_error": "paper filing uploaded as a scan; its "
                        "transactions cannot be read automatically"
                    }
                )
            )
            continue
        try:
            transactions = client.transactions(filing)
        except Exception as exc:
            failed += 1
            store.upsert_filing(
                record.model_copy(update={"parse_error": f"{type(exc).__name__}: {exc}"})
            )
            continue

        trades = build_trades(record, transactions)
        trades_added += store.add_trades(trades)
        parsed += 1
        store.upsert_filing(
            record.model_copy(update={"parsed": True, "transaction_count": len(trades)})
        )

    reporter.note(f"Senate: parsed {parsed} filings, added {trades_added} trades")
    return SyncReport(
        source="senate-efd",
        ok=True,
        filings_seen=len(filings),
        filings_parsed=parsed,
        filings_failed=failed,
        trades_added=trades_added,
        politicians_added=politicians_added,
        details={"since": since.isoformat(), "queued": total},
        started_at=started,
        finished_at=datetime.now(UTC),
    )
