"""Ingest of congressional periodic transaction reports.

The community mirrors that used to serve this data as clean JSON (House and
Senate Stock Watcher) are gone -- their S3 buckets now refuse anonymous
reads -- so this module goes to the primary sources instead.

For the House that means: download `{year}FD.zip` from the Clerk, read the
XML index of every filing that year, keep the ones typed `P` (periodic
transaction report), fetch each one's PDF, and parse its transaction table
out of the extracted text. That last step is the fiddly part and
`parse_ptr_text` carries the details.

For the Senate, `efdsearch.senate.gov` sits behind a terms-of-access
interstitial and an edge WAF that refuses plenty of non-browser clients, so
`SenateEfdProvider` is written to try honestly and report a clear failure
rather than pretend the Senate simply has no trades.

Nothing here needs an API key or a paid plan.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx
from defusedxml import ElementTree as DefusedET

from trade_bot import ocr
from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import (
    MAX_PLAUSIBLE_LAG_DAYS,
    AssetClass,
    Chamber,
    DisclosedTrade,
    Filing,
    Owner,
    Politician,
    SyncReport,
    TransactionType,
    build_trade_id,
)

HOUSE_ARCHIVE_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
HOUSE_PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
LEGISLATORS_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.json"
SENATE_HOME_URL = "https://efdsearch.senate.gov/search/home/"
SENATE_SEARCH_URL = "https://efdsearch.senate.gov/search/report/data/"

# A PTR whose text layer is this short is a scanned paper filing: the Clerk
# accepts them, they contain no machine-readable text, and OCR is out of
# scope. They are recorded with an explicit parse error so the dashboard can
# show that coverage is incomplete rather than silently reporting no trades.
MIN_TEXT_LENGTH = 200

# Past this, a transaction date is not a late filing — it is a misread. No
# periodic transaction report describes a trade from a decade ago, so a date
# this old carries no information except that a digit came out wrong. It is
# the gate on corroborating a date from the rest of its filing; the far
# tighter `MAX_PLAUSIBLE_LAG_DAYS` is the separate question of whether a date
# can be trusted to price a member's entry.
ABSURD_LAG_DAYS = 3650


class MissingPdfDependencyError(RuntimeError):
    """`pypdf` is required to read filing PDFs."""


# -- PTR text parsing ------------------------------------------------------

# One transaction row: type, transaction date, notification date, amount.
# The dates run together in the extracted text ("03/16/202603/16/2026") and
# the type can butt directly against the asset-type code ("[ST]P 05/15/2026"),
# hence the lookbehind boundary and the tolerant separators.
TRANSACTION_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=\]))(?P<type>S \(partial\)|P|S|E)\s*"
    r"(?P<transaction_date>\d{2}/\d{2}/\d{4})\s*"
    r"(?P<notification_date>\d{2}/\d{2}/\d{4})\s*"
    r"(?P<amount>\$[\d,]+(?:\s*-\s*\$[\d,]+)?)",
    re.MULTILINE,
)

# The form renders its field labels in small caps, and the lowercase glyphs
# extract as NUL. Once those are stripped "Filing Status:" collapses to
# "F S:", "Subholding Of:" to "S O:", "Description:" to "D:", and so on.
METADATA_LABEL_RE = re.compile(r"\b(?:F S|S O|D|L|C|A)\s*:", re.IGNORECASE)
EXACT_NOISE = {
    "P T R",
    "F I",
    "T",
    "I P O",
    "C S",
    "I V D",
    "Type",
    "Date",
    "Yes No",
    "ID",
    "Owner",
    "Asset",
}
NOISE_FRAGMENTS = (
    "ID Owner Asset Transaction",
    "Date Notification",
    "Amount Cap.",
    "Gains >",
    "$200?",
    "Filing ID #",
    "For the complete list",
    "Clerk of the House",
    "Name:",
    "Status:",
    "State/District:",
    "I CERTIFY",
    "my knowledge",
    "Digitally Signed:",
    "Yes  No",
)
# Free text from the previous row's description, which must not be mistaken
# for the next row's asset name.
DESCRIPTIVE_RE = re.compile(
    r"@ \$|shares (?:sold|purchased|bought)|\bpurchase of\b", re.IGNORECASE
)
OWNER_RE = re.compile(r"^(SP|DC|JT)\b\s*")
# Prose, not a noun phrase: two or more lower-case words running into a full
# stop. A description wraps over several lines and only its *first* line
# carries the "D:" label, so the unlabelled continuation lines are invisible
# to METADATA_LABEL_RE and have to be recognised by their shape instead.
SENTENCE_RE = re.compile(r"[a-z]{3,}\s+[a-z]{3,}.*\.\s*$")
# The owner column (JT/SP/DC) sits immediately to the left of the asset name on
# every row that declares one, which makes it the most reliable left edge the
# form gives us when a previous row's description has bled downwards.
OWNER_INLINE_RE = re.compile(r"\b(SP|DC|JT)\b\s+(?=[A-Z0-9])")
# Long names are not in themselves suspicious -- "AT&T Inc. Depositary Shares,
# each representing a 1/1,000th interest in a share of 5.000% Perpetual
# Preferred Stock" is real, and contains sentence punctuation and lower-case
# prose. There is therefore no length rule here: truncating on one would
# corrupt genuine names to tidy up a handful of bad ones. Descriptions are
# stopped at the source instead, by SENTENCE_RE and the owner-code cut.
ASSET_CODE_RE = re.compile(r"\[([A-Z]{2})\]")
PAREN_TICKER_RE = re.compile(r"\(([A-Z][A-Z.\-]{0,5})\)")
# Parenthesised text that is not a symbol. Filers routinely write the state of
# incorporation, the instrument type or a disambiguating article in exactly the
# position a ticker goes, and reading those as symbols priced UnitedHealth as
# Deere and American Tower against a ticker that does not exist.
NOT_TICKERS = frozenset(
    {
        "THE", "DE", "REIT", "HC", "NEW", "CMN", "COM", "CS", "ADR", "ADS", "ETF",
        "LP", "LLC", "INC", "CORP", "CO", "LTD", "PLC", "NV", "SA", "AG", "USD",
        "CL", "CLA", "CLB", "A", "B", "I", "II", "III", "IV", "V",
    }
)
EXCHANGE_TICKER_RE = re.compile(
    r"(?:NYSE|NASDAQ|NYSEARCA|AMEX|BATS|OTC)[A-Z]*:\s*([A-Z][A-Z.\-]{0,5})"
)
AMOUNT_RE = re.compile(r"\$([\d,]+)")
# What a ticker can look like. Anything else in the symbol column came out of
# the parser rather than out of a filing, and is handed to the price fetcher
# every run to fail there. Kept in step with `validate.SYMBOL_RE`.
SYMBOL_SHAPE_RE = re.compile(r"[A-Z][A-Z0-9.\-]{0,5}")

OWNER_CODES = {"SP": Owner.SPOUSE, "DC": Owner.CHILD, "JT": Owner.JOINT}
TRANSACTION_CODES = {
    "P": TransactionType.PURCHASE,
    "S": TransactionType.SALE,
    "S (partial)": TransactionType.PARTIAL_SALE,
    "E": TransactionType.EXCHANGE,
}
# https://fd.house.gov/reference/asset-type-codes.aspx
STOCK_CODES = {"ST", "RS", "SA", "RE"}
FUND_CODES = {"EF", "ET", "MF"}
BOND_CODES = {"CS", "GS", "AB", "DB", "DO"}


@dataclass(frozen=True)
class ParsedTransaction:
    asset_name: str
    symbol: str | None
    asset_class: AssetClass
    transaction_type: TransactionType
    owner: Owner
    transaction_date: date
    notification_date: date
    amount_low: float
    amount_high: float


def _normalize(text: str) -> str:
    return text.replace("\x00", "").replace("\u2013", "-").replace("\u2014", "-")


def _is_noise(line: str) -> bool:
    if not line or line in EXACT_NOISE:
        return True
    if len(line) <= 3 and not line[0].isdigit():
        return True
    return any(fragment in line for fragment in NOISE_FRAGMENTS)


def _asset_block(segment: str) -> str:
    """Rebuild the asset description that precedes a transaction row.

    Asset names wrap across up to three lines and the type code (`[ST]`) can
    land on a line of its own, so the block is assembled backwards from the
    code. Walking backwards stops at the previous row's metadata labels or at
    text that is clearly still part of its free-form description, which is
    what keeps one row's notes out of the next row's asset name.
    """
    lines = [line.strip() for line in segment.splitlines()]
    lines = [line for line in lines if not _is_noise(line)]
    if not lines:
        return ""
    anchor = next(
        (index for index in range(len(lines) - 1, -1, -1) if ASSET_CODE_RE.search(lines[index])),
        len(lines) - 1,
    )
    block = [lines[anchor]]
    for index in range(anchor - 1, max(anchor - 4, -1), -1):
        candidate = lines[index]
        if (
            METADATA_LABEL_RE.search(candidate)
            or DESCRIPTIVE_RE.search(candidate)
            or SENTENCE_RE.search(candidate)
        ):
            break
        block.insert(0, candidate)
        if PAREN_TICKER_RE.search(" ".join(block)):
            break
    return _trim_to_owner(" ".join(block).strip())


def _trim_to_owner(block: str) -> str:
    """Cut anything to the left of the owner code, when prose precedes it.

    A description that wrapped without a label can still reach here. Where the
    row declares an owner, that code marks exactly where the asset name starts,
    so cutting to it recovers the name *and* leaves the code in place for
    `OWNER_RE` to read -- which is how the owner comes back too. Without this,
    a bled description pushes the code off position 0, `OWNER_RE` fails to
    match, and the trade is silently stored as `Owner.UNKNOWN`.
    """
    owners = list(OWNER_INLINE_RE.finditer(block))
    if not owners:
        return block
    head = block[: owners[-1].start()]
    # Only cut when what precedes the code is prose. A name that merely
    # contains the letters is left alone.
    if re.search(r"[a-z]{3,}", head):
        return block[owners[-1].start() :]
    return block


def classify_asset(code: str | None, symbol: str | None) -> AssetClass:
    if code in STOCK_CODES:
        return AssetClass.STOCK
    if code in FUND_CODES:
        return AssetClass.FUND
    if code in BOND_CODES:
        return AssetClass.BOND
    if code == "OP":
        return AssetClass.OPTION
    if code == "CT":
        return AssetClass.CRYPTO
    # "Other" is what filers reach for when an ETF has no better code. A real
    # exchange ticker alongside it is the best available evidence that the
    # line is something actually tradeable.
    if code == "OT" and symbol:
        return AssetClass.FUND
    return AssetClass.OTHER


def paren_ticker(name: str) -> str | None:
    """The ticker printed in parentheses in an asset name, if there is one.

    Read the *last* one, not the first. The convention is that the symbol
    closes the name -- ``UnitedHealth Group Incorporated Common Stock (DE)
    (UNH)`` -- and taking the first parenthesis stored `DE`, which is Deere,
    so every price and return on that row belonged to another company.
    Parentheses that hold something other than a symbol are skipped.
    """
    for match in reversed(list(PAREN_TICKER_RE.finditer(name))):
        candidate = match.group(1).upper()
        if candidate not in NOT_TICKERS:
            return candidate
    return None


def parse_amount(raw: str) -> tuple[float, float]:
    values = [float(value.replace(",", "")) for value in AMOUNT_RE.findall(raw)]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[1]


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%m/%d/%Y").date()


def parse_ptr_text(text: str) -> list[ParsedTransaction]:
    """Extract every transaction row from a House PTR's extracted text."""
    text = _normalize(text)
    transactions: list[ParsedTransaction] = []
    cursor = 0
    for match in TRANSACTION_RE.finditer(text):
        block = _asset_block(text[cursor : match.start()])
        cursor = match.end()

        owner = Owner.UNKNOWN
        owner_match = OWNER_RE.match(block)
        if owner_match:
            owner = OWNER_CODES[owner_match.group(1)]
            block = block[owner_match.end() :]

        code_match = ASSET_CODE_RE.search(block)
        code = code_match.group(1) if code_match else None
        name = (block[: code_match.start()] if code_match else block).strip(" -,")

        symbol = paren_ticker(name)
        if symbol is None:
            exchange = EXCHANGE_TICKER_RE.search(name)
            symbol = exchange.group(1) if exchange else None

        amount_low, amount_high = parse_amount(match.group("amount"))
        try:
            transaction_date = _parse_date(match.group("transaction_date"))
            notification_date = _parse_date(match.group("notification_date"))
        except ValueError:
            continue

        transactions.append(
            ParsedTransaction(
                asset_name=name[:300] or "(unnamed asset)",
                symbol=symbol,
                asset_class=classify_asset(code, symbol),
                transaction_type=TRANSACTION_CODES.get(
                    match.group("type"), TransactionType.UNKNOWN
                ),
                owner=owner,
                transaction_date=transaction_date,
                notification_date=notification_date,
                amount_low=amount_low,
                amount_high=amount_high,
            )
        )
    return transactions


def extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise MissingPdfDependencyError(
            "reading filing PDFs requires pypdf; run `uv sync`"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# -- identity --------------------------------------------------------------


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def politician_id(chamber: Chamber, state: str, last_name: str, first_name: str) -> str:
    """Stable id derived only from filing data.

    Deliberately not the bioguide id: the House filing index never carries
    one, so deriving the id from the roster would make a member's identity
    depend on whether a name match happened to succeed on a given run.
    """
    return "-".join(
        part for part in (chamber.value, state.lower(), _slug(last_name), _slug(first_name)) if part
    )


def split_state_district(state_dst: str) -> tuple[str, str | None]:
    value = (state_dst or "").strip().upper()
    if len(value) >= 4 and value[2:].isdigit():
        district = value[2:]
        return value[:2], (None if district == "00" else district)
    return value[:2], None


# -- House Clerk -----------------------------------------------------------


@dataclass(frozen=True)
class IndexEntry:
    last_name: str
    first_name: str
    suffix: str
    filing_type: str
    state_dst: str
    year: int
    filed_at: date
    doc_id: str

    @property
    def state(self) -> str:
        return split_state_district(self.state_dst)[0]

    @property
    def district(self) -> str | None:
        return split_state_district(self.state_dst)[1]

    @property
    def full_name(self) -> str:
        return " ".join(
            part for part in (self.first_name, self.last_name, self.suffix) if part
        ).strip()


class HouseClerkProvider:
    """Reads the House Clerk's public financial-disclosure archive."""

    name = "house-clerk"
    chamber = Chamber.HOUSE

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_root = settings.project_root / "var" / "filings" / "house"

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.settings.disclosures.request_timeout_seconds,
            headers={"User-Agent": self.settings.disclosures.user_agent},
            follow_redirects=True,
        )

    def fetch_index(self, year: int) -> list[IndexEntry]:
        with self._client() as client:
            response = client.get(HOUSE_ARCHIVE_URL.format(year=year))
            response.raise_for_status()
            payload = response.content
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not names:
                raise RuntimeError(f"{year}FD.zip contained no XML index")
            xml_bytes = archive.read(names[0])
        root = DefusedET.fromstring(xml_bytes.decode("utf-8-sig", errors="replace"))
        entries: list[IndexEntry] = []
        for member in root.findall("Member"):
            filed_raw = (member.findtext("FilingDate") or "").strip()
            doc_id = (member.findtext("DocID") or "").strip()
            if not doc_id or not filed_raw:
                continue
            try:
                filed_at = _parse_date(filed_raw)
            except ValueError:
                continue
            entries.append(
                IndexEntry(
                    last_name=(member.findtext("Last") or "").strip(),
                    first_name=(member.findtext("First") or "").strip(),
                    suffix=(member.findtext("Suffix") or "").strip(),
                    filing_type=(member.findtext("FilingType") or "").strip(),
                    state_dst=(member.findtext("StateDst") or "").strip(),
                    year=int(member.findtext("Year") or year),
                    filed_at=filed_at,
                    doc_id=doc_id,
                )
            )
        return entries

    def filing_url(self, year: int, doc_id: str) -> str:
        return HOUSE_PTR_URL.format(year=year, doc_id=doc_id)

    def fetch_filing_pdf(self, year: int, doc_id: str, client: httpx.Client) -> bytes:
        """Fetch a PTR PDF, caching it so a re-sync never re-downloads."""
        cached = self.cache_root / str(year) / f"{doc_id}.pdf"
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()
        response = client.get(self.filing_url(year, doc_id))
        response.raise_for_status()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(response.content)
        return response.content


def house_politician(entry: IndexEntry) -> Politician:
    state, district = split_state_district(entry.state_dst)
    return Politician(
        politician_id=politician_id(Chamber.HOUSE, state, entry.last_name, entry.first_name),
        full_name=entry.full_name,
        first_name=entry.first_name,
        last_name=entry.last_name,
        chamber=Chamber.HOUSE,
        state=state,
        district=district,
    )


def repair_transaction_date(transaction_date: date, filed_at: date) -> tuple[date, str | None]:
    """Fix the two ways a filing's transaction date comes out impossible.

    **After the filing.** A trade cannot be reported before it happens, so a
    date past the filing date is a typo. It is clamped, and the original kept
    in the comment.

    **A century early.** A scanned filing read by a vision model produced
    `1925-02-25` for `2025-02-25` — a 36,535-day reporting lag, in a system
    where the lag is the whole point. The repair is deliberately narrow: the
    last two digits of the year must already be right, so only the century is
    being corrected and no digit is invented. `1975` for `2025` fails that
    test here and falls through to `corroborate_transaction_dates`, which asks
    the rest of the filing rather than guessing.
    """
    if transaction_date > filed_at:
        return filed_at, (
            f"filing reports a transaction date of {transaction_date.isoformat()}, "
            f"after the filing date; treated as {filed_at.isoformat()}"
        )
    # Only look at dates too old to be a late filing of any kind.
    if (filed_at - transaction_date).days <= MAX_PLAUSIBLE_LAG_DAYS:
        return transaction_date, None
    for year in (filed_at.year, filed_at.year - 1):
        if year % 100 != transaction_date.year % 100:
            continue
        try:
            shifted = transaction_date.replace(year=year)
        except ValueError:  # 29 February in a non-leap year
            continue
        if shifted <= filed_at and (filed_at - shifted).days <= MAX_PLAUSIBLE_LAG_DAYS:
            return shifted, (
                f"filing reports a transaction date of {transaction_date.isoformat()}, "
                f"a century before the filing; read as {shifted.isoformat()}"
            )
    return transaction_date, None


def corroborate_transaction_dates(
    dates: Sequence[date], filed_at: date
) -> list[tuple[date, str | None]]:
    """Repair each date, then let the filing's own good rows vouch for the rest.

    A vision model reading a scanned form gets the same column right on most
    rows and wrong on a few, and the wrong ones are wrong in the year alone.
    One filing carried `1975-02-04` and `2025-02-04`, `1975-02-27` and
    `2025-02-27`, and so on down the page — the day and month were read
    correctly every time.

    So where a date is *absurd* after the per-row repair, the filing is asked
    whether it contains a believable row with the same day and month. If
    exactly one year answers, that is evidence rather than a guess: the
    scanner read those digits correctly elsewhere on the same page. Where no
    row answers, or two disagree, the date is left alone — `1975-02-14` in
    that same filing has no sibling and stays exactly as it was read.

    "Absurd" is a much colder line than "late". A member filing a 2023 trade
    in 2026 is three years late, which is a real and reportable thing; shifting
    that to 2026 would delete a genuine fact to tidy up a number. Only a date
    no periodic transaction report could ever be describing is corroborated.
    """
    repaired = [repair_transaction_date(value, filed_at) for value in dates]

    # Years vouched for by a row that came out plausible, keyed on day+month.
    vouched: dict[tuple[int, int], set[int]] = {}
    for value, _comment in repaired:
        if 0 <= (filed_at - value).days <= MAX_PLAUSIBLE_LAG_DAYS:
            vouched.setdefault((value.month, value.day), set()).add(value.year)

    out: list[tuple[date, str | None]] = []
    for original, (value, comment) in zip(dates, repaired, strict=True):
        if (filed_at - value).days <= ABSURD_LAG_DAYS:
            out.append((value, comment))
            continue
        years = vouched.get((value.month, value.day), set())
        if len(years) != 1:
            out.append((value, comment))
            continue
        try:
            shifted = value.replace(year=next(iter(years)))
        except ValueError:  # 29 February in a non-leap year
            out.append((value, comment))
            continue
        out.append(
            (
                shifted,
                f"filing reports a transaction date of {original.isoformat()}; "
                f"other rows of the same filing date that day {shifted.isoformat()}, "
                "so it is read as that",
            )
        )
    return out


def build_trades(
    filing: Filing,
    transactions: Iterable[ParsedTransaction],
) -> list[DisclosedTrade]:
    """Turn parsed rows into stored trades.

    `disclosure_date` is the filing date from the index, not the PTR's
    "notification date" column. The notification date is when the member's
    broker told *them*; the filing date is the first moment the public could
    have known, which is the only date a copier could have acted on.

    Transaction dates go through `corroborate_transaction_dates` first, which
    repairs the impossible ones and records what it changed in the comment, so
    no anomaly is corrected silently and none is corrected by guesswork.
    """
    rows = list(transactions)
    dates = corroborate_transaction_dates(
        [transaction.transaction_date for transaction in rows], filing.filed_at
    )
    trades: list[DisclosedTrade] = []
    for ordinal, (transaction, (transaction_date, comment)) in enumerate(
        zip(rows, dates, strict=True)
    ):
        disclosure_date = filing.filed_at
        trades.append(
            DisclosedTrade(
                trade_id=build_trade_id(
                    filing.filing_id,
                    transaction.symbol or transaction.asset_name,
                    transaction.transaction_type.value,
                    transaction.transaction_date,
                    transaction.owner.value,
                    transaction.amount_low,
                    ordinal,
                ),
                filing_id=filing.filing_id,
                politician_id=filing.politician_id,
                chamber=filing.chamber,
                symbol=transaction.symbol.upper() if transaction.symbol else None,
                asset_name=transaction.asset_name,
                asset_class=transaction.asset_class,
                transaction_type=transaction.transaction_type,
                owner=transaction.owner,
                transaction_date=transaction_date,
                disclosure_date=disclosure_date,
                amount_low=transaction.amount_low,
                amount_high=transaction.amount_high,
                comment=comment,
                source_url=filing.source_url,
            )
        )
    return trades


def sync_house(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    years: list[int] | None = None,
    limit: int | None = None,
    max_ocr_filings: int = 40,
) -> SyncReport:
    """Download and parse every new House PTR for the configured years.

    `max_ocr_filings` caps how many scanned filings a single run will send to
    a vision model, so a backlog cannot turn into an unexpected bill. Already
    cached reads do not count against it.
    """
    reporter = reporter or NullReporter()
    started = datetime.now(UTC)
    provider = HouseClerkProvider(settings)
    target_years = years or settings.disclosures.resolved_house_years
    cap = limit or settings.disclosures.max_filings_per_sync

    seen = parsed = failed = trades_added = 0
    politicians_added = 0
    # Scanned filings are read by a vision model, which costs money, so the
    # index of names the reader resolves tickers against is built once and
    # only when there is actually something to read.
    ocr_budget = max_ocr_filings if ocr.is_available(settings) else 0
    ocr_read = 0
    symbol_index: dict[str, str] = {}
    known = store.known_filing_ids(Chamber.HOUSE.value)
    pending: list[IndexEntry] = []

    try:
        for year in target_years:
            reporter.step(f"reading House filing index for {year}")
            entries = provider.fetch_index(year)
            ptrs = [entry for entry in entries if entry.filing_type == "P"]
            seen += len(ptrs)
            politicians_added += store.upsert_politicians(
                house_politician(entry) for entry in ptrs
            )
            for entry in ptrs:
                filing_id = f"house:{entry.year}:{entry.doc_id}"
                if filing_id not in known:
                    pending.append(entry)
            reporter.note(
                f"House {year}: {len(ptrs)} periodic transaction reports "
                f"({len([e for e in ptrs if f'house:{e.year}:{e.doc_id}' not in known])} new)"
            )
    except Exception as exc:
        return SyncReport(
            source="house-clerk",
            ok=False,
            error=f"could not read the House filing index: {exc}",
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    pending.sort(key=lambda entry: entry.filed_at, reverse=True)
    pending = pending[:cap]
    total = len(pending)

    # Downloads dominate the wall clock and are pure network wait, so they run
    # on a small pool while parsing stays on this thread. Parsing is CPU-bound
    # under the GIL and would gain nothing from more threads, and keeping all
    # database writes here means the store is only ever touched from one.
    workers = max(1, settings.disclosures.max_concurrent_downloads)
    with provider._client() as client:

        def _download(entry: IndexEntry) -> tuple[IndexEntry, bytes | None, str | None]:
            try:
                return entry, provider.fetch_filing_pdf(entry.year, entry.doc_id, client), None
            except Exception as exc:
                return entry, None, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=min(workers, max(total, 1))) as pool:
            downloads = pool.map(_download, pending)

            for index, (entry, payload, download_error) in enumerate(downloads, start=1):
                filing_id = f"house:{entry.year}:{entry.doc_id}"
                state, _ = split_state_district(entry.state_dst)
                member_id = politician_id(
                    Chamber.HOUSE, state, entry.last_name, entry.first_name
                )
                reporter.step(
                    f"parsing House filing {index}/{total}: "
                    f"{entry.full_name} ({entry.filed_at})",
                    index,
                    total,
                )
                filing = Filing(
                    filing_id=filing_id,
                    politician_id=member_id,
                    chamber=Chamber.HOUSE,
                    doc_id=entry.doc_id,
                    year=entry.year,
                    filing_type=entry.filing_type,
                    filed_at=entry.filed_at,
                    source_url=provider.filing_url(entry.year, entry.doc_id),
                )
                if payload is None:
                    failed += 1
                    store.upsert_filing(filing.model_copy(update={"parse_error": download_error}))
                    continue
                try:
                    text = extract_pdf_text(payload)
                except Exception as exc:
                    failed += 1
                    store.upsert_filing(
                        filing.model_copy(update={"parse_error": f"{type(exc).__name__}: {exc}"})
                    )
                    continue

                if len(text.strip()) < MIN_TEXT_LENGTH:
                    # A scan. The text parser cannot help, but a vision model
                    # can, and these are not marginal filers -- several of the
                    # most active members file exclusively on paper.
                    if ocr_read < ocr_budget:
                        if not symbol_index:
                            from trade_bot.listings import combined_index

                            symbol_index = combined_index(settings, store)
                        reporter.step(f"reading scanned filing {entry.doc_id}")
                        result = ocr.read_scanned_filing(
                            settings,
                            entry.doc_id,
                            payload,
                            member_hint=f"{entry.first_name} {entry.last_name}".strip(),
                        )
                        if not result.cached:
                            ocr_read += 1
                        if result.rows:
                            scanned = ocr.to_transactions(
                                result.rows, symbol_index, filing.filed_at
                            )
                            trades = build_trades(filing, scanned)
                            trades_added += store.add_trades(trades)
                            parsed += 1
                            store.upsert_filing(
                                filing.model_copy(
                                    update={
                                        "parsed": True,
                                        "transaction_count": len(trades),
                                        "parse_error": (
                                            "read from a scanned paper filing by OCR; "
                                            "verify against the original before trading"
                                        ),
                                    }
                                )
                            )
                            continue

                    failed += 1
                    store.upsert_filing(
                        filing.model_copy(
                            update={
                                "parse_error": "scanned paper filing with no text layer; "
                                "its transactions cannot be read automatically"
                            }
                        )
                    )
                    continue

                trades = build_trades(filing, parse_ptr_text(text))
                trades_added += store.add_trades(trades)
                parsed += 1
                store.upsert_filing(
                    filing.model_copy(update={"parsed": True, "transaction_count": len(trades)})
                )

    return SyncReport(
        source="house-clerk",
        ok=True,
        filings_seen=seen,
        filings_parsed=parsed,
        filings_failed=failed,
        trades_added=trades_added,
        politicians_added=politicians_added,
        details={"years": target_years, "queued": total},
        started_at=started,
        finished_at=datetime.now(UTC),
    )


# Senate ingest lives in `trade_bot.senate`: it needs a TLS-impersonating
# client that the House archive does not, and keeping that dependency out of
# this module keeps House syncing working even if it is unavailable.


# -- roster enrichment -----------------------------------------------------


def sync_legislators(
    settings: Settings, store: Store, reporter: JobReporter | None = None
) -> SyncReport:
    """Attach party, bioguide id and photo to members already seen filing.

    Purely additive: it never creates members, because the set of people this
    application cares about is exactly the set who have filed a transaction
    report, not the whole chamber.
    """
    reporter = reporter or NullReporter()
    started = datetime.now(UTC)
    reporter.step("fetching the congressional roster")
    try:
        with httpx.Client(
            timeout=settings.disclosures.request_timeout_seconds,
            headers={"User-Agent": settings.disclosures.user_agent},
            follow_redirects=True,
        ) as client:
            response = client.get(LEGISLATORS_URL)
            response.raise_for_status()
            roster = response.json()
    except Exception as exc:
        return SyncReport(
            source="congress-legislators",
            ok=False,
            error=f"could not fetch the roster: {exc}",
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    existing = {member["politician_id"]: member for member in store.list_politicians()}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    # The Senate portal does not publish a state alongside its filings, so
    # every senator missed the (state, surname) key and none of them ever got
    # a party, a state or a photo. A surname is enough on its own when only
    # one sitting member of that chamber has it.
    by_surname: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for person in roster:
        terms = person.get("terms") or []
        if not terms:
            continue
        term = terms[-1]
        last = (person.get("name") or {}).get("last", "")
        state = (term.get("state") or "").upper()
        chamber = "senate" if term.get("type") == "sen" else "house"
        entry = {"person": person, "term": term}
        by_key[(state, _slug(last))] = entry
        by_surname.setdefault((chamber, _slug(last)), []).append(entry)

    updates: list[Politician] = []
    for member in existing.values():
        key = ((member["state"] or "").upper(), _slug(member["last_name"]))
        match = by_key.get(key)
        if match is None:
            # Fall back to the surname within the chamber, and only when it is
            # unambiguous -- two sitting senators sharing one is not a match.
            candidates = by_surname.get((member["chamber"], _slug(member["last_name"])), [])
            if len(candidates) != 1:
                continue
            match = candidates[0]
        person, term = match["person"], match["term"]
        bioguide = (person.get("id") or {}).get("bioguide")
        updates.append(
            Politician(
                politician_id=member["politician_id"],
                full_name=member["full_name"],
                first_name=member["first_name"],
                last_name=member["last_name"],
                chamber=Chamber(member["chamber"]),
                # The roster knows the state even where the filing did not.
                state=member["state"] or (term.get("state") or "").upper(),
                district=member["district"],
                party=term.get("party"),
                bioguide_id=bioguide,
                photo_url=(
                    f"https://unitedstates.github.io/images/congress/225x275/{bioguide}.jpg"
                    if bioguide
                    else None
                ),
            )
        )

    store.upsert_politicians(updates)
    unmatched = len(existing) - len(updates)
    reporter.note(
        f"enriched {len(updates)} of {len(existing)} tracked members; {unmatched} unmatched"
    )
    return SyncReport(
        source="congress-legislators",
        ok=True,
        politicians_added=0,
        # `tracked` is the denominator that means something. The roster size is
        # not: this never creates members, so the several hundred sitting
        # members who have never filed a transaction report are not a gap to
        # close. Reporting "165 enriched of 537 on the roster" read as though
        # two thirds of Congress were still to do.
        details={
            "enriched": len(updates),
            "tracked": len(existing),
            "unmatched_members": unmatched,
            "roster_size": len(roster),
        },
        started_at=started,
        finished_at=datetime.now(UTC),
    )


def sync_all(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run every disclosure source, recording per-source health."""
    reporter = reporter or NullReporter()
    from trade_bot.committees import sync_committees
    from trade_bot.senate import sync_senate

    # Committees run last of the roster sources because they join on the
    # bioguide id that `sync_legislators` has just attached.
    reports = [
        sync_house(settings, store, reporter, limit=limit),
        sync_senate(settings, store, reporter, limit=limit),
        sync_legislators(settings, store, reporter),
        sync_committees(settings, store, reporter),
    ]
    payload = {
        report.source: report.model_dump(mode="json") for report in reports
    }
    store.set_state("sync.disclosures", {"sources": payload, "at": datetime.now(UTC).isoformat()})
    return {
        "sources": payload,
        "trades_added": sum(report.trades_added for report in reports),
        "filings_parsed": sum(report.filings_parsed for report in reports),
        "filings_failed": sum(report.filings_failed for report in reports),
    }


def repair_stored_trades(
    store: Store,
    apply: bool = False,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Apply the parser's two corrections to rows already in the database.

    The parser fix stops new bad rows being created; this reaches the ones
    already stored. Both repairs are deterministic -- neither invents a value
    the filing does not already contain:

    * **dates** that are impossible, repaired only where the filing itself
      says what they should be (`corroborate_transaction_dates`);
    * **symbols that are not tickers at all**, which reach the price fetcher
      every run and can only ever fail there. `-- AMCR` is a parser artefact,
      not a company. The symbol is cleared rather than guessed at: the asset
      name is untouched, so the row keeps everything the filing actually said
      and simply stops claiming to be tradeable.
    * **symbols** contradicted by a later ticker in the row's own asset name,
      and only where the stored one came from an earlier parenthesis in that
      same name. `UnitedHealth Group Incorporated Common Stock (DE) (UNH)`
      was stored as `DE`, so the row was priced as Deere.

    Anything that cannot be resolved by those rules is left alone and stays
    visible in the validation report. Defaults to reporting: this rewrites
    user-visible data, so the change is shown before it is made.
    """
    reporter = reporter or NullReporter()
    trades = store.list_trades(limit=500_000)

    date_changes: list[tuple[str, date, str]] = []
    symbol_changes: list[tuple[str, str, str]] = []

    # Grouped by filing, because corroboration is a filing-level question: the
    # rows that came out right are what vouch for the rows that did not.
    by_filing: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        by_filing.setdefault(trade["filing_id"], []).append(trade)
    for rows in by_filing.values():
        repaired = corroborate_transaction_dates(
            [row["transaction_date"] for row in rows], rows[0]["disclosure_date"]
        )
        for row, (value, comment) in zip(rows, repaired, strict=True):
            if value != row["transaction_date"] and comment:
                date_changes.append((row["trade_id"], value, comment))

    cleared: list[tuple[str, str]] = []
    for trade in trades:
        stored, name = trade.get("symbol"), trade.get("asset_name") or ""
        if not stored:
            continue
        if not SYMBOL_SHAPE_RE.fullmatch(str(stored)):
            cleared.append((trade["trade_id"], str(stored)))
            continue
        printed = paren_ticker(name)
        if printed is None or printed == str(stored).upper():
            continue
        # Only rewrite where the stored symbol is demonstrably an earlier
        # parenthesis of the same name. Otherwise the two disagree for a
        # reason this rule cannot see, and guessing is worse than reporting.
        earlier = [match.group(1).upper() for match in PAREN_TICKER_RE.finditer(name)]
        if str(stored).upper() in earlier[:-1]:
            symbol_changes.append((trade["trade_id"], str(stored), printed))

    if apply:
        store.update_transaction_dates(date_changes)
        store.update_symbols([(tid, new) for tid, _old, new in symbol_changes])
        store.clear_symbols([tid for tid, _old in cleared])

    reporter.note(
        f"stored trades: {len(date_changes)} dates, {len(symbol_changes)} symbols and "
        f"{len(cleared)} non-tickers repairable{' (applied)' if apply else ' (dry run)'}"
    )
    return {
        "dates_repaired": len(date_changes),
        "symbols_repaired": len(symbol_changes),
        "symbols_cleared": len(cleared),
        "cleared_samples": [{"trade_id": tid, "was": was} for tid, was in cleared[:5]],
        "applied": bool(apply),
        "date_samples": [
            {"trade_id": tid, "to": value.isoformat(), "why": why}
            for tid, value, why in date_changes[:5]
        ],
        "symbol_samples": [
            {"trade_id": tid, "from": old, "to": new} for tid, old, new in symbol_changes[:5]
        ],
    }


def repair_asset_names(
    settings: Settings,
    store: Store,
    apply: bool = False,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Re-derive asset names that captured a previous row's description.

    The parser fix stops new ones being created; this reaches the rows already
    stored. Filings are re-read from the PDFs already on disk and matched on
    the fields a name has no part in -- member, both dates, amount and
    direction -- so a row is only rewritten when the same trade is found again.

    Defaults to reporting. A name is user-visible data and this rewrites it in
    place, so the change is shown before it is made.
    """
    reporter = reporter or NullReporter()
    trades = store.list_trades(limit=300_000)
    suspect = [
        trade
        for trade in trades
        if trade.get("asset_name")
        and len(trade["asset_name"]) > 120
        and ". " in trade["asset_name"]
    ]
    root = settings.project_root / "var" / "filings" / "house"
    by_filing: dict[str, list[dict[str, Any]]] = {}
    for trade in suspect:
        by_filing.setdefault(trade["filing_id"], []).append(trade)

    changes: list[tuple[str, str, str]] = []
    unmatched = missing = 0
    for position, (filing_id, rows) in enumerate(by_filing.items(), start=1):
        reporter.step(f"re-reading {filing_id}", position, len(by_filing))
        parts = filing_id.split(":")
        pdf = root / parts[1] / f"{parts[-1]}.pdf"
        if not pdf.exists():
            missing += len(rows)
            continue
        try:
            parsed = parse_ptr_text(extract_pdf_text(pdf.read_bytes()))
        except Exception:
            missing += len(rows)
            continue
        for trade in rows:
            # Date, amount and direction are not enough to identify a row: one
            # filing routinely holds several trades sharing all three, and
            # taking the first put another company's name on this trade. The
            # symbol must agree, and the match must be the only one.
            candidates = [
                candidate
                for candidate in parsed
                if candidate.transaction_date == trade["transaction_date"]
                and candidate.amount_low == trade["amount_low"]
                and candidate.transaction_type.value == trade["transaction_type"]
                and candidate.asset_name != trade["asset_name"]
                and (
                    candidate.symbol is None
                    or trade.get("symbol") is None
                    or candidate.symbol.upper() == str(trade["symbol"]).upper()
                )
            ]
            # Ambiguity about *which* row only matters if the rows disagree.
            # A member often files the same holding twice in one report, and
            # when every candidate carries the same name there is nothing to
            # guess at. Otherwise leave it: a wrong name is worse than the
            # prose it would replace, because it looks right.
            if not candidates or len({c.asset_name for c in candidates}) != 1:
                unmatched += 1
                continue
            match = candidates[0]
            changes.append((trade["trade_id"], trade["asset_name"], match.asset_name))

    if apply and changes:
        store.update_asset_names([(tid, name) for tid, _old, name in changes])

    reporter.note(
        f"asset names: {len(changes)} recoverable of {len(suspect)} suspect"
        f"{' (applied)' if apply else ' (dry run)'}"
    )
    return {
        "suspect": len(suspect),
        "recoverable": len(changes),
        "unmatched": unmatched,
        "pdf_missing": missing,
        "applied": bool(apply and changes),
        "samples": [(old[:70], new[:70]) for _tid, old, new in changes[:5]],
    }
