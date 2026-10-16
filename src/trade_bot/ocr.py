"""Reading the filings that were submitted on paper.

A few hundred House filings are scans: a member filled in the paper PTR form
by hand or on a typewriter, someone fed it through a scanner, and the Clerk
published the resulting image. There is no text layer, so the parser in
`disclosures` correctly refuses them and they sit in the archive marked
unreadable. The members who file this way are not marginal -- McCaul, Khanna,
Rogers, Blumenthal and Malliotakis are all in the list -- so leaving them out
biases the whole dataset toward whoever happens to file electronically.

Conventional OCR does not solve this. The PTR form is a dense table with
rotated column headers and checkbox columns, and tesseract returns unusable
noise on it (the asset rows come back as strings like ``rx{ [| Pt ft txt``).
A vision model reads the same page as a table and returns structured rows,
which is the one job here that no free path does at all.

Three rules keep the cost negligible and bounded:

* Only filings already marked unreadable are ever sent. A filing with a text
  layer is parsed locally and never touches this module.
* Every result is cached on disk by document id, so a filing is read once and
  never billed again, including across re-syncs and CI runs.
* Each run has a hard ceiling on how many new filings it will read, so a
  surprise backlog cannot turn into a surprise bill.

Without an API key the module is a no-op and the filings stay marked
unreadable, exactly as they are today.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

from trade_bot.config import Settings
from trade_bot.secrets import resolve_api_key

logger = logging.getLogger(__name__)

_pace_lock = Lock()
_last_request = 0.0


def _pace() -> None:
    """Hold every caller to one request per `_MIN_REQUEST_INTERVAL`."""
    global _last_request
    with _pace_lock:
        wait = _last_request + _MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()

# The form is a landscape table; below this the scan is a cover sheet or a
# continuation page with nothing on it worth a request.
_MIN_IMAGE_BYTES = 12_000
# Vision models degrade on very large images and bill by tile. The scans are
# around 2200px on the long edge, which is already close to ideal.
_MAX_EDGE_PX = 2200
_MAX_PAGES_PER_FILING = 8
# Pages of one filing read at once. The account limit is a token budget per
# minute, not a request count, so the constraint is how fast pages are pushed
# rather than how many are in flight; six saturated it instantly and spent the
# rest of the minute being refused.
_PAGE_WORKERS = 2
# The SDK retries 429s with exponential backoff and honours Retry-After. Its
# default of two attempts is not enough when the budget is already spent: a
# whole run of filings failed because each gave up while the limit was still
# saturated from the previous one.
_MAX_RETRIES = 8
# Minimum gap between requests, across every thread. At detail="high" a page
# of this form tiles into roughly 3,000 tokens, and the account limit is
# 200,000 tokens a minute, so the ceiling is about sixty-five pages a minute.
# This is set below that rather than at it: pacing exactly to the limit leaves
# no room for a retry or a larger-than-usual page, and once the budget is
# saturated it stays saturated -- backoff expires while the window is still
# full, which is how an entire run came back empty.
_MIN_REQUEST_INTERVAL = 1.3
# How many times to wait out a saturated window before giving up on a page.
_RATE_LIMIT_ATTEMPTS = 4
_RATE_LIMIT_COOLDOWN = 20.0
# Hard ceiling on billable requests for the lifetime of the process. The
# per-run filing cap does not bound spend on its own: a filing is many pages,
# and a page can be retried. This is the number that actually decides the
# bill, so it is enforced here rather than inferred from the filing count.
_MAX_REQUESTS_PER_PROCESS = 400
_requests_made = 0


class BudgetExhausted(RuntimeError):
    """Raised when the process has sent as many requests as it is allowed."""

# Things that appear in a ticker-shaped position on the form but are not
# tickers. Three separate traps, each of which collapses many unrelated
# companies onto one symbol if believed:
#   * instrument and share-class suffixes trailing the asset name
#     ("BROWN & BROWN, INC. CMN" is common stock, not ticker CMN);
#   * the owner column at the far left of every row (JT/SP/DC, for joint,
#     spouse and dependent child);
#   * a model writing out its own null rather than emitting JSON null.
_NOT_TICKERS = frozenset(
    {
        "CMN", "COM", "CS", "COMMON", "STOCK", "SHS", "SH", "CL", "CLA", "CLB",
        "ADR", "ADS", "ETF", "FUND", "LP", "LLC", "INC", "CORP", "CO", "LTD",
        "NA", "NV", "SA", "PLC", "TR", "REIT", "BOND", "NOTE", "N", "A", "B",
        "JT", "SP", "DC",
        "NULL", "NONE", "NIL", "UNKNOWN", "TICKER",
        # The CMN suffix again, as the scanner variously renders it.
        "CMT", "CNA", "CAN", "CMH", "OMN",
        # Denomination on a hybrid or perpetual line, not a security.
        "USD", "EUR", "GBP", "CAD", "CHF", "JPY", "AUD",
    }
)

# The blank PTR form carries a worked example above the first real row, and a
# scan of the form includes it. Read as data it invents transactions under a
# real member's name -- fifty of them in one backfill, all in a stock that
# does not exist.
_EXAMPLE_ROW = re.compile(r"\bexample\b|\bmega\s*corp\b|\bsample\b", re.IGNORECASE)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["transactions"],
    "properties": {
        "transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "asset_name",
                    "ticker",
                    "transaction_date",
                    "transaction_type",
                    "amount_range",
                    "confident",
                ],
                "properties": {
                    "asset_name": {"type": "string"},
                    "ticker": {
                        "type": ["string", "null"],
                        "description": "Ticker in parentheses after the asset name, else null.",
                    },
                    "transaction_date": {
                        "type": ["string", "null"],
                        "description": "ISO YYYY-MM-DD, else null if unreadable.",
                    },
                    "transaction_type": {
                        "type": "string",
                        "enum": ["purchase", "sale", "exchange", "unknown"],
                    },
                    "amount_range": {
                        "type": ["string", "null"],
                        "description": "The bracket exactly as printed, e.g. '$1,001 - $15,000'.",
                    },
                    "confident": {
                        "type": "boolean",
                        "description": "False if any field was guessed from an unclear scan.",
                    },
                },
            },
        }
    },
}

_PROMPT = """You are reading a scanned US House of Representatives Periodic \
Transaction Report (PTR). These are images of a paper form, often handwritten \
or typewritten and sometimes skewed.

Extract every transaction row from the table. For each row:

- asset_name: the full asset description as written.
- ticker: the symbol in parentheses after the asset name, uppercase, or null \
if the row shows none. Do not invent or infer a ticker from the company name.
- transaction_date: the transaction/trade date as YYYY-MM-DD. The form is \
US-style MM/DD/YYYY. Use the transaction date column, NOT the notification or \
filing date column.
- transaction_type: the marked type. P or "purchase" -> purchase; S, "sale", \
"S (partial)" -> sale; E -> exchange. If no type is legible use "unknown".
- amount_range: the checked amount bracket, copied exactly as printed.
- confident: true only if you read every field of the row clearly. Set false \
if you guessed at any character.

Rules:
- Report only rows that actually appear. Never fill the table out to a round \
number of rows, and never carry a value down from the row above.
- If a page has no transaction table at all, return an empty list.
- Accuracy matters far more than completeness. A row marked confident=false is \
useful; a plausible invented row is harmful."""


@dataclass
class OcrRow:
    """One transaction read off a scanned page."""

    asset_name: str
    ticker: str | None
    transaction_date: date | None
    transaction_type: str
    amount_range: str | None
    confident: bool


@dataclass
class OcrResult:
    doc_id: str
    rows: list[OcrRow]
    pages_read: int
    cached: bool = False
    error: str | None = None


def is_available(settings: Settings) -> bool:
    """Whether scanned filings can be read at all in this environment."""
    if resolve_api_key(settings.openai_keyring_name, "OPENAI_API_KEY") is None:
        return False
    try:
        import openai  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _cache_path(settings: Settings, doc_id: str) -> Path:
    directory = settings.project_root / "var" / "ocr"
    directory.mkdir(parents=True, exist_ok=True)
    # Document ids come from the Clerk's index and are numeric, but a filing
    # id is not something to interpolate into a path unchecked.
    safe = "".join(character for character in doc_id if character.isalnum() or character in "-_")
    return directory / f"{safe or 'unknown'}.json"


def page_images(pdf_bytes: bytes) -> list[bytes]:
    """The scanned page images embedded in a PDF, as PNG bytes.

    A scanned filing is a thin PDF wrapper around one image per page, so the
    images can be lifted out directly. That avoids a rasteriser dependency
    (poppler, ghostscript) that would otherwise have to exist on every machine
    and in CI.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: list[bytes] = []
    for page in reader.pages[:_MAX_PAGES_PER_FILING]:
        try:
            embedded = list(page.images)
        except Exception:  # a malformed or unsupported image stream
            continue
        for image in embedded:
            data = _normalise(image.data)
            if data is not None and len(image.data) >= _MIN_IMAGE_BYTES:
                out.append(data)
    return out


def _normalise(data: bytes) -> bytes | None:
    """Re-encode to an upright PNG the API will accept.

    The rotation step is not cosmetic. The PTR form is always landscape, but
    some pages are scanned into a portrait frame with the whole table turned
    on its side, and no `/Rotate` entry records it. Measured against a known
    filing, a vision model reads 88% of the rows off an upright page and only
    20% off the same form sideways, so turning the page is the single largest
    accuracy win available here -- and it is free.
    """
    from PIL import Image

    try:
        opened = Image.open(io.BytesIO(data))
        image: Image.Image = opened
        # Scans are 1-bit bilevel, which several encoders and the API itself
        # handle poorly; greyscale is universally safe and no larger.
        if image.mode not in ("RGB", "L"):
            image = image.convert("L")
        if image.height > image.width:
            # PIL rotates counter-clockwise, which puts the form's header back
            # at the top left where every upright page in these filings has it.
            image = image.rotate(90, expand=True)
        longest = max(image.size)
        if longest > _MAX_EDGE_PX:
            scale = _MAX_EDGE_PX / longest
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except Exception:
        return None


def _parse_row(raw: dict[str, Any]) -> OcrRow | None:
    name = (raw.get("asset_name") or "").strip()
    if not name or _EXAMPLE_ROW.search(name):
        return None
    ticker = (raw.get("ticker") or "").strip().upper() or None
    # A vision model asked for a ticker will occasionally return the asset
    # name again, or a placeholder. Anything that is not shaped like a symbol
    # is dropped rather than trusted.
    if ticker and not (1 <= len(ticker) <= 6 and ticker.isalpha()):
        ticker = None
    # Filers abbreviate the instrument type at the end of the asset name --
    # "BROWN & BROWN, INC. CMN" is common stock, not ticker CMN. Read as a
    # symbol these collapse a dozen unrelated companies onto one ticker, which
    # is worse than having no ticker at all.
    if ticker in _NOT_TICKERS:
        ticker = None
    parsed_date: date | None = None
    if raw.get("transaction_date"):
        try:
            parsed_date = date.fromisoformat(str(raw["transaction_date"])[:10])
        except ValueError:
            parsed_date = None
    return OcrRow(
        asset_name=name[:300],
        ticker=ticker,
        transaction_date=parsed_date,
        transaction_type=str(raw.get("transaction_type") or "unknown").lower(),
        amount_range=(raw.get("amount_range") or None),
        confident=bool(raw.get("confident")),
    )


def read_scanned_filing(
    settings: Settings,
    doc_id: str,
    pdf_bytes: bytes,
    member_hint: str | None = None,
) -> OcrResult:
    """Read one scanned filing, using the disk cache when it has been read before."""
    cache = _cache_path(settings, doc_id)
    if cache.exists():
        try:
            payload = json.loads(cache.read_text())
            rows = [row for row in (_parse_row(item) for item in payload["rows"]) if row]
            return OcrResult(doc_id, rows, payload.get("pages_read", 0), cached=True)
        except Exception:
            cache.unlink(missing_ok=True)  # rewrite a corrupt entry rather than fail

    if not is_available(settings):
        return OcrResult(doc_id, [], 0, error="no OpenAI key or vision dependencies")

    images = page_images(pdf_bytes)
    if not images:
        return OcrResult(doc_id, [], 0, error="no page images could be extracted")

    # One request per page. A single call carrying every page of a six-page
    # filing has to emit well over a hundred rows in one response, and it
    # quietly stops short -- a measured 70 rows against roughly 130 on the
    # page. Per-page requests cost the same in tokens and do not truncate.
    #
    # The pages are independent, so they go out together: read serially, a
    # six-page filing takes about a minute, which turns a few hundred filings
    # into an overnight job for no reason.
    def read_page(numbered: tuple[int, bytes]) -> list[dict[str, Any]] | None:
        number, image = numbered
        try:
            return _ask_vision(settings, image, member_hint, number, len(images))
        except Exception as exc:
            logger.warning("page %s of %s failed: %s", number, doc_id, exc)
            return None

    # Pages that came back are kept even when a later one fails. Discarding
    # the filing threw away reads that had already been paid for, and the
    # retry then paid for them again -- the single largest source of waste in
    # the first backfill, which cost roughly three times its estimate.
    pages: list[list[dict[str, Any]]] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=min(len(images), _PAGE_WORKERS)) as pool:
        for outcome in pool.map(read_page, enumerate(images, start=1)):
            if outcome is None:
                failures += 1
            else:
                pages.append(outcome)
    if failures:
        logger.warning("%s: %s of %s pages unread", doc_id, failures, len(images))
    if not pages:
        return OcrResult(doc_id, [], len(images), error="no page could be read")
    raw_rows: list[dict[str, Any]] = [row for page in pages for row in page]

    rows = [row for row in (_parse_row(item) for item in raw_rows) if row]
    cache.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "pages_read": len(images),
                "model": settings.openai_model,
                "rows": raw_rows,
            },
            indent=1,
        )
    )
    return OcrResult(doc_id, rows, len(images))


def _ask_vision(
    settings: Settings,
    image: bytes,
    member_hint: str | None,
    page_number: int,
    page_count: int,
) -> list[dict[str, Any]]:
    """Read the transaction table off a single page."""
    global _requests_made
    if _requests_made >= _MAX_REQUESTS_PER_PROCESS:
        raise BudgetExhausted(
            f"reached the {_MAX_REQUESTS_PER_PROCESS}-request ceiling for this run"
        )
    from openai import OpenAI

    client = OpenAI(
        api_key=resolve_api_key(settings.openai_keyring_name, "OPENAI_API_KEY"),
        max_retries=_MAX_RETRIES,
    )
    prompt = _PROMPT
    if member_hint:
        prompt += f"\n\nThis filing was submitted by {member_hint}."
    prompt += (
        f"\n\nThis is page {page_number} of {page_count}. Report every row on "
        "this page and nothing from any other page."
    )

    encoded = base64.b64encode(image).decode("ascii")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
        },
    ]

    # The SDK's own backoff gives up in seconds, but a spent token budget only
    # refills on a minute boundary, so a page that arrives during a saturated
    # window has to outlast the window rather than retry inside it. Abandoning
    # it instead discards every other page of the filing too, which is how a
    # 202-filing run produced 48 usable reads.
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        if _requests_made >= _MAX_REQUESTS_PER_PROCESS:
            raise BudgetExhausted(
                f"reached the {_MAX_REQUESTS_PER_PROCESS}-request ceiling for this run"
            )
        _pace()
        _requests_made += 1
        try:
            return _complete(client, settings, content)
        except Exception as exc:
            if "rate_limit" not in str(exc) or attempt == _RATE_LIMIT_ATTEMPTS - 1:
                raise
            logger.info("token budget saturated, waiting out the window")
            time.sleep(_RATE_LIMIT_COOLDOWN)
    raise RuntimeError("unreachable")


def _complete(
    client: Any, settings: Settings, content: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # The SDK's message type is a deep union of TypedDicts that a dynamically
    # built multi-image payload does not narrow to cleanly.
    messages: Any = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ptr_transactions", "schema": _SCHEMA, "strict": True},
        },
        temperature=0,
        max_tokens=4000,
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    rows = payload.get("transactions") or []
    return [row for row in rows if isinstance(row, dict)]


def to_transactions(
    rows: list[OcrRow],
    symbol_index: dict[str, str],
    filed_at: date,
) -> list[Any]:
    """Convert read rows into the same `ParsedTransaction` the parser emits.

    Once a scanned row is in this shape it flows through `build_trades` and
    every downstream stage identically to an electronically filed one, so the
    scanned path shares the date clamping, deduplication and scoring rather
    than reimplementing any of it.
    """
    from trade_bot.disclosures import ParsedTransaction, parse_amount
    from trade_bot.domain import AssetClass, Owner, TransactionType
    from trade_bot.symbols import resolve

    types = {
        "purchase": TransactionType.PURCHASE,
        "sale": TransactionType.SALE,
        "exchange": TransactionType.EXCHANGE,
    }
    out: list[Any] = []
    for row in rows:
        transaction_type = types.get(row.transaction_type)
        if transaction_type is None or row.transaction_date is None:
            # Without a direction or a date the row cannot be scored, and a
            # default for either would be an invention.
            continue
        symbol = row.ticker or resolve(row.asset_name, symbol_index)
        low, high = parse_amount(row.amount_range or "")
        out.append(
            ParsedTransaction(
                asset_name=row.asset_name,
                symbol=symbol,
                # The paper form has no asset-class column. Anything that
                # resolved to a ticker is a listed security; the rest are the
                # private partnerships, municipal bonds and funds that make up
                # most of these filings, and "other" keeps them out of the
                # board without pretending to classify them.
                asset_class=AssetClass.STOCK if symbol else AssetClass.OTHER,
                transaction_type=transaction_type,
                owner=Owner.UNKNOWN,
                transaction_date=row.transaction_date,
                notification_date=filed_at,
                amount_low=low,
                amount_high=high,
            )
        )
    return out


def backfill_scanned_filings(
    settings: Settings,
    store: Any,
    reporter: Any = None,
    limit: int = 40,
) -> dict[str, Any]:
    """Read filings already stored as unreadable scans.

    A normal sync only looks at filings it has not seen, so the scans that
    accumulated before this path existed would never be revisited. This walks
    them directly off the disk cache, newest first, and stops at `limit` new
    reads so the spend on any one run is known in advance.
    """
    from trade_bot.activity import NullReporter
    from trade_bot.disclosures import build_trades
    from trade_bot.domain import Filing
    from trade_bot.listings import combined_index

    reporter = reporter or NullReporter()
    if not is_available(settings):
        reporter.note("scanned filings skipped: no OpenAI key or vision extras installed")
        return {"read": 0, "trades": 0, "skipped": 0}

    pending = [
        filing
        for filing in store.list_filings(parsed=False, limit=5000)
        if "scanned paper filing" in (filing.get("parse_error") or "")
    ]
    pending.sort(key=lambda filing: filing["filed_at"], reverse=True)
    if not pending:
        return {"read": 0, "trades": 0, "skipped": 0}

    symbol_index = combined_index(settings, store)
    root = settings.project_root / "var" / "filings" / "house"
    read = trades_added = skipped = 0

    for position, record in enumerate(pending, start=1):
        if read >= limit:
            reporter.note(f"stopping at the {limit}-filing budget; {len(pending) - read} left")
            break
        path = root / str(record["year"]) / f"{record['doc_id']}.pdf"
        if not path.exists():
            skipped += 1
            continue

        reporter.step(
            f"reading scan {record['doc_id']} ({position}/{len(pending)})",
            position,
            len(pending),
        )
        result = read_scanned_filing(settings, record["doc_id"], path.read_bytes())
        if not result.cached:
            read += 1
        if not result.rows:
            skipped += 1
            continue

        filing = Filing(**{key: record[key] for key in Filing.model_fields if key in record})
        transactions = to_transactions(result.rows, symbol_index, filing.filed_at)
        trades = build_trades(filing, transactions)
        trades_added += store.add_trades(trades)
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

    reporter.note(f"scanned filings: {read} read, {trades_added} trades recovered")
    return {"read": read, "trades": trades_added, "skipped": skipped, "pending": len(pending)}
