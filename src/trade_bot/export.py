"""Render the dashboard to a static site.

A FastAPI app cannot run on Cloudflare Pages, but almost every page here is
a read-only view over data that a scheduled job already refreshed. So the
same routes are rendered once into flat HTML and published, giving an
always-available dashboard at no cost. The interactive parts (running a
refresh, creating a watchlist) need a server and are hidden in the export.

Pages are written extension-less-friendly: Cloudflare Pages serves `/members`
from `members.html` and `/members/x` from `members/x.html`, so the links in
the templates keep working untouched.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.market import market_status

PACKAGE_ROOT = Path(__file__).resolve().parent


def _write(destination: Path, html: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")


def export_site(
    settings: Settings,
    store: Store,
    out_dir: Path,
    reporter: JobReporter | None = None,
    max_members: int = 250,
    max_tickers: int = 250,
) -> dict[str, Any]:
    """Render every page to `out_dir`, returning what was written."""
    reporter = reporter or NullReporter()
    from fastapi.testclient import TestClient

    from trade_bot import web

    out_dir = Path(out_dir)
    # Build into a sibling directory and swap it in only once everything is
    # present. An earlier version wrote pages straight into `out_dir` and
    # copied the stylesheet afterwards; when rendering failed midway the
    # half-built directory was still published, and the site went live with
    # no CSS at all. A partial export must never be deployable.
    staging = out_dir.with_name(out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Static assets first: they do not depend on the database, so copying
    # them up front means the cheapest, most essential file is never the
    # casualty of a failure later on.
    static_source = PACKAGE_ROOT / "static"
    if static_source.exists():
        shutil.copytree(static_source, staging / "static", dirs_exist_ok=True)

    stamp = datetime.now(UTC)
    web.templates.env.globals["static_export"] = True
    web.templates.env.globals["exported_at"] = stamp
    # No JavaScript runs in a snapshot, so the market badge is resolved here
    # rather than left showing its loading placeholder forever.
    try:
        web.templates.env.globals["static_market"] = market_status()
    except Exception:
        web.templates.env.globals["static_market"] = None

    pages: list[tuple[str, Path]] = [
        ("/", staging / "index.html"),
        ("/digest", staging / "digest.html"),
        ("/evidence", staging / "evidence.html"),
        ("/members", staging / "members.html"),
        ("/trades", staging / "trades.html"),
        ("/watchlists", staging / "watchlists.html"),
        ("/politics", staging / "politics.html"),
        ("/news", staging / "news.html"),
        ("/data", staging / "data.html"),
    ]
    # Detail pages are written as `<name>/index.html` rather than `<name>.html`.
    # Tickers like BRK.B contain a dot, and a static host reading
    # `/tickers/BRK.B` treats `.B` as the file extension, so it never tries the
    # `.html` suffix and the link 404s. A directory index has no such ambiguity
    # and keeps the URL identical to the one the live server serves.
    members = store.list_politicians(with_trades_only=True)[:max_members]
    pages += [
        (
            f"/members/{member['politician_id']}",
            staging / "members" / member["politician_id"] / "index.html",
        )
        for member in members
    ]
    symbols = store.distinct_trade_symbols()[:max_tickers]
    pages += [
        (f"/tickers/{symbol}", staging / "tickers" / symbol / "index.html")
        for symbol in symbols
    ]

    written = 0
    failed: dict[str, str] = {}
    try:
        written, failed = _render(web, TestClient, settings, store, pages, reporter)
    except Exception:
        # Leave no half-built directory behind for a later run to trip over.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        web.templates.env.globals["static_export"] = False
        web.templates.env.globals["exported_at"] = None
        web.templates.env.globals["static_market"] = None

    # Cloudflare Pages serves this verbatim; it keeps the snapshot out of
    # search engines even if the hostname is ever guessed.
    (staging / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    (staging / "_headers").write_text(
        "/*\n"
        "  X-Robots-Tag: noindex, nofollow\n"
        "  X-Frame-Options: DENY\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: no-referrer\n",
        encoding="utf-8",
    )

    # Refuse to publish something that would render unstyled or empty.
    required = [staging / "index.html", staging / "static" / "style.css"]
    missing = [str(path.relative_to(staging)) for path in required if not path.exists()]
    if missing or written == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"refusing to publish an incomplete export: {written} pages rendered, "
            f"missing {missing or 'nothing'}"
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)

    if failed:
        reporter.note(f"{len(failed)} pages failed to render", level="warn")
    reporter.note(f"exported {written} pages to {out_dir}")
    return {
        "pages": written,
        "failed": failed,
        "members": len(members),
        "tickers": len(symbols),
        "out_dir": str(out_dir),
        "exported_at": stamp.isoformat(),
    }


def _render(
    web: Any,
    TestClient: Any,
    settings: Settings,
    store: Store,
    pages: list[tuple[str, Path]],
    reporter: JobReporter,
) -> tuple[int, dict[str, str]]:
    """Render every route, returning how many succeeded and which did not."""
    written = 0
    failed: dict[str, str] = {}
    app = web.create_app()
    with TestClient(app) as client:
        # The export renders against the same store the caller already
        # opened, so a Postgres-backed run publishes Postgres data.
        app.state.store = store
        app.state.settings = settings
        total = len(pages)
        for index, (route, destination) in enumerate(pages, start=1):
            if index % 25 == 0 or index == total:
                reporter.step(f"rendering {index}/{total}: {route}", index, total)
            response = client.get(route)
            if response.status_code != 200:
                failed[route] = f"HTTP {response.status_code}"
                continue
            _write(destination, response.text)
            written += 1
    return written, failed
