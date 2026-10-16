"""The published snapshot must not pretend to be the live app."""

from __future__ import annotations

from datetime import date

import pytest

from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, Politician, TransactionType
from trade_bot.export import export_site


@pytest.fixture
def seeded(store):
    store.upsert_politicians(
        [
            Politician(
                politician_id="house-tx-doe-jane",
                full_name="Jane Doe",
                first_name="Jane",
                last_name="Doe",
                chamber=Chamber.HOUSE,
                state="TX",
                district="21",
                party="Republican",
            )
        ]
    )
    store.add_trades(
        [
            DisclosedTrade(
                trade_id="t1",
                filing_id="house:2026:1",
                politician_id="house-tx-doe-jane",
                chamber=Chamber.HOUSE,
                symbol="ACME",
                asset_name="ACME Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(2026, 1, 5),
                disclosure_date=date(2026, 2, 5),
                amount_low=1001,
                amount_high=15000,
            ),
            # A real symbol with a dot in it. Static hosts read the dot as a
            # file extension, so this is the shape that silently 404s.
            DisclosedTrade(
                trade_id="t2",
                filing_id="house:2026:1",
                politician_id="house-tx-doe-jane",
                chamber=Chamber.HOUSE,
                symbol="BRK.B",
                asset_name="Berkshire Hathaway Inc Class B",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=date(2026, 1, 6),
                disclosure_date=date(2026, 2, 5),
                amount_low=1001,
                amount_high=15000,
            ),
        ]
    )
    return store


def test_export_writes_every_page(settings, seeded, prices, tmp_path, monkeypatch):
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    result = export_site(settings, seeded, out)

    assert result["failed"] == {}
    for page in ("index", "members", "trades", "watchlists", "politics", "news", "data"):
        assert (out / f"{page}.html").exists(), page
    assert (out / "members" / "house-tx-doe-jane" / "index.html").exists()
    assert (out / "evidence.html").exists()


def test_dotted_tickers_resolve_as_directory_indexes(
    settings, seeded, prices, tmp_path, monkeypatch
):
    """A ticker containing a dot must still be reachable at its own URL.

    Written as `tickers/BRK.B.html`, a static host serving `/tickers/BRK.B`
    sees `.B` as the extension, never tries the `.html` suffix, and returns a
    404 -- so the link on the board looks live but goes nowhere.
    """
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)

    for symbol in ("ACME", "BRK.B"):
        assert (out / "tickers" / symbol / "index.html").exists(), symbol
        assert not (out / "tickers" / f"{symbol}.html").exists(), symbol
    assert (out / "static" / "style.css").exists()


def test_snapshot_contains_no_forms_that_need_a_server(
    settings, seeded, prices, tmp_path, monkeypatch
):
    """A POST form in a static snapshot is a button that lies.

    Refresh, watchlist creation and mirroring all need the running app; the
    published page has to say so rather than render dead controls.
    """
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)

    for page in out.rglob("*.html"):
        assert 'method="post"' not in page.read_text(), f"{page.name} has a dead form"


def test_snapshot_drops_the_live_status_bar(settings, seeded, prices, tmp_path, monkeypatch):
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)

    index = (out / "index.html").read_text()
    assert "/api/activity/stream" not in index
    assert "EventSource" not in index
    assert "Published" in index  # the snapshot banner replaces it


def test_snapshot_ships_noindex(settings, seeded, prices, tmp_path, monkeypatch):
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)

    assert "Disallow: /" in (out / "robots.txt").read_text()
    assert "X-Robots-Tag" in (out / "_headers").read_text()


def test_export_mode_does_not_leak_into_the_live_app(
    settings, seeded, prices, tmp_path, monkeypatch
):
    """`static_export` is global state; it must be reset even on failure."""
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    export_site(settings, seeded, tmp_path / "dist")
    assert web.templates.env.globals["static_export"] is False


def test_stylesheet_is_always_published(settings, seeded, prices, tmp_path, monkeypatch):
    """The bug that shipped an unstyled site.

    Static assets used to be copied after the page loop, so a render failure
    left a dist with HTML and no CSS — which then deployed happily.
    """
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)
    stylesheet = out / "static" / "style.css"
    assert stylesheet.exists()
    assert stylesheet.stat().st_size > 1000
    assert 'href="/static/style.css"' in (out / "index.html").read_text()


def test_a_failed_render_publishes_nothing(settings, seeded, prices, tmp_path, monkeypatch):
    """Half a site must never replace a whole one."""
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    out.mkdir()
    (out / "index.html").write_text("<html>previous good build</html>")

    def _boom():
        raise RuntimeError("database went away")

    monkeypatch.setattr(web, "create_app", _boom)
    with pytest.raises(RuntimeError):
        export_site(settings, seeded, out)

    # The previous build survives, and no partial directory is left behind.
    assert (out / "index.html").read_text() == "<html>previous good build</html>"
    assert not out.with_name("dist.partial").exists()


def test_market_badge_is_resolved_not_left_loading(
    settings, seeded, prices, tmp_path, monkeypatch
):
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    out = tmp_path / "dist"
    export_site(settings, seeded, out)
    index = (out / "index.html").read_text()
    assert "market …" not in index
    assert "market open" in index or "market closed" in index
