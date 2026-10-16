"""Command line for the copy-trading dashboard.

Every command calls the exact modules the web dashboard calls; there is no
CLI-only code path. Progress is rendered from the same activity log the
dashboard's status bar streams, so a sync looks the same in both.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.table import Table

from trade_bot.activity import ACTIVITY, JobReporter, run_job
from trade_bot.config import Settings, get_settings
from trade_bot.database import Store
from trade_bot.disclosures import sync_all, sync_house, sync_legislators
from trade_bot.follow import build_portfolio, create_watchlist, mirror_watchlist
from trade_bot.news import sync_figure_news, sync_member_news, sync_world_news
from trade_bot.pipeline import STAGES, run_refresh
from trade_bot.politics import extract_mentions
from trade_bot.prices import PriceRepository
from trade_bot.ranking import (
    MissingMLDependencyError,
    model_path,
    rank_members,
    train_edge_model,
)
from trade_bot.recommend import build_recommendations
from trade_bot.secrets import delete_api_key, get_api_key, set_api_key
from trade_bot.senate import sync_senate

app = typer.Typer(help="Congressional copy-trading dashboard: who is buying what, and why.")
sync_app = typer.Typer(help="Pull disclosures, prices and news.")
members_app = typer.Typer(help="Inspect and rank members of Congress.")
watch_app = typer.Typer(help="Create and mirror watchlists.")
secrets_app = typer.Typer(help="Manage optional API keys in the system keychain.")
app.add_typer(sync_app, name="sync")
app.add_typer(members_app, name="members")
app.add_typer(watch_app, name="watch")
app.add_typer(secrets_app, name="secrets")

console = Console()


def _store() -> tuple[Settings, Store]:
    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    return settings, store


def _run_with_progress(
    name: str, channel: str, work: Callable[[JobReporter], Any]
) -> dict[str, Any]:
    """Run a job while streaming its activity messages to the terminal."""
    cursor = ACTIVITY.last_id()
    result: dict[str, Any] = {}

    with Live(console=console, refresh_per_second=8, transient=True) as live:
        import threading

        done = threading.Event()

        def _target() -> None:
            nonlocal result
            result = run_job(ACTIVITY, name, channel, work)
            done.set()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        while not done.wait(timeout=0.15):
            status = ACTIVITY.status()
            suffix = (
                f" [{status['current']}/{status['total']}]"
                if status.get("total")
                else ""
            )
            live.update(f"[cyan]{status['channel']}[/cyan] {status['message']}{suffix}")
        worker.join()

    for event in ACTIVITY.since(cursor):
        colour = {
            "error": "red",
            "warn": "yellow",
            "success": "green",
        }.get(event.level, "dim")
        if event.level != "info":
            console.print(f"[{colour}]{event.message}[/{colour}]")
    return result


@app.command()
def config() -> None:
    """Print the resolved runtime configuration."""
    settings = get_settings()
    console.print_json(json.dumps(settings.model_dump(mode="json", exclude={"session_secret"})))


@app.command()
def status() -> None:
    """Show what has been ingested so far."""
    settings, store = _store()
    counts = store.trade_counts()
    filings = store.filing_counts()
    table = Table("metric", "value")
    table.add_row("disclosed trades", str(counts["total"]))
    table.add_row("tradeable trades", str(counts["actionable"]))
    table.add_row("members with trades", str(counts["members"]))
    table.add_row("filings parsed", f"{filings['parsed']} / {filings['total']}")
    table.add_row("filings unreadable", str(filings["failed"]))
    table.add_row("price series cached", str(PriceRepository(settings).coverage()["symbols"]))
    table.add_row("headlines", str(sum(store.news_counts().values())))
    latest = store.latest_disclosure_date()
    table.add_row("latest disclosure", str(latest) if latest else "—")
    console.print(table)


@app.command()
def refresh(
    stages: Annotated[
        list[str] | None,
        typer.Option("--stage", help=f"Run only these stages. Choices: {', '.join(STAGES)}"),
    ] = None,
    filings: Annotated[
        int, typer.Option(help="Cap how many new filings to download this run (0 = configured)")
    ] = 0,
) -> None:
    """Run the whole pipeline: disclosures, prices, ranking, news, mentions, board."""
    settings, store = _store()
    selected = [stage for stage in (stages or []) if stage in STAGES] or None
    result = _run_with_progress(
        "refresh",
        "sync",
        lambda reporter: run_refresh(
            settings, store, reporter, stages=selected, filing_limit=filings or None
        ),
    )
    if not result.get("ok", True) and result.get("error"):
        console.print(f"[red]refresh failed:[/red] {result['error']}")
        raise typer.Exit(code=1)
    errors = result.get("errors") or {}
    if errors:
        console.print("[yellow]some stages failed:[/yellow]")
        for stage, error in errors.items():
            console.print(f"  [red]{stage}[/red]: {error}")
        # Exit non-zero: a stage that silently produced nothing leaves the
        # dashboard quietly wrong, which is worse than a red build.
        raise typer.Exit(code=1)
    console.print("[green]refresh complete[/green]")


@sync_app.command("house")
def sync_house_command(
    year: Annotated[list[int] | None, typer.Option("--year", help="Filing year(s)")] = None,
    limit: Annotated[int, typer.Option(help="Cap filings downloaded this run")] = 0,
) -> None:
    """Download and parse House periodic transaction reports."""
    settings, store = _store()
    report = _run_with_progress(
        "sync-house",
        "sync",
        lambda reporter: sync_house(
            settings, store, reporter, years=list(year) if year else None, limit=limit or None
        ).model_dump(mode="json"),
    )
    console.print_json(json.dumps(report, default=str))


@sync_app.command("senate")
def sync_senate_command(
    limit: Annotated[int, typer.Option(help="Cap filings downloaded this run")] = 0,
) -> None:
    """Download and parse Senate periodic transaction reports."""
    settings, store = _store()
    report = _run_with_progress(
        "sync-senate",
        "sync",
        lambda reporter: sync_senate(settings, store, reporter, limit=limit or None).model_dump(
            mode="json"
        ),
    )
    console.print_json(json.dumps(report, default=str))


@sync_app.command("roster")
def sync_roster_command() -> None:
    """Attach party, bioguide id and photos from the congressional roster."""
    settings, store = _store()
    report = sync_legislators(settings, store)
    console.print_json(json.dumps(report.model_dump(mode="json"), default=str))


@sync_app.command("all")
def sync_all_command(
    limit: Annotated[int, typer.Option(help="Cap filings downloaded this run")] = 0,
) -> None:
    """Run every disclosure source."""
    settings, store = _store()
    result = _run_with_progress(
        "sync-all", "sync", lambda reporter: sync_all(settings, store, reporter, limit or None)
    )
    console.print_json(json.dumps(result, default=str))


@sync_app.command("prices")
def sync_prices_command(
    force: Annotated[bool, typer.Option(help="Refetch even if the cache is fresh")] = False,
) -> None:
    """Fetch daily prices for every disclosed ticker plus the benchmark."""
    settings, store = _store()
    repository = PriceRepository(settings)
    symbols = [*store.distinct_trade_symbols(), settings.prices.benchmark_symbol]
    result = _run_with_progress(
        "sync-prices",
        "prices",
        lambda reporter: repository.ensure(symbols, reporter, force=force),
    )
    console.print(
        f"fetched {result.get('fetched', 0)}, skipped {result.get('skipped', 0)}, "
        f"failed {result.get('failed', 0)}"
    )


@sync_app.command("news")
def sync_news_command() -> None:
    """Pull world, business, figure and member news, scoring sentiment on the way in."""
    settings, store = _store()

    def _work(reporter: JobReporter) -> dict[str, Any]:
        return {
            "world": sync_world_news(settings, store, reporter),
            "figures": sync_figure_news(settings, store, reporter),
            "members": sync_member_news(settings, store, reporter),
            "mentions": extract_mentions(settings, store, reporter),
        }

    result = _run_with_progress("sync-news", "news", _work)
    console.print_json(json.dumps(result, default=str))


@members_app.command("rank")
def members_rank(
    model: Annotated[bool, typer.Option(help="Apply the trained edge model if present")] = True,
) -> None:
    """Score every member's disclosed trades and persist the ranking."""
    settings, store = _store()
    repository = PriceRepository(settings)
    result = _run_with_progress(
        "rank",
        "ranking",
        lambda reporter: rank_members(settings, store, repository, reporter, use_model=model),
    )
    console.print_json(json.dumps(result, default=str))


@members_app.command("top")
def members_top(
    limit: Annotated[int, typer.Option(help="How many members to show")] = 20,
) -> None:
    """Show the best-performing members."""
    settings, store = _store()
    horizon = settings.ranking.primary_horizon_days
    members = {member["politician_id"]: member for member in store.list_politicians()}
    table = Table(title=f"Top members by shrunk excess return over {horizon} days")
    for column in ("member", "party", "score", "raw", "trades", "win", "lag"):
        table.add_column(column, justify="right" if column not in ("member", "party") else "left")
    for score in store.list_member_scores(horizon_days=horizon, rankable_only=True)[:limit]:
        member = members.get(score["politician_id"], {})
        table.add_row(
            member.get("full_name", score["politician_id"]),
            (member.get("party") or "—")[:3],
            f"{score['score']:+.2%}",
            f"{score['raw_excess_return']:+.2%}",
            str(score["trades_scored"]),
            f"{score['win_rate']:.0%}",
            f"{score['average_reporting_lag_days']:.0f}d",
        )
    console.print(table)


@members_app.command("train")
def members_train(
    horizon: Annotated[int, typer.Option(help="Outcome horizon in days")] = 0,
) -> None:
    """Train the optional pooled trade-edge model (needs the `ml` extra)."""
    settings, store = _store()
    repository = PriceRepository(settings)
    try:
        model = _run_with_progress(
            "train",
            "ranking",
            lambda reporter: train_edge_model(
                settings, store, repository, reporter, horizon_days=horizon or None
            ),
        )
    except (MissingMLDependencyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not isinstance(model, dict) and hasattr(model, "save"):
        model.save(model_path(settings))
    console.print("[green]model trained[/green]")


@app.command("backfill-scans")
def backfill_scans(
    limit: int = typer.Option(40, help="How many new filings to read in this run."),
) -> None:
    """Read the scanned paper filings that have no text layer.

    Costs a fraction of a cent per filing and is cached permanently, so a
    filing is only ever paid for once.
    """
    from trade_bot.ocr import backfill_scanned_filings

    settings, store = _store()
    result = _run_with_progress(
        "backfill-scans",
        "backfill-scans",
        lambda reporter: backfill_scanned_filings(settings, store, reporter, limit=limit),
    )
    table = Table(title="Scanned paper filings")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, key in (
        ("filings read", "read"),
        ("trades recovered", "trades"),
        ("skipped", "skipped"),
        ("still unread", "pending"),
    ):
        if key in result:
            table.add_row(label, str(result[key]))
    console.print(table)


@app.command()
def validate() -> None:
    """Check that the stored data is plausible, not merely that it parsed.

    Runs as the last stage of every refresh too. Nothing here rewrites a row;
    it names what does not add up and links to the filing.
    """
    from trade_bot.validate import load_report, run_validation

    settings, store = _store()
    repository = PriceRepository(settings)
    _run_with_progress(
        "validate",
        "validate",
        lambda reporter: run_validation(settings, store, repository, reporter),
    )
    report = load_report(store) or {}
    findings = report.get("findings", [])
    if not findings:
        console.print("[green]nothing implausible found[/green]")
        return
    colours = {"error": "red", "warn": "yellow", "info": "dim"}
    for finding in findings:
        colour = colours.get(finding["severity"], "white")
        console.print(f"\n[{colour}]{finding['severity'].upper():5}[/{colour}] {finding['title']}")
        console.print(f"      [dim]{finding['detail']}[/dim]")
        for sample in finding["samples"][:3]:
            console.print(f"      [dim]· {sample}[/dim]")
    console.print(
        f"\n{report['errors']} error(s), {report['warnings']} warning(s), "
        f"{report['notes']} note(s) over {report['trades_checked']:,} trades."
    )


@app.command()
def repair(
    apply: Annotated[bool, typer.Option(help="Write the changes; otherwise report only")] = False,
    names: Annotated[bool, typer.Option(help="Also re-derive asset names from the PDFs")] = False,
) -> None:
    """Apply the parser's deterministic corrections to rows already stored.

    Dates a century early and symbols contradicted by their own asset name.
    Reports by default, because both rewrite user-visible data.
    """
    from trade_bot.disclosures import repair_asset_names, repair_stored_trades

    settings, store = _store()
    result = repair_stored_trades(store, apply=apply)
    table = Table(title="Repairs" + ("" if apply else " (dry run)"))
    table.add_column("what")
    table.add_column("rows", justify="right")
    table.add_row("transaction dates a century early", str(result["dates_repaired"]))
    table.add_row("symbols contradicted by their name", str(result["symbols_repaired"]))
    table.add_row("symbols that are not tickers at all", str(result["symbols_cleared"]))
    console.print(table)
    for sample in result["date_samples"] + result["symbol_samples"] + result["cleared_samples"]:
        console.print(f"  [dim]· {sample}[/dim]")
    if names:
        console.print(
            _run_with_progress(
                "repair-names",
                "repair",
                lambda reporter: repair_asset_names(
                    settings, store, apply=apply, reporter=reporter
                ),
            )
        )
    if not apply:
        console.print("[yellow]dry run — pass --apply to write these[/yellow]")


@app.command()
def backtest(
    top: Annotated[int, typer.Option(help="How many top-ranked members to follow")] = 10,
    years: Annotated[float, typer.Option(help="How far back to walk forward from")] = 2.0,
    horizon: Annotated[int, typer.Option(help="Holding period in days (0 = configured)")] = 0,
) -> None:
    """Test the premise: what would following the top-ranked members have earned?

    Walk-forward and lookahead-free — members are ranked at each rebalance
    using only outcomes known on that date.
    """
    from trade_bot.backtest import run_backtest

    settings, store = _store()
    repository = PriceRepository(settings)
    result = _run_with_progress(
        "backtest",
        "backtest",
        lambda reporter: run_backtest(
            settings, store, repository, reporter,
            top_n=top, years=years, horizon_days=horizon or None,
        ).summary(),
    )
    if not result.get("positions"):
        console.print(f"[yellow]{result.get('note', 'no positions taken')}[/yellow]")
        return
    table = Table(title=f"Following the top {top} members, {result['horizon_days']}-day holds")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("positions taken", str(result["positions"]))
    table.add_row("period", f"{result['start']} to {result['end']}")
    table.add_row("mean return", f"{result['mean_return']:+.2%}")
    table.add_row("mean benchmark", f"{result['mean_benchmark_return']:+.2%}")
    table.add_row("mean excess", f"{result['mean_excess_return']:+.2%}")
    table.add_row("median excess", f"{result['median_excess_return']:+.2%}")
    table.add_row("beat benchmark", f"{result['beat_benchmark_rate']:.0%}")
    table.add_row("unique symbols", str(result["unique_symbols"]))
    table.add_row("unique members", str(result["unique_members"]))
    console.print(table)
    console.print(
        "[dim]A measurement, not a promise: one window, small samples, and who "
        "happens to be filing changes over time.[/dim]"
    )


@app.command()
def board(
    limit: Annotated[int, typer.Option(help="How many opportunities to print")] = 10,
) -> None:
    """Print the opportunity board with its reasoning."""
    settings, store = _store()
    repository = PriceRepository(settings)
    recommendations = build_recommendations(settings, store, repository)
    if not recommendations:
        console.print("[yellow]nothing on the board; run `trade-bot refresh` first[/yellow]")
        return
    for item in recommendations[:limit]:
        console.print(
            f"\n[bold cyan]{item.symbol}[/bold cyan] "
            f"conviction [bold]{item.conviction:.3f}[/bold] "
            f"· {item.buyer_count} buyer(s)"
            + (f" · {item.seller_count} seller(s)" if item.seller_count else "")
        )
        for reason in item.reasons:
            console.print(f"  [dim]•[/dim] {reason}")


@watch_app.command("create")
def watch_create(
    name: Annotated[str, typer.Argument(help="Watchlist name")],
    members: Annotated[list[str], typer.Option("--member", help="politician_id to copy")],
    amount: Annotated[float, typer.Option(help="Dollars per mirrored trade")] = 1000.0,
    mode: Annotated[
        str,
        typer.Option(help="fixed_dollars|proportional_to_amount|percent_of_equity"),
    ] = "fixed_dollars",
) -> None:
    """Create a watchlist that mirrors one or more members."""
    settings, store = _store()
    watchlist = create_watchlist(
        settings, store, name, list(members), allocation_mode=mode, allocation_value=amount
    )
    repository = PriceRepository(settings)
    result = mirror_watchlist(settings, store, repository, watchlist.watchlist_id)
    console.print(
        f"[green]created[/green] {watchlist.name} ({watchlist.watchlist_id}) "
        f"with {result['fills_added']} mirrored fills"
    )


@watch_app.command("list")
def watch_list() -> None:
    """Show every watchlist and its notional performance."""
    settings, store = _store()
    repository = PriceRepository(settings)
    watchlists = store.list_watchlists()
    if not watchlists:
        console.print("no watchlists yet; create one with `trade-bot watch create`")
        return
    table = Table("watchlist", "members", "fills", "equity", "return", "unrealised", "realised")
    for watchlist in watchlists:
        portfolio = build_portfolio(settings, store, repository, watchlist["watchlist_id"])
        table.add_row(
            portfolio.name,
            str(len(watchlist["politician_ids"])),
            str(portfolio.fill_count),
            f"${portfolio.equity:,.0f}",
            f"{portfolio.total_return:+.2%}",
            f"${portfolio.unrealized_pnl:,.0f}",
            f"${portfolio.realized_pnl:,.0f}",
        )
    console.print(table)


@watch_app.command("mirror")
def watch_mirror(
    watchlist_id: Annotated[str, typer.Argument(help="Watchlist id")],
) -> None:
    """Bring one watchlist up to date with the latest disclosures."""
    settings, store = _store()
    repository = PriceRepository(settings)
    result = _run_with_progress(
        f"mirror:{watchlist_id}",
        "watchlists",
        lambda reporter: mirror_watchlist(settings, store, repository, watchlist_id, reporter),
    )
    console.print_json(json.dumps(result, default=str))


@secrets_app.command("set")
def secrets_set(service: Annotated[str, typer.Argument(help="Keyring service name")]) -> None:
    """Store an optional API key (nothing here is required)."""
    api_key = typer.prompt(f"API key for {service!r}", hide_input=True)
    if not api_key.strip():
        console.print("[red]refusing to store an empty key[/red]")
        raise typer.Exit(code=1)
    set_api_key(service, api_key.strip())
    console.print(f"[green]stored[/green] a key for {service!r}")


@secrets_app.command("delete")
def secrets_delete(service: Annotated[str, typer.Argument(help="Keyring service name")]) -> None:
    """Remove a stored API key."""
    delete_api_key(service)
    console.print(f"[green]removed[/green] any key for {service!r}")


@app.command("db-check")
def db_check() -> None:
    """Report what the database URL resolves to, and whether it connects.

    Prints the host, user and database but never the password, so a failing
    CI log says which target was attempted instead of leaving a masked
    secret and an unexplained error.
    """
    from sqlalchemy import text

    from trade_bot.database import describe_database_url

    settings = get_settings()
    console.print(f"target: {describe_database_url(settings.database_url)}")
    store = Store(settings.database_url)
    try:
        with store.engine.connect() as connection:
            version = connection.execute(text("select version()")).scalar_one()
            user = connection.execute(text("select current_user")).scalar_one()
    except Exception as exc:
        console.print(f"[red]connection failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]connected[/green] as {user}")
    console.print(str(version)[:80])


@app.command("db-init")
def db_init() -> None:
    """Create any missing tables in the configured database.

    Works against local SQLite or a Supabase/Postgres `TRADE_BOT_DATABASE_URL`.
    """
    # `_store()` runs `Store.initialize()`, which is the whole job here.
    settings, _ = _store()
    backend = "postgres" if settings.database_url.startswith("postgres") else "sqlite"
    console.print(f"[green]initialised[/green] {backend} schema (ct_* tables)")


@app.command()
def export(
    out: Annotated[str, typer.Option(help="Directory to write the static site into")] = "dist",
    members: Annotated[int, typer.Option(help="Cap member pages rendered")] = 250,
    tickers: Annotated[int, typer.Option(help="Cap ticker pages rendered")] = 250,
) -> None:
    """Render the dashboard to static HTML for free hosting.

    The published snapshot is read-only: refreshing and editing watchlists
    need the local app.
    """
    from trade_bot.export import export_site

    settings, store = _store()
    result = _run_with_progress(
        "export",
        "export",
        lambda reporter: export_site(
            settings, store, Path(out), reporter, max_members=members, max_tickers=tickers
        ),
    )
    if not result.get("ok", True):
        console.print(f"[red]export failed:[/red] {result.get('error')}")
        raise typer.Exit(code=1)
    if not result.get("pages"):
        console.print("[red]export produced no pages[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]exported[/green] {result.get('pages', 0)} pages to {result.get('out_dir', out)}"
    )
    if result.get("failed"):
        console.print(f"[yellow]{len(result['failed'])} pages failed[/yellow]")


@secrets_app.command("status")
def secrets_status() -> None:
    """Show which optional providers have a stored key."""
    settings = get_settings()
    table = Table("provider", "service", "stored")
    for label, service in (
        ("OpenAI", settings.openai_keyring_name),
        ("Twelve Data", settings.twelve_data_keyring_name),
        ("Quiver", settings.quiver_keyring_name),
    ):
        table.add_row(label, service, "yes" if get_api_key(service) else "no")
    console.print(table)


@app.command()
def agents(
    install: Annotated[bool, typer.Option(help="Write and load the agents")] = False,
    uninstall: Annotated[bool, typer.Option(help="Unload and remove them")] = False,
) -> None:
    """Keep the dashboard and its tunnel running across reboots.

    Started by hand in a terminal, the dashboard dies when the terminal
    closes, with nothing to say why. These are per-user
    launchd agents: no sudo, no system daemon, and they stop when you log out
    -- the right scope for a service that only exists while this Mac is awake.

    They do not solve the other half of that failure. A long-running server
    serves *new* templates through *old* Python, because Jinja reloads from
    disk and the module does not, so a code change needs a restart:

        launchctl kickstart -k gui/$(id -u)/com.example.copytrade.serve

    Prints what it would do unless you pass a flag.
    """
    import plistlib
    import shutil
    import subprocess
    import sys

    settings = get_settings()
    root = settings.project_root
    target = Path.home() / "Library" / "LaunchAgents"
    source = root / "deploy" / "launchd"
    cloudflared = shutil.which("cloudflared") or "/opt/homebrew/bin/cloudflared"
    fields = {"python": sys.executable, "root": str(root), "cloudflared": cloudflared}
    domain = f"gui/{os.getuid()}"
    plans = [(path.stem, path, target / path.name) for path in sorted(source.glob("*.plist"))]

    if not install and not uninstall:
        console.print("[bold]Would install:[/bold]")
        for label, _source, destination in plans:
            console.print(f"  {label}  ->  {destination}")
        console.print("")
        console.print(f"  python:      {fields['python']}")
        console.print(f"  working dir: {fields['root']}")
        console.print(f"  cloudflared: {cloudflared}")
        console.print("[dim]--install writes and loads them; --uninstall removes them.[/dim]")
        console.print(
            f"[dim]After a code change: launchctl kickstart -k {domain}/"
            "com.example.copytrade.serve[/dim]"
        )
        return

    for label, template, destination in plans:
        if uninstall:
            subprocess.run(
                ["launchctl", "bootout", f"{domain}/{label}"],
                check=False,
                capture_output=True,
            )
            destination.unlink(missing_ok=True)
            console.print(f"[yellow]removed[/yellow] {label}")
            continue
        # Parsed and re-dumped rather than written as substituted text, so a
        # path containing an ampersand cannot produce a plist that launchd
        # silently declines to load.
        parsed = plistlib.loads(template.read_text().format(**fields).encode())
        target.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            plistlib.dump(parsed, handle)
        # Unload first so a reinstall picks up an edited plist. It fails
        # loudly when nothing is loaded, which is the normal case on a first
        # install and not worth showing anyone.
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True
        )
        result = subprocess.run(
            ["launchctl", "bootstrap", domain, str(destination)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            console.print(f"[red]{label} did not load:[/red] {result.stderr.strip()}")
        else:
            console.print(f"[green]loaded[/green] {label}")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Override the bind host")] = None,
    port: Annotated[int | None, typer.Option(help="Override the bind port")] = None,
    reload: Annotated[bool, typer.Option(help="Autoreload on source changes")] = False,
) -> None:
    """Serve the dashboard."""
    settings = get_settings()
    # Hosting platforms assign a port at run time and expect the process to
    # bind every interface; a container listening on 127.0.0.1 is unreachable
    # from outside itself and reads as a service that never came up.
    env_port = os.environ.get("PORT")
    in_container = env_port is not None
    uvicorn.run(
        "trade_bot.web:app",
        host=host or ("0.0.0.0" if in_container else settings.host),
        port=port or (int(env_port) if env_port else settings.port),
        reload=reload,
    )


if __name__ == "__main__":
    app()
