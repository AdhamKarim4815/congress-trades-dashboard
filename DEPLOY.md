# Deploying your own copy

Everything below is a template. Replace every placeholder (`YOUR_...`, `example.com`) with your own values. Nothing here is tied to any particular account, domain or provider, and no credentials belong in the repository: pass them as environment variables or repository secrets.

The disclosures this app reads are public record. Your watchlists and notional portfolios are the part worth protecting, so put any hosted copy that can write behind authentication.

## Options at a glance

| | Local app | Static snapshot | Hosted app |
|---|---|---|---|
| What | The full FastAPI dashboard on your machine | Static HTML rendered by `trade-bot export` | The same dashboard in a container |
| Database | Local SQLite (default) | Whatever the export read from | Postgres you provide |
| Can edit watchlists and refresh | Yes | No (read-only by construction) | Yes |
| Runs when | Your machine is on | Always, on any static host | Always (free tiers may sleep) |

Start with the local app. Add the others only if you need them.

## 1. Local app

```bash
uv sync
uv run trade-bot refresh      # first run downloads and parses filings
uv run trade-bot serve        # http://127.0.0.1:8765
```

Runtime data (filings, prices, OCR reads, reports, the SQLite database) is created under `var/` the first time it is needed. That directory is ignored by git and safe to delete: it rebuilds on the next refresh.

Set a descriptive `TRADE_BOT_DISCLOSURES__USER_AGENT` that identifies you (for example your project name and a contact address you control). The SEC and the disclosure sites ask scrapers to identify themselves.

### Keep it running on macOS

Two per-user launchd agents keep the server (and optionally a tunnel) alive. No sudo is needed.

```bash
uv run trade-bot agents            # prints what it would install
uv run trade-bot agents --install  # writes to ~/Library/LaunchAgents and loads them
uv run trade-bot agents --uninstall
```

The templates live in `deploy/launchd/`. The labels use `com.example.copytrade.*`; change them if you want your own reverse-DNS prefix. After changing code, restart the server agent, because a long-running process serves new templates through old Python:

```bash
launchctl kickstart -k gui/$(id -u)/com.example.copytrade.serve
```

The tunnel agent is optional and assumes you already created a Cloudflare tunnel named `trade-bot-live` with `cloudflared`. Skip it if you do not use one.

## 2. Static snapshot

```bash
uv run trade-bot refresh
uv run trade-bot export --out dist
```

Upload the contents of `dist/` to any static host. The export includes a `robots.txt` and an `X-Robots-Tag: noindex` header file, but that only deters crawlers. If the snapshot should be private, use your host's access control.

Interactive features (refresh, watchlist editing) are hidden in the snapshot because there is no server to post to.

## 3. Postgres instead of SQLite

Install the extra and point the app at your database:

```bash
uv sync --extra postgres
export TRADE_BOT_DATABASE_URL='postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@YOUR_HOST:5432/YOUR_DB'
uv run trade-bot db-check     # prints host, user and database (never the password)
uv run trade-bot db-init      # creates the tables
uv run trade-bot status
```

Notes for hosted Postgres:

- If you run the pipeline from CI, use a host that is reachable over IPv4. Many CI runners are IPv4-only, so a database that publishes only an IPv6 address will fail to resolve before it authenticates. Connection poolers usually publish IPv4.
- Special characters in a password (`@`, `/`, `#`, `?`, spaces) are percent-encoded automatically, but pasting the value without surrounding quotes or newlines avoids confusion.
- The app disables server-side prepared statements for Postgres, so a transaction-mode pooler also works.
- Prefer a dedicated database role that can read and write only this app's `ct_*` tables, rather than an administrator credential. Such a role cannot create tables, so apply any schema change with an administrator role first, then let the app connect with the limited one.
- Enable row level security on the tables if your provider exposes them through an HTTP API, and keep the API roles locked out. This app connects directly over Postgres instead.

Leave `TRADE_BOT_DATABASE_URL` unset to use local SQLite.

## 4. Scheduled refresh with GitHub Actions

`deploy/examples/refresh-workflow.example.yml` is a workflow you can copy to `.github/workflows/refresh.yml` in your own repository. It runs the pipeline once a day against your Postgres database, exports a snapshot and stores it as a build artifact. Add these repository secrets (Settings, Secrets and variables, Actions):

| Secret | Value |
|---|---|
| `TRADE_BOT_DATABASE_URL` | the `postgresql+psycopg://...` string for your database |
| `OPENAI_API_KEY` | optional; without it the scan-reading stage is skipped |

Add a step that publishes `dist/` to your static host of choice, using that host's own deploy action and a token stored as a secret.

Heavy work (downloading and parsing filing PDFs, fetching hundreds of price series, reading news feeds) belongs on a CI runner or your own machine, not in a serverless function with a tight CPU limit.

## 5. Hosted interactive app (Docker)

The `Dockerfile` runs the same `trade-bot serve` command used locally. `serve` reads `PORT` itself and binds every interface when it is set.

`render.yaml` is an example blueprint for Render's free web service, but any host that runs a container works. Set these environment variables in the host's dashboard:

| Key | Value |
|---|---|
| `TRADE_BOT_DATABASE_URL` | the Postgres URL for your database |
| `OPENAI_API_KEY` | optional |
| `TRADE_BOT_SESSION_SECRET` | any long random string, so sessions survive restarts |

A free instance has about 512 MB of memory, and parsing filing PDFs is the one thing that can exhaust it, so the example blueprint lowers the download concurrency and the per-sync filing cap.

Before the first visit, put the service behind authentication (a reverse proxy, your host's access feature, or a zero-trust access product). A hosted instance can write to your database and trigger syncs, so do not leave it open to the internet.

## Optional API keys

Neither is needed for the dashboard to work:

- Twelve Data is a fallback price source. Store a key on the Secrets page, or set it through your system keychain.
- OpenAI reads scanned paper filings. Without a key that stage is skipped. Reads are cached under `var/ocr` so each document is paid for once per machine.

## Operating notes

- The first run is the slow one: it downloads every PTR PDF for the configured years. Later runs only fetch filings it has not seen.
- Scanned paper filings cannot be read without the OpenAI stage. They are recorded with an explicit parse error and listed on the Data page, so the gap is visible rather than silent.
- Prices come from Nasdaq's public endpoint first, then Stooq and Yahoo, then Twelve Data if you have stored a key.
