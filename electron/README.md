# Trade Bot Desktop

An Electron shell around the existing FastAPI dashboard (`src/trade_bot/web.py`) --
a native window instead of a browser tab, so every action already wired up there is
a button rather than something typed into a terminal.

`main.js` doesn't reimplement anything: on launch it spawns
`../.venv/bin/trade-bot serve --port 8850`, waits for `/health`, then opens a window
pointed at it. Quitting the app stops that backend process.

## Setup

```sh
# once, from the trade-bot project root:
uv sync              # or `uv sync --extra ml` for the ML/Generate pages

# then, from this directory:
npm install
npm start
```

## Notes

- The backend runs with its working directory set to the project root, so its
  default relative database path (`var/trade_bot.sqlite3`) lands in the same
  place `trade-bot` CLI commands run from that directory would use.
- Set `TRADE_BOT_ELECTRON_PORT` to change the port if 8850 is taken.
- Everything else (env vars like `TRADE_BOT_DATABASE_URL`) is read by the Python
  process exactly as it would be for `trade-bot serve` on its own.
- This is a dev-mode launcher (`npm start`), not a packaged/signed `.app` --
  packaging (electron-builder, code signing, notarization) is a separate step
  not set up here.
