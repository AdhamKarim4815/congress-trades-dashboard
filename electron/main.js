// Electron shell for the copy-trading dashboard.
//
// Spawns the existing FastAPI app (`trade-bot serve`) as a child process --
// nothing here re-implements any logic, it just gives the Python web app
// (src/trade_bot/web.py) a native window instead of a browser tab, so
// refreshing disclosures, creating watchlists and reading the board are
// buttons rather than typed commands.

const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const PYTHON_BIN = path.join(PROJECT_ROOT, ".venv", "bin", "trade-bot");
const PORT = process.env.TRADE_BOT_ELECTRON_PORT || "8850";
const BASE_URL = `http://127.0.0.1:${PORT}`;
const HEALTH_URL = `${BASE_URL}/health`;

let backendProcess = null;
let mainWindow = null;
let quitting = false;

function waitForHealth(attemptsLeft) {
  return new Promise((resolve, reject) => {
    function attempt(remaining) {
      if (remaining <= 0) {
        reject(new Error("Timed out waiting for the backend to start."));
        return;
      }
      const request = http.get(HEALTH_URL, (response) => {
        if (response.statusCode === 200) {
          response.resume();
          resolve();
        } else {
          response.resume();
          setTimeout(() => attempt(remaining - 1), 250);
        }
      });
      request.on("error", () => setTimeout(() => attempt(remaining - 1), 250));
    }
    attempt(attemptsLeft);
  });
}

function startBackend() {
  if (!fs.existsSync(PYTHON_BIN)) {
    throw new Error(
      `The Python environment wasn't found at:\n${PYTHON_BIN}\n\n` +
        "Run `uv sync` in the trade-bot project directory first."
    );
  }
  backendProcess = spawn(PYTHON_BIN, ["serve", "--port", PORT], {
    cwd: PROJECT_ROOT,
    env: process.env,
  });
  backendProcess.stdout.on("data", (data) => process.stdout.write(`[trade-bot] ${data}`));
  backendProcess.stderr.on("data", (data) => process.stderr.write(`[trade-bot] ${data}`));
  backendProcess.on("exit", (code) => {
    backendProcess = null;
    if (!quitting) {
      dialog.showErrorBox(
        "Copy Trade backend stopped",
        `The Python backend process exited unexpectedly (code ${code}). ` +
          "Quit and reopen the app to restart it."
      );
    }
  });
  return waitForHealth(60);
}

function stopBackend() {
  quitting = true;
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
    backendProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#0d1117",
    title: "Copy Trade",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadURL(BASE_URL);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  try {
    await startBackend();
  } catch (error) {
    dialog.showErrorBox("Copy Trade failed to start", error.message);
    app.quit();
    return;
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", stopBackend);
app.on("will-quit", stopBackend);
process.on("exit", stopBackend);
