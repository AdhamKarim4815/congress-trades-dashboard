# The interactive dashboard, for hosts that run a real process.
#
# A static snapshot (`trade-bot export`) is fast and cheap to host but
# read-only. This image is the same application with its buttons working:
# refresh on demand, watchlists, mirrored positions.
#
# Deliberately plain: no build step, no compiled assets, no entrypoint script.
# It runs the same `trade-bot serve` command used locally, so a hosted
# instance cannot quietly drift into behaving differently from a laptop.

FROM python:3.12-slim

# uv comes from PyPI rather than a copied binary from another image: one fewer
# external tag to pin, and the version is resolved at build time by the same
# index everything else uses.
RUN pip install --no-cache-dir uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Install into the system environment, so the `trade-bot` console script
    # lands on PATH and CMD needs no venv activation.
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# Dependencies first, so editing application code does not reinstall them.
# --frozen installs exactly what uv.lock pins, the same versions CI resolves.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra postgres --extra vision

COPY . .
RUN uv sync --frozen --extra postgres --extra vision

# `serve` reads PORT itself and binds every interface when it is set, so there
# is nothing to interpolate here and no shell in the way of signal handling.
EXPOSE 8765
CMD ["trade-bot", "serve"]
