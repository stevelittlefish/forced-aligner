#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Read host/port out of config.toml so there is a single source of truth.
read -r HOST PORT < <(python - <<'PY'
import tomllib
with open("config.toml", "rb") as f:
    c = tomllib.load(f).get("server", {})
print(c.get("host", "0.0.0.0"), c.get("port", 8830))
PY
)

exec uvicorn app.main:app --host "$HOST" --port "$PORT"
