#!/usr/bin/env bash
# Start the private, loopback-only FlyBrain Control Center (Panel v2).
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${FLYBRAIN_DATA_DIR:-$PROJECT_ROOT/data/processed/malecns-v1.0}"
OUTPUT_DIR="${FLYBRAIN_OUTPUT_DIR:-$PROJECT_ROOT/runs/panel}"
PANEL_PORT="${FLYBRAIN_PORT:-8765}"

if [[ ! "$PANEL_PORT" =~ ^[0-9]+$ ]] || (( 10#$PANEL_PORT < 1 || 10#$PANEL_PORT > 65535 )); then
  printf 'Invalid FLYBRAIN_PORT: %s (expected 1–65535).\n' "$PANEL_PORT" >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  printf 'MaleCNS dataset not found: %s\n' "$DATA_DIR" >&2
  printf 'Set FLYBRAIN_DATA_DIR to your normalized MaleCNS v1.0 directory.\n' >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
  UV="$HOME/.cargo/bin/uv"
else
  printf 'uv was not found. Install uv or add it to PATH.\n' >&2
  exit 1
fi

printf 'FlyBrain Control Center · Panel v2\n'
printf 'URL:       http://127.0.0.1:%s\n' "$PANEL_PORT"
printf 'MaleCNS:   %s\n' "$DATA_DIR"
printf 'Run output: %s\n' "$OUTPUT_DIR"
printf 'Sandbox:   QEMU guest disconnected until explicitly configured.\n'
printf 'Stop:      Ctrl+C\n\n'

# The packaged UI is served by FastAPI; no separate frontend server or host
# desktop adapter is launched. --locked prevents implicit lockfile updates.
exec "$UV" run --locked flybrain-panel \
  --data-directory "$DATA_DIR" \
  --output-directory "$OUTPUT_DIR" \
  --host 127.0.0.1 \
  --port "$PANEL_PORT"
