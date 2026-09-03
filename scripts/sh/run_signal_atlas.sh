#!/usr/bin/env bash

cd "$(dirname "$0")/../.." || exit 1   # raiz del repo, sea cual sea el clon
LOG="${LOG:-results/logs/signal_atlas.log}"
mkdir -p results/logs

PY="${PY:-}"
[ -z "$PY" ] && [ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
[ -z "$PY" ] && [ -x .venv/bin/python ]         && PY=.venv/bin/python
[ -z "$PY" ] && PY=python
"$PY" scripts/signal_atlas.py "$@" > "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
