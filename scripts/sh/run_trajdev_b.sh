#!/usr/bin/env bash

cd "$(dirname "$0")/../.." || exit 1   # raiz del repo, sea cual sea el clon
LOG="${LOG:-results/logs/trajdev_b.log}"
mkdir -p results/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="${PY:-}"
[ -z "$PY" ] && [ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
[ -z "$PY" ] && [ -x .venv/bin/python ]         && PY=.venv/bin/python
[ -z "$PY" ] && PY=python
"$PY" scripts/run_trajdev_b.py "$@" > "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
