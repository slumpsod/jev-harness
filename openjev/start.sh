#!/usr/bin/env bash
# The local Jev: the SemIf bridge (:33003), then the System One adapter (:33001), in the background; logs in ./logs.
# Pick the GPU with CUDA_VISIBLE_DEVICES. Uses ./.venv if it exists.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
mkdir -p logs
nohup "$PY" semif_bridge.py > logs/bridge.log 2>&1 &
echo "bridge: pid $!, loading (about 30 s)"
for _ in $(seq 1 90); do curl -sf http://127.0.0.1:${SEMIF_PORT:-33003}/health | grep -q '"up"' && break; sleep 2; done
nohup "$PY" systemone.py > logs/adapter.log 2>&1 &
echo "adapter: pid $! on :${SYSTEMONE_PORT:-33001}"
