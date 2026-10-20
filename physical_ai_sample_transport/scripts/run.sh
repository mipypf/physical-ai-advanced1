#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$TASK_ROOT/.local/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
cd "$TASK_ROOT"
exec "$TASK_ROOT/.venv/bin/python" "$@"
