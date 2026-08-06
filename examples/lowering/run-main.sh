#!/usr/bin/env bash
set -euo pipefail

exec env \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python3 main.py "$@"
