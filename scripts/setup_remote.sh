#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python - <<'PY'
import torch
import ultralytics
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("CUDA disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}:", torch.cuda.get_device_name(index))
PY

