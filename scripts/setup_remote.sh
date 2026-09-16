#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python - <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError:
    raise SystemExit(1)

compatible = (
    torch.__version__.startswith("2.6.0")
    and torchvision.__version__.startswith("0.21.0")
    and str(torch.version.cuda).startswith("12.4")
)
raise SystemExit(0 if compatible else 1)
PY
then
  .venv/bin/python -m pip install --upgrade --force-reinstall -r requirements-torch-cu124.txt
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python - <<'PY'
import torch
import ultralytics

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("ultralytics:", ultralytics.__version__)
print("CUDA disponivel:", torch.cuda.is_available())
if not str(torch.version.cuda).startswith("12.4"):
    raise SystemExit("ERRO: era esperado um wheel PyTorch CUDA 12.4 (cu124).")
if not torch.cuda.is_available():
    raise SystemExit(
        "ERRO: CUDA continua indisponivel. Confira nvidia-smi, driver e CUDA_VISIBLE_DEVICES."
    )
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {properties.name} ({properties.total_memory / 2**30:.1f} GiB)")
PY
