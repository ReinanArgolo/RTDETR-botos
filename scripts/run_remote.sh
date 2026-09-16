#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="BOTOS_RTDETR_E2C_TEST_20260915_r1"
ZIP_PATH="${1:-$ROOT/data_upload/$DATASET_ID.zip}"
RUN_ID="${2:-rtdetr_l_e2c_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="$ROOT/.venv/bin/python"
DATA_YAML="$ROOT/data/$DATASET_ID/data.yaml"
RUN_ROOT="$ROOT/runs/$RUN_ID"
CHECKSUM_FILE="$ROOT/checksums/$DATASET_ID.zip.sha256"

if [[ ! -x "$PYTHON" ]]; then
  echo "Ambiente ausente. Execute primeiro: bash scripts/setup_remote.sh" >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -f "$DATA_YAML" ]]; then
  EXPECTED_SHA="$(cut -d' ' -f1 "$CHECKSUM_FILE")"
  "$PYTHON" scripts/unpack_dataset.py "$ZIP_PATH" --destination "$ROOT/data" --sha256 "$EXPECTED_SHA"
fi
mkdir -p "$ROOT/runs/preflight"
"$PYTHON" scripts/preflight.py --data "$DATA_YAML" --output "$ROOT/runs/preflight/${RUN_ID}.json"
"$PYTHON" scripts/train.py --data "$DATA_YAML" --run-id "$RUN_ID"
BEST="$RUN_ROOT/weights/best.pt"
"$PYTHON" scripts/evaluate.py --weights "$BEST" --data "$DATA_YAML" --split val --output "$RUN_ROOT/evaluation/val"
"$PYTHON" scripts/evaluate.py --weights "$BEST" --data "$DATA_YAML" --split test --output "$RUN_ROOT/evaluation/test"
echo "Pipeline concluido em: $RUN_ROOT"
