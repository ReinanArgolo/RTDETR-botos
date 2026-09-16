from __future__ import annotations

import argparse
from pathlib import Path

from common import environment_snapshot, write_json
from verify_dataset import verify


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida ambiente remoto e dataset antes do treino")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    environment = environment_snapshot()
    dataset = verify(args.data)
    report = {"environment": environment, "dataset": dataset, "ok": bool(dataset["ok"])}
    if not environment.get("cuda_available") and not args.allow_cpu:
        report["ok"] = False
        report["environment_error"] = "CUDA indisponivel; use --allow-cpu somente para smoke test."
    write_json(args.output, report)
    if not report["ok"]:
        raise SystemExit(report.get("environment_error") or "Dataset reprovado no preflight")
    print("Preflight aprovado.")
    print(environment)


if __name__ == "__main__":
    main()

