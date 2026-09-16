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
    parser.add_argument("--device", type=int, default=0, help="GPU logica que sera usada no treino")
    parser.add_argument("--min-free-gib", type=float, default=0.0, help="VRAM livre minima exigida")
    args = parser.parse_args()
    environment = environment_snapshot()
    dataset = verify(args.data)
    report = {"environment": environment, "dataset": dataset, "ok": bool(dataset["ok"])}
    if not environment.get("cuda_available") and not args.allow_cpu:
        report["ok"] = False
        report["environment_error"] = "CUDA indisponivel; use --allow-cpu somente para smoke test."
    elif environment.get("cuda_available"):
        import torch

        if args.device >= torch.cuda.device_count():
            report["ok"] = False
            report["environment_error"] = f"GPU logica {args.device} nao existe neste ambiente."
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
            gpu_memory = {
                "logical_device": args.device,
                "name": torch.cuda.get_device_name(args.device),
                "free_gib": free_bytes / 2**30,
                "total_gib": total_bytes / 2**30,
                "minimum_required_gib": args.min_free_gib,
            }
            report["gpu_memory"] = gpu_memory
            if gpu_memory["free_gib"] < args.min_free_gib:
                report["ok"] = False
                report["environment_error"] = (
                    f"VRAM livre insuficiente na GPU {args.device}: "
                    f"{gpu_memory['free_gib']:.2f} GiB; minimo solicitado {args.min_free_gib:.2f} GiB."
                )
    write_json(args.output, report)
    if not report["ok"]:
        raise SystemExit(report.get("environment_error") or "Dataset reprovado no preflight")
    print("Preflight aprovado.")
    print(environment)


if __name__ == "__main__":
    main()
