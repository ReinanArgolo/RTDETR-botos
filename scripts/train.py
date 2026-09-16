from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ultralytics import RTDETR

from common import PROJECT_ROOT, environment_snapshot, load_yaml, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina RT-DETR no BOTOS E2c")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/rtdetr_l_e2c.yaml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-id", default=f"rtdetr_l_e2c_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}")
    parser.add_argument("--device", help="Sobrescreve o device do YAML, por exemplo 0 ou 0,1")
    parser.add_argument("--batch", type=int, help="Sobrescreve batch do YAML")
    parser.add_argument("--epochs", type=int, help="Sobrescreve epochs do YAML")
    parser.add_argument("--imgsz", type=int, help="Sobrescreve image_size do YAML")
    parser.add_argument("--model", help="Checkpoint/arquitetura inicial")
    args = parser.parse_args()
    config = load_yaml(args.config)
    training = config["training"]
    augmentation = config["augmentation"]
    run_root = (PROJECT_ROOT / "runs" / args.run_id).resolve()
    if run_root.exists():
        raise SystemExit(f"Recusando sobrescrever run existente: {run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    model_source = args.model or config["model"]["weights"]
    requested = {
        "config": config,
        "overrides": vars(args),
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": sha256_file(args.data.resolve()),
        "model_source": model_source,
        "environment": environment_snapshot(),
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "test_used_for_training": False,
    }
    write_json(run_root.parent / f"{args.run_id}_REQUESTED.json", requested)

    model = RTDETR(model_source)
    results = model.train(
        data=str(args.data.resolve()),
        project=str(run_root.parent),
        name=args.run_id,
        exist_ok=False,
        epochs=args.epochs or training["epochs"],
        patience=training["patience"],
        imgsz=args.imgsz or training["image_size"],
        batch=args.batch or training["batch"],
        workers=training["workers"],
        device=args.device or training["device"],
        seed=training["seed"],
        optimizer=training["optimizer"],
        lr0=training["learning_rate"],
        lrf=training["final_lr_fraction"],
        weight_decay=training["weight_decay"],
        amp=training["amp"],
        deterministic=training["deterministic"],
        cache=training["cache"],
        save_period=training["save_period"],
        cos_lr=training["cos_lr"],
        close_mosaic=training["close_mosaic"],
        val=True,
        plots=True,
        **augmentation,
    )
    best = run_root / "weights/best.pt"
    final = {
        **requested,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256_file(best) if best.is_file() else None,
        "trainer_save_dir": str(results.save_dir) if hasattr(results, "save_dir") else str(run_root),
    }
    write_json(run_root / "TRAINING_COMPLETE.json", final)
    print(f"Treino concluido. Melhor checkpoint: {best}")


if __name__ == "__main__":
    main()

