from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import RTDETR

from common import PROJECT_ROOT, environment_snapshot, jsonable, load_yaml, sha256_file, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_dataset(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = load_yaml(data_yaml)
    root_value = Path(str(data.get("path", ".")))
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value).resolve()
    images = root / str(data[split])
    return images, root / "labels" / split


def coco_ground_truth(images_dir: Path, labels_dir: Path) -> tuple[dict[str, Any], dict[str, int]]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        ids[str(image_path.resolve())] = image_id
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = labels_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, bw, bh = map(float, line.split())
            if int(class_id) != 0:
                raise ValueError(f"Classe inesperada em {label_path}: {class_id}")
            box_w, box_h = bw * width, bh * height
            x, y = (cx - bw / 2) * width, (cy - bh / 2) * height
            annotations.append(
                {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [x, y, box_w, box_h], "area": box_w * box_h, "iscrowd": 0}
            )
            annotation_id += 1
    dataset = {
        "info": {"description": "BOTOS RT-DETR evaluation"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "boto-cinza", "supercategory": "animal"}],
    }
    return dataset, ids


def official_coco_metrics(model: RTDETR, images_dir: Path, labels_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ground_truth, image_ids = coco_ground_truth(images_dir, labels_dir)
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    predictions: list[dict[str, Any]] = []
    results = model.predict(
        source=str(images_dir), imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=args.conf, iou=args.iou, max_det=args.max_det, stream=True, verbose=False,
    )
    for result in results:
        image_id = image_ids[str(Path(result.path).resolve())]
        for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            x1, y1, x2, y2 = xyxy
            predictions.append(
                {"image_id": image_id, "category_id": int(class_id) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": confidence}
            )
    if not predictions:
        return {"predictions": 0, "error": "Modelo nao produziu predicoes no confidence_floor."}
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    # COCOeval calcula os indices padrao de AP somente para maxDets=100.
    # O validador Ultralytics continua usando args.max_det (300 por padrao),
    # enquanto esta tabela preserva a convencao oficial COCO 1/10/100.
    evaluator.params.maxDets = [1, 10, 100]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ["AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    return {"predictions": len(predictions), "metrics": dict(zip(names, [float(item) for item in evaluator.stats])), "summary": output.getvalue()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia RT-DETR em val ou no teste temporal fechado")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Recusando sobrescrever avaliacao existente: {args.output}")
    model = RTDETR(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, iou=args.iou, max_det=args.max_det,
        plots=True, save_json=True, project=str(args.output.parent.resolve()), name=args.output.name,
        exist_ok=False, verbose=True,
    )
    images_dir, labels_dir = resolve_dataset(args.data, args.split)
    coco = official_coco_metrics(model, images_dir, labels_dir, args)
    report = {
        "split": args.split,
        "test_is_evaluation_only": args.split == "test",
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights.resolve()),
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": sha256_file(args.data.resolve()),
        "settings": vars(args),
        "ultralytics_metrics": jsonable(metrics.results_dict),
        "speed_ms_per_image": jsonable(metrics.speed),
        "official_coco_area_metrics": coco,
        "environment": environment_snapshot(),
    }
    write_json(args.output / "EVALUATION.json", report)
    print(f"Avaliacao concluida: {args.output / 'EVALUATION.json'}")


if __name__ == "__main__":
    main()
