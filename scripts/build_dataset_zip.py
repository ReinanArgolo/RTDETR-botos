from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


DATASET_ID = "BOTOS_RTDETR_E2C_TEST_20260915_r1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_files(source_root: Path, split: str):
    images_dir = source_root / "images" / split
    labels_dir = source_root / "labels" / split
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label ausente para {image}")
        yield image, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o ZIP auditavel do dataset RT-DETR BOTOS")
    parser.add_argument("--dev-root", type=Path, required=True, help="Dataset E2c contendo train e val")
    parser.add_argument("--test-root", type=Path, required=True, help="Dataset temporal contendo test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--leakage-report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Recusando sobrescrever: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, int]] = {}
    all_items: list[tuple[Path, str, str]] = []
    for split, source_root in (("train", args.dev_root), ("val", args.dev_root), ("test", args.test_root)):
        pairs = list(paired_files(source_root, split))
        split_counts[split] = {"images": len(pairs), "labels": len(pairs)}
        for image, label in pairs:
            all_items.append((image, f"{DATASET_ID}/images/{split}/{image.name}", split))
            all_items.append((label, f"{DATASET_ID}/labels/{split}/{label.name}", split))

    # Sem `path`: o Ultralytics usa a pasta do proprio YAML como raiz.
    # `path: .` seria resolvido a partir do cwd e quebraria a portabilidade.
    data_yaml = "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: {0: boto-cinza}\n"
    provenance = {
        "dataset_id": DATASET_ID,
        "created_for": "RT-DETR training, development validation and closed temporal test",
        "development_split": "v3_E2c_balanced_cap75_gap10",
        "development_release": "BOTOS_PPGZOO_v3_EXP1_20260829_r1",
        "test_source": "botos_yolo_temporal_split/test",
        "test_policy": "evaluation_only; never train or tune on test",
        "split_counts": split_counts,
        "class": {"id": 0, "name": "boto-cinza"},
        "limitations": [
            "Exact-image hashing does not prove independence by animal identity.",
            "The closed test comes from a held-out temporal interval of the legacy DJI_0203 video.",
            "Validation groups G02, G03 and G08 remain development validation, not the final test.",
        ],
    }
    readme = f"""# {DATASET_ID}\n\nDataset YOLO de uma classe para RT-DETR.\n\n- treino/validacao: PPGZOO v3 E2c, revisado por humano;\n- teste: intervalo temporal legado fechado;\n- classe 0: `boto-cinza`;\n- nao usar `images/test` durante treino, escolha de hiperparametros ou early stopping.\n\nExecute a auditoria antes de treinar:\n\n```bash\npython scripts/verify_dataset.py --data data/{DATASET_ID}/data.yaml\n```\n"""

    manifest_buffer = io.StringIO()
    writer = csv.DictWriter(manifest_buffer, fieldnames=["path", "split", "bytes", "sha256"])
    writer.writeheader()
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(args.output, "w", compression=compression, compresslevel=6, allowZip64=True) as archive:
        archive.writestr(f"{DATASET_ID}/data.yaml", data_yaml)
        archive.writestr(f"{DATASET_ID}/README.md", readme)
        archive.writestr(f"{DATASET_ID}/PROVENANCE.json", json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
        for source, arcname, split in all_items:
            digest = sha256_file(source)
            size = source.stat().st_size
            archive.write(source, arcname)
            row = {"path": arcname.removeprefix(f"{DATASET_ID}/"), "split": split, "bytes": size, "sha256": digest}
            entries.append(row)
            writer.writerow(row)
        if args.source_summary:
            archive.write(args.source_summary, f"{DATASET_ID}/audit/source_split_summary.json")
        if args.leakage_report:
            archive.write(args.leakage_report, f"{DATASET_ID}/audit/source_leakage_report.json")
        archive.writestr(f"{DATASET_ID}/MANIFEST.csv", manifest_buffer.getvalue())

    digest = sha256_file(args.output)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(args.output), "sha256": digest, "files": len(entries), "splits": split_counts}, indent=2))


if __name__ == "__main__":
    main()
