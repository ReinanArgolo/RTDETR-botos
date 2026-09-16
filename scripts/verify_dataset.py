from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_yaml, sha256_file, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def verify(data_yaml: Path, hash_images: bool = True) -> dict[str, Any]:
    config = load_yaml(data_yaml)
    root_value = Path(str(config.get("path", "."))).expanduser()
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value).resolve()
    names = config.get("names")
    class_count = len(names) if isinstance(names, (dict, list)) else int(config.get("nc", 0))
    if class_count != 1:
        raise ValueError(f"Era esperada uma classe; encontrado nc={class_count}")

    report: dict[str, Any] = {"data_yaml": str(data_yaml.resolve()), "root": str(root), "splits": {}}
    digest_locations: dict[str, list[str]] = {}
    errors: list[str] = []

    for split in ("train", "val", "test"):
        relative = config.get(split)
        if not relative:
            errors.append(f"split ausente no YAML: {split}")
            continue
        images_dir = root / str(relative)
        labels_dir = root / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"diretorio ausente: {images_dir} ou {labels_dir}")
            continue

        images = image_files(images_dir)
        counters = Counter(images=len(images), labels=0, positive_images=0, negative_images=0, boxes=0)
        stems: set[str] = set()
        for image in images:
            if image.stem in stems:
                errors.append(f"stem de imagem duplicado em {split}: {image.stem}")
            stems.add(image.stem)
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"label ausente: {label}")
                continue
            counters["labels"] += 1
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                counters["positive_images"] += 1
            else:
                counters["negative_images"] += 1
            counters["boxes"] += len(lines)
            for line_number, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"label invalido {label}:{line_number}: esperado 5 campos")
                    continue
                try:
                    class_id = int(parts[0])
                    coordinates = [float(item) for item in parts[1:]]
                except ValueError:
                    errors.append(f"label nao numerico {label}:{line_number}")
                    continue
                if class_id != 0:
                    errors.append(f"classe fora do protocolo {label}:{line_number}: {class_id}")
                if not all(0.0 <= item <= 1.0 for item in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
                    errors.append(f"caixa YOLO fora dos limites {label}:{line_number}")
            if hash_images:
                digest_locations.setdefault(sha256_file(image), []).append(f"{split}/{image.name}")

        extra_labels = [path for path in labels_dir.glob("*.txt") if path.stem not in stems]
        for label in extra_labels:
            errors.append(f"label sem imagem em {split}: {label.name}")
        report["splits"][split] = dict(counters)

    collisions = [locations for locations in digest_locations.values() if len({item.split("/", 1)[0] for item in locations}) > 1]
    report["cross_split_exact_hash_collisions"] = collisions
    if collisions:
        errors.append(f"{len(collisions)} imagem(ns) exatamente repetida(s) entre splits")

    manifest_path = root / "MANIFEST.csv"
    if manifest_path.is_file():
        manifest_files = 0
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                manifest_files += 1
                path = root / row["path"]
                if not path.is_file():
                    errors.append(f"arquivo do manifesto ausente: {path}")
                    continue
                if path.stat().st_size != int(row["bytes"]):
                    errors.append(f"tamanho divergente do manifesto: {path}")
                    continue
                if sha256_file(path) != row["sha256"]:
                    errors.append(f"SHA-256 divergente do manifesto: {path}")
        report["manifest"] = {"path": str(manifest_path), "files_verified": manifest_files, "ok": not any("manifesto" in item for item in errors)}
    else:
        report["manifest"] = None
    report["errors"] = errors
    report["ok"] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita estrutura, labels e vazamento exato do dataset YOLO")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-image-hashes", action="store_true")
    args = parser.parse_args()
    report = verify(args.data, hash_images=not args.skip_image_hashes)
    if args.output:
        write_json(args.output, report)
    for split, counts in report["splits"].items():
        print(split, counts)
    if not report["ok"]:
        for error in report["errors"][:50]:
            print("ERRO:", error)
        raise SystemExit(f"Dataset reprovado com {len(report['errors'])} erro(s).")
    print("Dataset aprovado; nenhuma colisao SHA-256 entre splits.")


if __name__ == "__main__":
    main()
