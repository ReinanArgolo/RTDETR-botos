from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica e extrai com seguranca o ZIP do dataset")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sha256", help="SHA-256 esperado do ZIP")
    args = parser.parse_args()
    archive_path = args.zip_path.resolve()
    destination = args.destination.resolve()
    if args.sha256:
        observed = sha256_file(archive_path)
        if observed.lower() != args.sha256.lower():
            raise SystemExit(f"SHA-256 divergente: esperado {args.sha256}, observado {observed}")

    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Entrada corrompida no ZIP: {bad}")
        roots = {Path(name).parts[0] for name in archive.namelist() if Path(name).parts}
        if len(roots) != 1:
            raise SystemExit(f"ZIP deve ter exatamente uma raiz; encontrado: {sorted(roots)}")
        dataset_root = destination / next(iter(roots))
        if dataset_root.exists():
            raise SystemExit(f"Recusando sobrescrever dataset existente: {dataset_root}")
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if destination not in target.parents and target != destination:
                raise SystemExit(f"Caminho inseguro no ZIP: {info.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)

    manifest = dataset_root / "MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path = dataset_root / row["path"]
            if not path.is_file() or path.stat().st_size != int(row["bytes"]):
                raise SystemExit(f"Arquivo ausente ou tamanho divergente: {path}")
            if sha256_file(path) != row["sha256"]:
                raise SystemExit(f"SHA-256 divergente apos extracao: {path}")
    print(f"Dataset extraido e verificado em: {dataset_root}")


if __name__ == "__main__":
    main()

