"""
Download the Nature Sci Data 2024 fingernail+Hb dataset from Figshare.

Source: https://doi.org/10.6084/m9.figshare.c.6760179
Paper:  https://www.nature.com/articles/s41597-024-03895-9

250 subjects (128 M / 122 F, ages 18-95), RGB finger images + metadata CSV
(patient IDs, dates, hemoglobin levels, bounding boxes for nail/skin regions).

Run from repo root:
    python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://ndownloader.figshare.com/files/49504041"
EXPECTED_MD5 = "7bd86daf16c69e370173d9a7a92474b0"
DEST_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DEST_DIR / "fingernail_hb_dataset.zip"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already downloaded: {dest}")
        return
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))


def verify_md5(path: Path, expected: str) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(f"md5 mismatch: got {actual}, expected {expected}")
    print(f"md5 ok: {actual}")


def extract(zip_path: Path, dest_dir: Path) -> None:
    target = dest_dir / "extracted"
    if target.exists() and any(target.iterdir()):
        print(f"already extracted: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.namelist(), desc="extracting"):
            zf.extract(member, target)
    print(f"extracted to {target}")


def main() -> int:
    download(URL, ZIP_PATH)
    verify_md5(ZIP_PATH, EXPECTED_MD5)
    extract(ZIP_PATH, DEST_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
