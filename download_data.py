"""
Phase 1 Project 3 - download the BGL log dataset (loghub family).

Source: https://github.com/logpai/loghub (BGL subset)
Zenodo archive: https://zenodo.org/records/8196385/files/BGL.zip?download=1

Usage:
    python -u download_data.py

Run this on Windows (native venv), not from a sandbox without real internet -
the file is ~700MB raw / a good chunk smaller zipped, so this needs a real
network connection.

What it does:
1. Downloads BGL.zip into data/raw/ (skips if already present).
2. Unzips it into data/raw/.
3. Prints the first 20 lines of the extracted log file as a sanity check -
   we have NOT seen a real line of this file yet, so before writing
   prepare_data.py we need to look at actual formatting/column layout here.
4. Does a rough line count and a rough count of "not a dash in the first
   field" lines, to sanity-check against the published stats:
   4,747,963 total lines, 348,460 anomalous (~7.34%).

If the Zenodo URL below has moved, check the BGL row at
https://github.com/logpai/loghub for the current link and update DATA_URL.
"""

import sys
import zipfile
from pathlib import Path

import requests

DATA_URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
RAW_DIR = Path(__file__).parent / "data" / "raw"
ZIP_PATH = RAW_DIR / "BGL.zip"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written:,}/{total:,} bytes ({pct:5.1f}%)", end="")
        print()


def unzip(zip_path: Path, dest_dir: Path) -> list[Path]:
    print(f"[unzip] {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"  archive contains: {names}")
        zf.extractall(dest_dir)
    return [dest_dir / n for n in names]


def sanity_check(log_path: Path, sample_lines: int = 20) -> None:
    print(f"\n[sanity check] {log_path}")
    print(f"  first {sample_lines} lines:")
    total = 0
    non_dash = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < sample_lines:
                print(f"    {line.rstrip()}")
            total += 1
            first_field = line.split(" ", 1)[0] if line.strip() else ""
            if first_field and first_field != "-":
                non_dash += 1
    print(f"\n  total lines:      {total:,}  (published: 4,747,963)")
    print(f"  non-'-' (alert):  {non_dash:,}  (published: 348,460, ~7.34%)")
    if total:
        print(f"  alert fraction:   {non_dash / total:.4f}")


def main() -> int:
    try:
        download(DATA_URL, ZIP_PATH)
    except requests.RequestException as e:
        print(f"[error] download failed: {e}", file=sys.stderr)
        print(
            "Check that the Zenodo URL is still current at "
            "https://github.com/logpai/loghub (BGL row) and update DATA_URL "
            "in this script if it moved.",
            file=sys.stderr,
        )
        return 1

    extracted = unzip(ZIP_PATH, RAW_DIR)
    log_files = [p for p in extracted if p.suffix == ".log" or p.name.upper() == "BGL.LOG"]
    if not log_files:
        # fall back: just look for anything that isn't the zip itself
        log_files = [p for p in RAW_DIR.iterdir() if p.is_file() and p.name != "BGL.zip"]
    if not log_files:
        print("[error] could not find an extracted log file - check data/raw/ manually", file=sys.stderr)
        return 1

    sanity_check(log_files[0])
    print("\nDone. Next: look at the printed sample lines together and confirm "
          "the column layout before writing prepare_data.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
