"""Validate a YOLO-format dataset folder.

This script is intentionally dependency-free (stdlib only).

Usage:
  python scripts/validate_yolo_dataset.py --data data.yaml

What it checks:
- images/train and images/val exist (via data.yaml)
- for every image, a matching label txt exists (labels/<split>/<stem>.txt)
- for every label, an image exists
- label lines are parseable and class ids are non-negative integers
- normalized coords are within [0, 1]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_yaml_minimal(path: Path) -> dict:
    """A tiny YAML reader for the limited keys we use (path/train/val/names/nc).

    This avoids adding PyYAML as a dependency.
    Supports:
      key: value
      names:
        0: class
        1: class
    """

    data: dict = {}
    current_section = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.startswith(" ") and current_section == "names":
            # e.g. "  0: cat"
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                k, v = parts
                try:
                    idx = int(k.strip())
                except ValueError:
                    continue
                data.setdefault("names", {})[idx] = v.strip().strip("\"'")
            continue

        # new top-level key
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("\"'")
            if v == "":
                current_section = k
                data[k] = {}
            else:
                current_section = None
                data[k] = v

    return data


def _iter_images(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _label_for_image(dataset_root: Path, split: str, img_path: Path) -> Path:
    # Standard YOLO layout: labels/<split>/<stem>.txt
    # We assume images are inside <dataset_root>/<train|val> (could be nested)
    # so we only use the image stem.
    return dataset_root / "labels" / split / f"{img_path.stem}.txt"


def _parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 columns, got {len(parts)}")
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return cls, x, y, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml", help="dataset yaml")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset yaml not found: {data_path}")
        return 2

    cfg = _read_yaml_minimal(data_path)
    root = Path(cfg.get("path", ".")).expanduser()
    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")

    train_dir = (root / train_rel).resolve()
    val_dir = (root / val_rel).resolve()

    # Determine split names from the relative paths (best effort)
    # If train_rel is "images/train" we use "train".
    def split_name(rel: str, fallback: str) -> str:
        parts = Path(rel).parts
        return parts[-1] if parts else fallback

    train_split = split_name(train_rel, "train")
    val_split = split_name(val_rel, "val")

    names = cfg.get("names", {})
    max_class = max(names.keys()) if isinstance(names, dict) and names else None

    print(f"YAML: {data_path}")
    print(f"root: {root}")
    print(f"train: {train_dir} (split='{train_split}')")
    print(f"val:   {val_dir} (split='{val_split}')")
    if max_class is not None:
        print(f"classes: 0..{max_class} ({len(names)} names)")

    problems = 0

    def check_split(images_dir: Path, split: str) -> None:
        nonlocal problems
        if not images_dir.exists():
            print(f"ERROR: missing images dir: {images_dir}")
            problems += 1
            return

        imgs = list(_iter_images(images_dir))
        print(f"[{split}] images: {len(imgs)}")

        # image -> label existence
        missing_labels = 0
        bad_labels = 0
        labels_seen: set[Path] = set()

        for img in imgs:
            lab = _label_for_image(root, split, img)
            labels_seen.add(lab)
            if not lab.exists():
                missing_labels += 1
                continue

            # validate label contents
            for i, raw in enumerate(lab.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    cls, x, y, w, h = _parse_label_line(line)
                except Exception as e:
                    print(f"ERROR: {lab}:{i}: {e} | line='{raw}'")
                    bad_labels += 1
                    continue

                if cls < 0:
                    print(f"ERROR: {lab}:{i}: class id must be >= 0")
                    bad_labels += 1
                if max_class is not None and cls > max_class:
                    print(f"ERROR: {lab}:{i}: class id {cls} exceeds max defined {max_class}")
                    bad_labels += 1

                for v, name in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
                    if not (0.0 <= v <= 1.0):
                        print(f"ERROR: {lab}:{i}: {name}={v} not in [0,1]")
                        bad_labels += 1

        if missing_labels:
            print(f"[{split}] WARNING: missing labels for {missing_labels} images")
        if bad_labels:
            print(f"[{split}] ERROR: found {bad_labels} invalid label lines")

        # label -> image existence (only in the split folder)
        label_dir = root / "labels" / split
        if label_dir.exists():
            label_files = list(label_dir.rglob("*.txt"))
            orphans = 0
            img_stems = {p.stem for p in imgs}
            for lf in label_files:
                if lf.stem not in img_stems:
                    orphans += 1
            if orphans:
                print(f"[{split}] WARNING: {orphans} label files without matching images")
        else:
            print(f"[{split}] WARNING: missing label dir: {label_dir}")

        problems += bad_labels

    check_split(train_dir, train_split)
    check_split(val_dir, val_split)

    if problems:
        print(f"\nFAILED: found {problems} issue(s)")
        return 1

    print("\nOK: dataset looks consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
