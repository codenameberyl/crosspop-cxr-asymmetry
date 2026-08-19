"""
Data loading, stratification, downsampling, and manifest generation.

Provides deterministic utilities for creating cross-validation folds, downsampling datasets
while maintaining 1:1 balance, and outputting trackable manifest DataFrames.
"""

from pathlib import Path
from collections import defaultdict
import random
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    """Return sorted list of valid image file paths in the given folder."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def collect_records(class_dirs, class_to_idx):
    """Traverse directories and assemble image path and label mapping records."""
    records = []
    for cls in sorted(class_dirs):
        label = class_to_idx[cls]
        for p in list_images(class_dirs[cls]):
            records.append({"path": str(p), "label": label, "class": cls})
    return records


def group_by_class(records):
    """Group list of image record dicts by class name."""
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["class"]].append(rec)
    return dict(grouped)


def stratified_holdout(records, holdout_size, seed):
    """Extract a stratified target holdout set while retaining proportional class balance."""
    rng = random.Random(seed)
    by_class = group_by_class(records)
    n_total = len(records)

    holdout, working = [], []
    for cls, recs in sorted(by_class.items()):
        recs_copy = recs[:]
        rng.shuffle(recs_copy)

        # Calculate exact per-class split count
        k = round(holdout_size * len(recs_copy) / n_total) if n_total > 0 else 0
        if holdout_size > 0:
            k = max(1, k)
        k = min(k, len(recs_copy))

        holdout.extend(recs_copy[:k])
        working.extend(recs_copy[k:])

    return holdout, working


def kfold_splits(records, n_folds, seed):
    """Generate train/validation splits for stratified K-fold cross-validation."""
    rng = random.Random(seed)
    by_class = group_by_class(records)

    fold_assignments = {}
    for cls, recs in sorted(by_class.items()):
        recs_copy = recs[:]
        rng.shuffle(recs_copy)
        for i, r in enumerate(recs_copy):
            fold_assignments[r["path"]] = i % n_folds

    folds = []
    for f in range(n_folds):
        val = [r for r in records if fold_assignments[r["path"]] == f]
        train = [r for r in records if fold_assignments[r["path"]] != f]
        folds.append((train, val))

    return folds


def balanced_downsample(records, total_size, class_to_idx, seed):
    """Downsample dataset records to target size under 1:1 binary class parity."""
    if total_size % 2 != 0:
        raise ValueError(f"total_size must be an even integer for 1:1 balance, received {total_size}")

    per_class = total_size // 2
    rng = random.Random(seed)
    by_class = group_by_class(records)

    chosen = []
    for cls, recs in sorted(by_class.items()):
        if len(recs) < per_class:
            raise ValueError(f"Class {cls!r} has {len(recs)} records, but {per_class} are required.")
        recs_copy = recs[:]
        rng.shuffle(recs_copy)
        chosen.extend(recs_copy[:per_class])

    return chosen


def records_to_manifest(records, split_name, dataset, direction, size, fold, seed):
    """Convert sample record dicts into an annotated DataFrame manifest."""
    df = pd.DataFrame(records)
    if len(df) == 0:
        df = pd.DataFrame(columns=["path", "label", "class"])

    df.insert(0, "split", split_name)
    df.insert(1, "dataset", dataset)
    df.insert(2, "direction", direction)
    df.insert(3, "size", size)
    df.insert(4, "fold", fold)
    df.insert(5, "seed", seed)
    return df


def attach_modes(df, mode_lookup):
    """Attach original image color mode metadata to manifest DataFrame."""
    if len(df) == 0:
        df["mode"] = []
        return df

    df = df.copy()
    df["mode"] = df["path"].map(mode_lookup).fillna("unknown")
    return df
