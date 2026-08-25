"""
Evaluation output aggregation, loading, and filtering helpers.

Utility functions for parsing filename strings, reading saved prediction CSVs,
and pooling metrics across multiple experimental runs.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def parse_run_id(name):
    """Extract metadata parameters (arch, direction, size, seed, fold) from filename stems."""
    stem = name.replace(".csv", "")
    if "__on__" in stem:
        stem = stem.split("__on__")[0]

    parts = stem.split("__")
    out = {
        "arch": parts[0],
        "direction": parts[1],
        "stem": stem
    }

    for param in parts[2:]:
        if param.startswith("size"):
            out["size"] = int(param[4:])
        elif param.startswith("seed"):
            out["seed"] = int(param[4:])
        elif param.startswith("fold"):
            out["fold"] = int(param[4:])

    return out


# Single source of truth for training-direction + eval-target -> analysis
# condition. Kept here (not duplicated per-notebook) so K2K/K2N/K2N_full/N2K/N2N
# labelling is consistent everywhere it's used (notebooks 04, 05, 06, 07, 08, 10).
_CONDITION_MAP = {
    ("K2N", "togunwa_holdout"): "K2N",
    ("K2N", "kermany_test"): "K2K",
    ("K2N", "togunwa_full"): "K2N_full",
    ("N2K", "kermany_test"): "N2K",
    ("N2K", "togunwa_holdout"): "N2N",
    ("K2N", "musa_id_test"): "K2M",
}

def derive_condition(direction, target):
    """Maps (training direction, evaluation target) -> analysis condition
    label. Falls back to a descriptive string for any combination not in
    the standard design (e.g. future additional eval targets)."""
    return _CONDITION_MAP.get((direction, target), f"{direction}_on_{target}")


def load_predictions(pred_dir):
    """Load prediction CSV files from output folder into metadata DataFrames and numpy arrays."""
    pred_dir = Path(pred_dir)
    meta_rows = []
    arrays = {}

    for csv_file in sorted(pred_dir.glob("*.csv")):
        info = parse_run_id(csv_file.name)
        target = csv_file.name.replace(".csv", "").split("__on__")[-1] if "__on__" in csv_file.name else ""

        df = pd.read_csv(csv_file)
        entry = {
            "y_true": df["y_true"].values,
            "y_prob": df["y_prob"].values
        }

        if "image_id" in df.columns:
            entry["image_id"] = df["image_id"].values

        if {"logit_0", "logit_1"}.issubset(df.columns):
            entry["logits"] = df[["logit_0", "logit_1"]].values

        arrays[info["stem"] + "__on__" + target if target else info["stem"]] = entry
        meta_rows.append({
            **info,
            "target": target,
            "condition": derive_condition(info["direction"], target),
            "stem_full": (info["stem"] + "__on__" + target) if target else info["stem"],
            "file": csv_file.name,
        })

    meta = pd.DataFrame(meta_rows)
    return meta, arrays


def pool_predictions(meta, arrays, direction=None, size=None, arch=None,
                      condition=None, which="y_prob", include_image_id=False):
    """Filter prediction dictionary matching specified criteria and concatenate
    output arrays. NOTE: pooled predictions from multiple runs are correlated
    (see src/stats.py module docstring) -- use only for descriptive purposes
    (e.g. reliability diagrams) or as input to cluster_bootstrap_ci, never as
    input to the legacy row-level bootstrap_ci for inferential claims."""
    selected = meta

    if direction is not None:
        selected = selected[selected["direction"] == direction]
    if size is not None:
        selected = selected[selected["size"] == size]
    if arch is not None:
        selected = selected[selected["arch"] == arch]
    if condition is not None and "condition" in selected.columns:
        selected = selected[selected["condition"] == condition]

    y_true_list, y_prob_list, image_id_list = [], [], []

    for _, row in selected.iterrows():
        key = row["stem_full"] if "stem_full" in row and row["stem_full"] in arrays else row["stem"]
        entry = arrays[key]
        y_true_list.append(entry["y_true"])
        y_prob_list.append(entry[which])
        if include_image_id:
            if "image_id" in entry:
                image_id_list.append(entry["image_id"])
            else:
                image_id_list.append(np.arange(len(entry["y_true"])).astype(str))

    if not y_true_list:
        empty = np.array([])
        return (empty, empty, empty) if include_image_id else (empty, empty)

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)

    if include_image_id:
        image_id = np.concatenate(image_id_list)
        return y_true, y_prob, image_id

    return y_true, y_prob
