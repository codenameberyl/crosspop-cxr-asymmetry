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

        if {"logit_0", "logit_1"}.issubset(df.columns):
            entry["logits"] = df[["logit_0", "logit_1"]].values

        arrays[info["stem"]] = entry
        meta_rows.append({**info, "target": target, "file": csv_file.name})

    return pd.DataFrame(meta_rows), arrays


def pool_predictions(meta, arrays, direction=None, size=None, arch=None, which="y_prob"):
    """Filter prediction dictionary matching specified criteria and concatenate output arrays."""
    selected = meta

    if direction is not None:
        selected = selected[selected["direction"] == direction]
    if size is not None:
        selected = selected[selected["size"] == size]
    if arch is not None:
        selected = selected[selected["arch"] == arch]

    y_true_list, y_prob_list = [], []

    for _, row in selected.iterrows():
        entry = arrays[row["stem"]]
        y_true_list.append(entry["y_true"])
        y_prob_list.append(entry[which])

    if not y_true_list:
        return np.array([]), np.array([])

    return np.concatenate(y_true_list), np.concatenate(y_prob_list)
