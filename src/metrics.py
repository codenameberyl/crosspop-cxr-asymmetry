"""
Performance metrics for model discrimination and calibration evaluation.

Computes AUROC, specificity at clinical sensitivity operating points, ECE, MCE,
and Brier Scores, along with baseline OOD detection metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


# ---------------------------------------------------------------------------
# Discrimination Metrics
# ---------------------------------------------------------------------------
def auroc(y_true, y_prob):
    """Compute Area Under Receiver Operating Characteristic Curve (AUROC)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def specificity_at_sensitivity(y_true, y_prob, target_sensitivity):
    """Calculate specificity at the lowest threshold meeting target sensitivity."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_points = tpr >= target_sensitivity

    if not valid_points.any():
        return float("nan")

    best_fpr = np.min(fpr[valid_points])
    return float(1.0 - best_fpr)


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------
def _bin_stats(y_true, y_prob, n_bins):
    """Group prediction probabilities into uniform equal-width bins and compute per-bin metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    conf, acc, weight = [], [], []
    total_samples = len(y_prob)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = mask.sum()
        if count == 0:
            continue

        conf.append(y_prob[mask].mean())
        acc.append(y_true[mask].mean())
        weight.append(count / total_samples)

    return np.array(conf), np.array(acc), np.array(weight)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Standard Equal-Width ECE (Guo et al., 2017). Upward-biased at small n
    (few samples per bin) -- see bootstrap_debiased_ece and n_bins_for_sample_size
    for small-evaluation-set corrections."""
    conf, acc, weight = _bin_stats(y_true, y_prob, n_bins)
    if len(conf) == 0:
        return float("nan")
    return float(np.sum(weight * np.abs(conf - acc)))


def adaptive_calibration_error(y_true, y_prob, n_bins=15):
    """Equal-frequency (adaptive) ECE: bins hold equal counts, not equal widths."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_prob)

    if n == 0 or len(y_true) == 0:
        return float("nan")

    # Quantile edges; unique() collapses duplicate edges from clustered probabilities
    edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
    
    if len(edges) < 2:
        return float(np.abs(y_prob.mean() - y_true.mean()))

    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge on the final bin so prob == 1.0 is properly captured
        mask = (y_prob >= lo) & (y_prob <= hi if i == len(edges) - 2 else y_prob < hi)
        
        count = mask.sum()
        if count == 0:
            continue
            
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (count / n) * abs(acc - conf)

    return float(ece)


def maximum_calibration_error(y_true, y_prob, n_bins=15):
    """Calculate Maximum Calibration Error (MCE)."""
    conf, acc, _ = _bin_stats(y_true, y_prob, n_bins)
    if len(conf) == 0:
        return float("nan")
    return float(np.max(np.abs(conf - acc)))


def brier_score(y_true, y_prob):
    """Calculate Brier score metric."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return float(np.mean((y_prob - y_true) ** 2))

    return float(brier_score_loss(y_true, y_prob))


def reliability_curve(y_true, y_prob, n_bins=15):
    """Extract confidence and accuracy points for rendering reliability curves."""
    conf, acc, _ = _bin_stats(y_true, y_prob, n_bins)
    return conf, acc


# Small-sample calibration corrections
def n_bins_for_sample_size(n, min_bins=5, max_bins=15):
    """Bin count scaled to roughly sqrt(n), clipped to [min_bins, max_bins].
    Callers should pass min_bins/max_bins from config (ECE_MIN_BINS/ECE_MAX_BINS)
    rather than relying on these defaults, to keep tuning centralised in config.py."""
    return int(np.clip(round(np.sqrt(max(n, 1))), min_bins, max_bins))


def bootstrap_debiased_ece(y_true, y_prob, n_bins=15, n_boot=200, seed=42):
    """Bootstrap bias-corrected ECE: debiased = ECE(full) - [mean(bootstrap ECE) - ECE(full)].
    Standard equal-width ECE is upward-biased at small n; this estimates that
    bias via the bootstrap and subtracts it. Most relevant for the 30-image
    K2N Togunwa holdout evaluations.

    Returns (debiased_ece, raw_ece, estimated_bias).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ece_full = expected_calibration_error(y_true, y_prob, n_bins)
    rng = np.random.default_rng(seed)
    boot_eces = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_eces[i] = expected_calibration_error(y_true[idx], y_prob[idx], n_bins)
    bias = float(boot_eces.mean()) - ece_full
    return float(ece_full - bias), float(ece_full), float(bias)


def all_metrics(y_true, y_prob, fixed_sensitivity=0.95, n_bins=15):
    """Compute complete metric performance suite including standard, adaptive,
    and (if n_bins was chosen via n_bins_for_sample_size) sample-size-scaled ECE."""
    return {
        "auroc": auroc(y_true, y_prob),
        "specificity_at_95_sens": specificity_at_sensitivity(y_true, y_prob, fixed_sensitivity),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "ece_adaptive": adaptive_calibration_error(y_true, y_prob, n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins),
        "brier": brier_score(y_true, y_prob),
        "n": int(len(y_true)),
        "n_positive": int(np.sum(np.asarray(y_true) == 1)),
    }


# ---------------------------------------------------------------------------
# Out-of-Distribution (OOD) Metrics
# ---------------------------------------------------------------------------
def confidence_score(y_prob):
    """Compute maximum softmax probability confidence score for binary prediction probabilities."""
    y_prob = np.asarray(y_prob, dtype=float)
    return np.maximum(y_prob, 1.0 - y_prob)


def ood_detection_auroc(conf_in, conf_ood):
    """Compute AUROC metric evaluating separation between in-distribution and OOD confidence."""
    conf_in = np.asarray(conf_in)
    conf_ood = np.asarray(conf_ood)

    y_labels = np.concatenate([np.ones_like(conf_in), np.zeros_like(conf_ood)])
    scores = np.concatenate([conf_in, conf_ood])

    if len(np.unique(y_labels)) < 2:
        return float("nan")

    return float(roc_auc_score(y_labels, scores))
