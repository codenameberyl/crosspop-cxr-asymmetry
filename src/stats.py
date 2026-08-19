"""
Statistical analysis, bootstrap validation, and hypothesis testing routines.

Includes percentile bootstrap confidence interval calculations, DeLong tests for correlated ROC curves,
and McNemar tests for paired model prediction comparisons.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Bootstrap Analysis
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, ci=0.95, seed=42, stratified=True):
    """Compute percentile bootstrap confidence intervals for arbitrary evaluation metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(y_true, y_prob)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_samples = len(y_true)

    boot_estimates = []
    for _ in range(n_boot):
        if stratified and len(idx_pos) > 0 and len(idx_neg) > 0:
            boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
        else:
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)

        val = metric_fn(y_true[boot_idx], y_prob[boot_idx])
        if not np.isnan(val):
            boot_estimates.append(val)

    boots = np.array(boot_estimates)
    if len(boots) == 0:
        return point_estimate, float("nan"), float("nan"), boots

    alpha = 1.0 - ci
    lower_bound = np.percentile(boots, 100 * alpha / 2)
    upper_bound = np.percentile(boots, 100 * (1.0 - alpha / 2))

    return point_estimate, float(lower_bound), float(upper_bound), boots


def bootstrap_diff_ci(boot_a, boot_b, ci=0.95):
    """Calculate confidence interval for differences between paired bootstrap distributions."""
    min_length = min(len(boot_a), len(boot_b))
    if min_length == 0:
        return float("nan"), float("nan"), float("nan")

    diffs = boot_a[:min_length] - boot_b[:min_length]
    alpha = 1.0 - ci

    mean_diff = float(np.mean(diffs))
    lower_bound = float(np.percentile(diffs, 100 * alpha / 2))
    upper_bound = float(np.percentile(diffs, 100 * (1.0 - alpha / 2)))

    return mean_diff, lower_bound, upper_bound


# ---------------------------------------------------------------------------
# DeLong Test Implementation
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    """Vectorized calculation of midranks for DeLong test implementation."""
    x = np.asarray(x)
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]

    # Identify ties using diffs
    unique_vals, inverse_idx, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    
    # Calculate average rank position for unique values
    sum_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    midranks = sum_ranks[inverse_idx]

    # Revert to original ordering
    original_order_ranks = np.empty_like(midranks)
    original_order_ranks[sorted_idx] = midranks
    return original_order_ranks


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Sun & Xu (2014) fast covariance computation algorithm for DeLong test."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def delong_test(y_true, prob_a, prob_b):
    """Perform DeLong hypothesis test comparing two correlated ROC curves."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")  # Sort positive class targets first

    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]

    aucs, cov = _fast_delong(preds, label_1_count)
    variance = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]

    if variance <= 0:
        p_val = 1.0 if aucs[0] == aucs[1] else 0.0
        return float(aucs[0]), float(aucs[1]), float(p_val)

    z_score = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return float(aucs[0]), float(aucs[1]), float(p_val)


# ---------------------------------------------------------------------------
# McNemar Test Implementation
# ---------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    """Perform McNemar test on paired binary predictions."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # Contingency discordances
    b = int(np.sum(correct_a & ~correct_b))  # A correct, B incorrect
    c = int(np.sum(~correct_a & correct_b))  # A incorrect, B correct

    total_discordant = b + c
    if total_discordant == 0:
        return b, c, 1.0

    p_value = stats.binomtest(b, total_discordant, 0.5).pvalue
    return b, c, float(p_value)
