"""
Statistical analysis, bootstrap validation, and hypothesis testing routines.

Includes:
- Percentile bootstrap CIs (legacy row-level, kept for backward compatibility
  but no longer used for pooled cross-run comparisons)
- Run-level bootstrap CI (PRIMARY unit of analysis for pooled-run comparisons)
- Cluster (image-level) bootstrap CI (secondary sensitivity check)
- TOST equivalence testing
- DeLong test for correlated ROC curves
- McNemar test for paired predictions
"""
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Legacy row-level bootstrap (SINGLE-RUN USE ONLY)
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, ci=0.95, seed=42, stratified=True):
    """Percentile bootstrap CI for one run's predictions.
    WARNING: assumes every row is an independent observation. Do not use on
    predictions pooled across multiple seeds/folds/architectures -- see module
    docstring."""
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
# PRIMARY: run-level bootstrap (unit of analysis = one run)
# ---------------------------------------------------------------------------
def run_level_bootstrap_ci(run_values, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI over per-run point estimates (e.g. one AUROC value per
    seed, or per seed x fold). Resamples runs, not raw prediction rows.
    Returns (mean_estimate, ci_lo, ci_hi, boot_array)."""
    run_values = np.asarray(run_values, dtype=float)
    run_values = run_values[~np.isnan(run_values)]
    n_runs = len(run_values)
    point_estimate = float(run_values.mean()) if n_runs > 0 else float("nan")
    if n_runs < 2:
        return point_estimate, float("nan"), float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[i] = run_values[idx].mean()
    alpha = 1.0 - ci
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_estimate, float(lo), float(hi), boot


def run_level_diff_ci(values_a, values_b, n_boot=2000, ci=0.95, seed=42, paired=True):
    """Bootstrap CI for the difference of two sets of per-run point estimates.

    If paired=True, values_a and values_b are assumed to be aligned by index
    (e.g. same seed evaluated under two conditions) and resampled with a
    shared bootstrap index, preserving pairing. If paired=False, they are
    resampled independently (use only when runs genuinely cannot be paired).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)

    if paired:
        if len(a) != len(b):
            raise ValueError("paired=True requires values_a and values_b of equal length")
        n = len(a)
        if n < 2:
            return float(np.mean(a - b)), float("nan"), float("nan"), np.array([])
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = a[idx].mean() - b[idx].mean()
    else:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return float(np.mean(a) - np.mean(b)), float("nan"), float("nan"), np.array([])
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx_a = rng.integers(0, na, size=na)
            idx_b = rng.integers(0, nb, size=nb)
            boot[i] = a[idx_a].mean() - b[idx_b].mean()

    alpha = 1.0 - ci
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(np.mean(a) - np.mean(b))
    return point, float(lo), float(hi), boot


# ---------------------------------------------------------------------------
# SECONDARY: cluster (image-level) bootstrap sensitivity check
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(y_true, y_prob, image_ids, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI resampling unique image clusters rather than individual rows.

    y_true, y_prob, image_ids must be aligned 1-D arrays of the same length
    (rows can repeat the same image_id multiple times, e.g. once per run).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    image_ids = np.asarray(image_ids)

    unique_ids = np.unique(image_ids)
    order = np.argsort(image_ids, kind="stable")
    sorted_ids = image_ids[order]
    starts = np.searchsorted(sorted_ids, unique_ids)
    ends = np.searchsorted(sorted_ids, unique_ids, side="right")
    index_lookup = {uid: order[s:e] for uid, s, e in zip(unique_ids, starts, ends)}

    rng = np.random.default_rng(seed)
    n_images = len(unique_ids)
    point_estimate = metric_fn(y_true, y_prob)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique_ids, size=n_images, replace=True)
        idx = np.concatenate([index_lookup[c] for c in chosen])
        boot[i] = metric_fn(y_true[idx], y_prob[idx])

    alpha = 1.0 - ci
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point_estimate), float(lo), float(hi), boot


# ---------------------------------------------------------------------------
# TOST equivalence testing
# ---------------------------------------------------------------------------
def tost_equivalence(diff_boot, margin, alpha=0.05):
    """Two one-sided test for equivalence on a bootstrap distribution of
    differences (condition_a - condition_b). margin must be fixed in advance."""
    diff_boot = np.asarray(diff_boot)
    if len(diff_boot) == 0:
        return float("nan"), False
    p_upper = float(np.mean(diff_boot >= margin))
    p_lower = float(np.mean(diff_boot <= -margin))
    p_tost = max(p_upper, p_lower)
    return p_tost, p_tost < alpha


def estimate_equivalence_margin(seed_level_values, k=2.0):
    """Data-driven equivalence margin = k * SD of per-seed point estimates for
    a single fixed configuration. Computed once, treated as pre-registered."""
    vals = np.asarray(seed_level_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan")
    return float(k * vals.std(ddof=1))


# ---------------------------------------------------------------------------
# DeLong Test Implementation
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    """Vectorized calculation of midranks for DeLong test implementation."""
    x = np.asarray(x)
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    unique_vals, inverse_idx, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    sum_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    midranks = sum_ranks[inverse_idx]
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
    order = (-y_true).argsort(kind="mergesort")
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
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    total_discordant = b + c
    if total_discordant == 0:
        return b, c, 1.0
    p_value = stats.binomtest(b, total_discordant, 0.5).pvalue
    return b, c, float(p_value)
