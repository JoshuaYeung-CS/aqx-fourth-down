"""Out-of-sample scoring: Brier score, log loss, calibration and game-clustered intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p, y, n_bins=10, strategy="quantile") -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if not m.any():
            continue
        n = int(m.sum())
        obs = float(y[m].mean())
        se = np.sqrt(max(obs * (1 - obs), 1e-4) / n)
        rows.append(dict(bin=k, n=n, predicted=float(p[m].mean()), observed=obs,
                         lo=max(0.0, obs - 1.96 * se), hi=min(1.0, obs + 1.96 * se)))
    return pd.DataFrame(rows)


def ece(p, y, n_bins=10):
    t = calibration_table(p, y, n_bins, strategy="uniform")
    return float(np.sum(t["n"] * np.abs(t["predicted"] - t["observed"])) / t["n"].sum())


def calibration_slope_intercept(p, y):
    """Logistic recalibration: logit(y) ~ a + b * logit(p). Perfect calibration is a=0, b=1."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def clustered_interval(groups, values_fn, arrays, n_boot=1000, seed=0, alpha=0.05):
    """Percentile interval for a statistic, resampling whole games (plays in a game are
    correlated, so a play-level bootstrap would be too narrow)."""
    groups = np.asarray(groups)
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(codes))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        stats.append(values_fn(*[np.asarray(a)[idx] for a in arrays]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarise(p, y, groups, baseline=None, n_boot=500) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = dict(n=int(len(y)), base_rate=float(y.mean()), brier=brier(p, y), log_loss=log_loss(p, y),
               ece=ece(p, y))
    out["brier_ci"] = clustered_interval(groups, brier, [p, y], n_boot=n_boot)
    a, b = calibration_slope_intercept(p, y)
    out["calibration_intercept"], out["calibration_slope"] = a, b
    if baseline is not None:
        bb = brier(baseline, y)
        out["brier_baseline"] = bb
        out["brier_skill"] = 1 - out["brier"] / bb
        out["brier_skill_ci"] = clustered_interval(
            groups, lambda p_, b_, y_: 1 - brier(p_, y_) / brier(b_, y_), [p, baseline, y],
            n_boot=n_boot)
    return out
