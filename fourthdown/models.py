"""Fitting the four model components and packing them into one engine specification.

1. Value of a state: an expected-points linear regression (next score in the half on a
   field-position spline, down and distance), feeding an L2-regularised logistic
   win-probability regression on score margin, expected points, clock and Vegas line.
   This is the value function used to price every resulting state. nflverse's own
   ``vegas_wp`` column is used only as an external benchmark, never as an input.
2. Conversion probability: L2-regularised logistic regression on third and fourth downs.
3. Field-goal make probability: logistic regression in distance, roof and season.
4. Punt outcomes and yards beyond the marker: empirical distributions (game-weighted).

Every fit accepts per-row weights so the game-clustered bootstrap is just a reweighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import config, data, features as F


# ---------------------------------------------------------------- win probability
def ep_X(df: pd.DataFrame):
    return F.ep_matrix(df["down"], df["ydstogo"], df["yardline_100"], df["half_seconds_remaining"])


def fit_ep(A, y, w=None, ridge=1e-3):
    """Weighted least squares (tiny ridge for stability) of next-score points on the EP design."""
    w = np.ones(len(y)) if w is None else w
    Aw = A * w[:, None]
    return np.linalg.solve(A.T @ Aw + ridge * np.eye(A.shape[1]), Aw.T @ y).tolist()


def wp_xy(df: pd.DataFrame, ep):
    X = F.wp_matrix(df["score_differential"], df["game_seconds_remaining"],
                    df["half_seconds_remaining"], df["posteam_timeouts_remaining"],
                    df["defteam_timeouts_remaining"], df["receive_2h_ko"], df["posteam_spread"],
                    df["is_home"], ep)
    return X, df["posteam_win"].to_numpy()


# ---------------------------------------------------------------- logistic helpers
class Standardiser:
    def __init__(self, X, w=None):
        w = np.ones(len(X)) if w is None else w
        self.mean = np.average(X, axis=0, weights=w)
        sd = np.sqrt(np.average((X - self.mean) ** 2, axis=0, weights=w))
        self.scale = np.where(sd > 0, sd, 1.0)

    def __call__(self, X):
        return (X - self.mean) / self.scale


def fit_logit(Z, y, w, C):
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(Z, y, sample_weight=w)
    return [float(m.intercept_[0])] + [float(c) for c in m.coef_[0]]


def logit_predict(params, Z):
    p = np.asarray(params)
    return 1.0 / (1.0 + np.exp(-(p[0] + Z @ p[1:])))


def choose_C(Z, y, groups, grid=(0.003, 0.01, 0.03, 0.1, 1.0, 10.0)):
    """Pick the L2 strength by 5-fold cross-validated log loss, folds grouped by game."""
    cv = GroupKFold(n_splits=5)
    scores = {}
    for C in grid:
        losses = []
        for tr, te in cv.split(Z, y, groups):
            p = np.clip(logit_predict(fit_logit(Z[tr], y[tr], None, C), Z[te]), 1e-9, 1 - 1e-9)
            losses.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        scores[C] = float(np.mean(losses))
    return min(scores, key=scores.get), scores


# ---------------------------------------------------------------- conversion
def conv_X(df):
    return F.conv_matrix(df["ydstogo"], df["yardline_100"], df["is_4th"], df["posteam_spread"],
                         df["total_line"], df["season"])


def fg_X(df):
    return F.fg_matrix(df["distance"], df["indoor"], df["season"])


# ---------------------------------------------------------------- empirical distributions
def extra_yards_pmf(conv: pd.DataFrame, w=None) -> np.ndarray:
    """P(yards beyond the line to gain = e | converted, ydstogo bucket), e = 0..99."""
    succ = conv["converted"].to_numpy() == 1
    w = np.ones(len(conv)) if w is None else w
    b = F.ydstogo_bucket(conv["ydstogo"].to_numpy())
    e = conv["extra_yards"].to_numpy().astype(int)
    pmf = np.zeros((F.N_BUCKETS, 100))
    np.add.at(pmf, (b[succ], e[succ]), w[succ])
    pmf += 1e-9
    return pmf / pmf.sum(axis=1, keepdims=True)


def punt_pmfs(punts: pd.DataFrame, w=None):
    """For each punt spot (yardline_100 1..99): distribution of the next possession.

    Returns opp[99, 100] (opponent ball at y), own[99, 100] (punting team keeps it at y),
    td[99] (return touchdown). Rows sum to one. Spots are pooled over a +/- window that
    widens until it holds at least 40 (weighted) punts.
    """
    w = np.ones(len(punts)) if w is None else w
    spot = punts["yardline_100"].to_numpy().astype(int)
    owner = punts["next_owner"].to_numpy()
    nxt = np.nan_to_num(punts["next_yardline"].to_numpy(), nan=0).astype(int).clip(0, 99)
    by_spot_opp = np.zeros((100, 100))
    by_spot_own = np.zeros((100, 100))
    by_spot_td = np.zeros(100)
    o, m, t = owner == "opp", owner == "own", owner == "opp_td"
    np.add.at(by_spot_opp, (spot[o], nxt[o]), w[o])
    np.add.at(by_spot_own, (spot[m], nxt[m]), w[m])
    np.add.at(by_spot_td, spot[t], w[t])
    cs = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    c_opp, c_own, c_td = cs(by_spot_opp), cs(by_spot_own), cs(by_spot_td)
    total = by_spot_opp.sum(1) + by_spot_own.sum(1) + by_spot_td
    c_tot = cs(total)
    opp = np.zeros((99, 100)); own = np.zeros((99, 100)); td = np.zeros(99)
    for yl in range(1, 100):
        for h in (config.PUNT_WINDOW, 5, 8, 12, 20, 40, 99):
            lo, hi = max(yl - h, 1), min(yl + h, 99) + 1
            if c_tot[hi] - c_tot[lo] >= 40:
                break
        n = c_tot[hi] - c_tot[lo]
        opp[yl - 1] = (c_opp[hi] - c_opp[lo]) / n
        own[yl - 1] = (c_own[hi] - c_own[lo]) / n
        td[yl - 1] = (c_td[hi] - c_td[lo]) / n
    return opp, own, td


# ---------------------------------------------------------------- the whole engine
def game_weights(games: np.ndarray, rng) -> dict:
    sample = rng.integers(0, len(games), len(games))
    counts = np.bincount(sample, minlength=len(games))
    return dict(zip(games, counts.astype(float)))


def fit_components(train: pd.DataFrame, n_boot: int, seed: int = config.SEED, log=print):
    """Fit win probability, conversion, FG, extra-yards and punt components plus ``n_boot``
    game-clustered bootstrap replicates of each (one resample of games drives all five)."""
    conv = data.conversion_frame(train)
    fg = data.fg_frame(train)
    punts = data.punt_frame(train)
    wpf = data.wp_frame(train)

    Aep, yep = ep_X(wpf), wpf["next_score"].to_numpy(float)
    ep_point = fit_ep(Aep, yep)
    Xw, yw = wp_xy(wpf, Aep @ np.asarray(ep_point))
    sw = Standardiser(Xw)
    Zw = sw(Xw)
    C_wp, wp_cv = choose_C(Zw, yw, wpf["game_id"].to_numpy(), grid=(0.01, 0.1, 1.0))

    def fit_value(w):
        """EP regression, then the WP logistic on the EP it implies (same resample)."""
        if w is None:
            return ep_point, fit_logit(Zw, yw, None, C_wp)
        ep_b = fit_ep(Aep, yep, w)
        Z = sw(wp_xy(wpf, Aep @ np.asarray(ep_b))[0])
        return ep_b, fit_logit(Z, yw, w, C_wp)

    Xc, yc = conv_X(conv), conv["converted"].to_numpy()
    sc = Standardiser(Xc)
    Zc = sc(Xc)
    C_conv, conv_cv = choose_C(Zc, yc, conv["game_id"].to_numpy())
    Xf, yf = fg_X(fg), fg["made"].to_numpy()
    sf = Standardiser(Xf)
    Zf = sf(Xf)
    C_fg, fg_cv = choose_C(Zf, yf, fg["game_id"].to_numpy())
    log(f"  win prob: n={len(wpf):,} C={C_wp}   conversion: n={len(conv):,} C={C_conv}   "
        f"field goals: n={len(fg):,} C={C_fg}")

    def one(ww, wc, wf, wx, wp_):
        ep_b, wp_b = fit_value(ww)
        return dict(ep=ep_b, wp=wp_b, conv=fit_logit(Zc, yc, wc, C_conv),
                    fg=fit_logit(Zf, yf, wf, C_fg), extra=extra_yards_pmf(conv, wx),
                    punt=punt_pmfs(punts, wp_))

    point = one(None, None, None, None, None)
    rng = np.random.default_rng(seed)
    games = np.array(sorted(set(wpf["game_id"]) | set(conv["game_id"]) | set(fg["game_id"])
                            | set(punts["game_id"])))
    boots = []
    for b in range(n_boot):
        gw = game_weights(games, rng)
        get = lambda d: d["game_id"].map(gw).fillna(0).to_numpy()
        boots.append(one(get(wpf), get(conv), get(fg), get(conv), get(punts)))
        if (b + 1) % 50 == 0:
            log(f"  bootstrap {b + 1}/{n_boot}")
    info = dict(n_wp=len(wpf), n_conv=len(conv), n_fg=len(fg), n_punt=len(punts), C_wp=C_wp,
                C_conv=C_conv, C_fg=C_fg, wp_cv=wp_cv, conv_cv=conv_cv, fg_cv=fg_cv,
                n_games=len(games))
    return dict(point=point, boots=boots, wp_scaler=sw, conv_scaler=sc, fg_scaler=sf, info=info)


def engine_spec(comps, constants: dict, n_calc_boot: int = config.N_BOOT_CALC) -> dict:
    p = comps["point"]
    r6 = lambda a: np.round(np.asarray(a), 6).tolist()
    spec = {
        "version": 1,
        "train_seasons": config.TRAIN_SEASONS,
        "constants": constants,
        "ep": {"features": F.EP_FEATURES, "params": p["ep"],
               "boot": [b["ep"] for b in comps["boots"][:n_calc_boot]]},
        "wp": {"features": F.WP_FEATURES, "mean": comps["wp_scaler"].mean.tolist(),
               "scale": comps["wp_scaler"].scale.tolist(), "params": p["wp"],
               "boot": [b["wp"] for b in comps["boots"][:n_calc_boot]]},
        "conv": {"features": F.CONV_FEATURES, "mean": comps["conv_scaler"].mean.tolist(),
                 "scale": comps["conv_scaler"].scale.tolist(), "params": p["conv"],
                 "boot": [b["conv"] for b in comps["boots"][:n_calc_boot]]},
        "fg": {"features": F.FG_FEATURES, "mean": comps["fg_scaler"].mean.tolist(),
               "scale": comps["fg_scaler"].scale.tolist(), "params": p["fg"],
               "boot": [b["fg"] for b in comps["boots"][:n_calc_boot]]},
        "extra": {"buckets": "1 | 2 | 3 | 4-5 | 6-10 | 11+", "pmf": r6(p["extra"])},
        "punt": {"opp": r6(p["punt"][0]), "own": r6(p["punt"][1]), "td": r6(p["punt"][2])},
    }
    return spec


def constants_from(df: pd.DataFrame) -> dict:
    ko = {str(s): round(data.kickoff_start(df, s), 2) for s in sorted(df["season"].unique())}
    return {
        "kickoff_start": ko,
        "xp_rate": round(data.xp_rate(df, config.TRAIN_SEASONS), 4),
        "clock": {"go_success": config.CLOCK_GO_SUCCESS, "go_fail": config.CLOCK_GO_FAIL,
                  "punt": config.CLOCK_PUNT, "fg": config.CLOCK_FG},
        "max_fg_distance": config.MAX_FG_DISTANCE,
        "min_punt_yardline": config.MIN_PUNT_YARDLINE,
        "default_season": int(max(config.AUDIT_SEASONS)),
    }
