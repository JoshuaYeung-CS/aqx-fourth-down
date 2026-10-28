"""Audit every fourth-down decision of a season and roll the results up by team and coach.

"Win probability left on the table" for a play is the pre-snap value of the recommended
call minus the pre-snap value of the call the coach actually made. It scores the decision,
never the outcome: a correct go that failed costs nothing, a bad punt that worked out still
counts. Intervals come from the game-clustered bootstrap: every replicate refits the
win-probability, conversion, field-goal, punt and gain models on the same resample of games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import config, data, features as F
from .engine import CALLS, Engine, Situation

CALL_INDEX = {c: i for i, c in enumerate(CALLS)}


def situation_from(df: pd.DataFrame) -> Situation:
    return Situation(
        score_diff=df["score_differential"].to_numpy(float), qtr=df["qtr"].to_numpy(float),
        quarter_seconds=df["quarter_seconds_remaining"].to_numpy(float),
        posteam_timeouts=df["posteam_timeouts_remaining"].to_numpy(float),
        defteam_timeouts=df["defteam_timeouts_remaining"].to_numpy(float),
        receive_2h_ko=df["receive_2h_ko"].to_numpy(float), spread=df["posteam_spread"].to_numpy(float),
        total=df["total_line"].to_numpy(float), is_home=df["is_home"].to_numpy(float),
        season=df["season"].to_numpy(float), indoor=df["indoor"].to_numpy(float))


def audit_plays(engine: Engine, plays: pd.DataFrame, boots: list) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns the per-play table and the (B, n) bootstrap matrix of WP lost."""
    plays = plays.reset_index(drop=True)
    n = len(plays)
    s = situation_from(plays).arrays(n, engine.c["default_season"])
    ytg = plays["ydstogo"].to_numpy(int)
    yl = plays["yardline_100"].to_numpy(int)
    V = engine.state_values(s)
    gathered = engine.gather_go(V, ytg, yl)
    pt = engine.score(s, V, ytg, yl, gathered=gathered)
    actual = plays["call"].map(CALL_INDEX).to_numpy()
    rec = pt["best"]
    rows = np.arange(n)
    ev_all = pt["ev_all"]
    lost = ev_all[rows, rec] - ev_all[rows, actual]
    ev = pt["ev"]
    sorted_ev = -np.sort(-np.nan_to_num(ev, nan=-1.0), axis=1)
    margin = sorted_ev[:, 0] - sorted_ev[:, 1]

    B = len(boots)
    lost_b = np.empty((B, n))
    agree = np.zeros(n)
    go_minus_kick_b = np.empty((B, n))
    for b, prm in enumerate(boots):
        Vb = engine.state_values(s, prm)
        o = engine.score(s, Vb, ytg, yl, params=prm)
        lost_b[b] = o["ev_all"][rows, rec] - o["ev_all"][rows, actual]
        agree += o["best"] == rec
        go_minus_kick_b[b] = o["ev"][:, 0] - np.nanmax(o["ev"][:, 1:], axis=1, initial=-1.0)
    agree /= max(B, 1)

    out = plays[["season", "week", "game_id", "play_id", "posteam", "defteam", "coach", "qtr",
                 "quarter_seconds_remaining", "score_differential", "ydstogo", "yardline_100",
                 "call", "desc", "posteam_win", "wp", "vegas_wp"]].copy()
    out["wp_go"], out["wp_punt"], out["wp_fg"] = pt["go"], pt["punt"], pt["fg"]
    out["p_convert"], out["p_make"] = pt["p_convert"], pt["p_make"]
    out["recommended"] = [CALLS[i] for i in rec]
    out["margin"] = margin
    out["wp_lost"] = lost
    out["wp_lost_lo"] = np.percentile(lost_b, 2.5, axis=0)
    out["wp_lost_hi"] = np.percentile(lost_b, 97.5, axis=0)
    out["agreement"] = agree
    go_minus_kick = pt["ev"][:, 0] - np.nanmax(pt["ev"][:, 1:], axis=1, initial=-1.0)
    out["go_advantage"] = go_minus_kick
    out["go_advantage_lo"] = np.percentile(go_minus_kick_b, 2.5, axis=0)
    out["go_advantage_hi"] = np.percentile(go_minus_kick_b, 97.5, axis=0)
    out["wp_actual"] = ev_all[rows, actual]
    # where the win-probability model is least reliable (see README, reliability map)
    out["late"] = (((plays["qtr"] == 4) & (plays["quarter_seconds_remaining"] < 300))
                   | ((plays["qtr"] == 2) & (plays["quarter_seconds_remaining"] < 120))).to_numpy()
    out["went"] = (out["call"] == "go").astype(int)
    return out, lost_b


def league_go_model(plays: pd.DataFrame, seed=config.SEED) -> np.ndarray:
    """Cross-fitted P(go | situation) describing league behaviour in the same season: a
    logistic regression with a field-position spline, yards to go, and score margin and
    clock terms. Each play's expectation comes from a fit that never saw that play's game."""
    X = go_design(plays)
    y = (plays["call"] == "go").to_numpy(int)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, plays["game_id"]):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return pred


def go_design(plays: pd.DataFrame) -> np.ndarray:
    ytg = plays["ydstogo"].to_numpy(float)
    yl = plays["yardline_100"].to_numpy(float)
    diff = plays["score_differential"].to_numpy(float)
    t = plays["game_seconds_remaining"].to_numpy(float) / 3600.0
    r = 1.0 / np.sqrt(t + 0.01)
    lg = np.log(ytg)
    h = F._hinges(yl)
    trail = np.minimum(diff, 0); lead = np.maximum(diff, 0)
    cols = h + [lg, lg ** 2, (ytg == 1) * 1.0, lg * yl / 100, trail, lead, trail * r, lead * r,
                (diff < 0) * (t < 0.1), (diff < -3) * (t < 0.1), (diff > 0) * (t < 0.1),
                plays["half_seconds_remaining"].to_numpy(float) < 120]
    return np.column_stack([np.asarray(c, float) for c in cols])


def _team_boot(df: pd.DataFrame, col: str, n_boot=1000, seed=0):
    """Game-resampled 95% interval for the mean of `col` within one team (or coach)."""
    rng = np.random.default_rng(seed)
    per_game = df.groupby("game_id")[col].agg(["sum", "count"]).to_numpy()
    g = len(per_game)
    stats = []
    for _ in range(n_boot):
        pick = per_game[rng.integers(0, g, g)]
        stats.append(pick[:, 0].sum() / pick[:, 1].sum())
    return np.percentile(stats, [2.5, 97.5])


def rollup(audit: pd.DataFrame, lost_b: np.ndarray, by: str) -> pd.DataFrame:
    rows = []
    for key, idx in audit.groupby(by).groups.items():
        d = audit.loc[idx]
        tot_b = lost_b[:, idx].sum(axis=1)
        gro_lo, gro_hi = _team_boot(d, "go_over_expected")
        clear_go = d[(d["recommended"] == "go") & (d["agreement"] >= 0.9) & (d["margin"] >= 0.01)]
        rows.append({
            by: key,
            "team": d["posteam"].mode().iat[0],
            "games": d["game_id"].nunique(),
            "fourth_downs": len(d),
            "go_rate": d["went"].mean(),
            "disagreements": int((d["recommended"] != d["call"]).sum()),
            "wp_lost": d["wp_lost"].sum(),
            "wp_lost_lo": float(np.percentile(tot_b, 2.5)),
            "wp_lost_hi": float(np.percentile(tot_b, 97.5)),
            "wp_lost_per_game": d["wp_lost"].sum() / d["game_id"].nunique(),
            "wp_lost_excl_late": d.loc[~d["late"], "wp_lost"].sum(),
            "lost_kicking": d.loc[d["recommended"] == "go", "wp_lost"].sum(),
            "lost_going": d.loc[d["recommended"] != "go", "wp_lost"].sum(),
            "aggressiveness": 100 * d["go_over_expected"].mean(),
            "aggressiveness_lo": 100 * gro_lo,
            "aggressiveness_hi": 100 * gro_hi,
            "clear_go_spots": len(clear_go),
            "clear_go_followed": float(clear_go["went"].mean()) if len(clear_go) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("wp_lost", ascending=False).reset_index(drop=True)
    return out


def run_season(engine: Engine, df: pd.DataFrame, season: int, boots: list):
    plays = data.decision_frame(df, [season])
    audit, lost_b = audit_plays(engine, plays, boots)
    audit["go_expected"] = league_go_model(plays.reset_index(drop=True))
    audit["go_over_expected"] = audit["went"] - audit["go_expected"]
    teams = rollup(audit, lost_b, "posteam")
    coaches = rollup(audit, lost_b, "coach")
    league_b = lost_b.sum(axis=1)
    summary = {
        "season": season,
        "weeks": int(plays["week"].max()),
        "games": int(plays["game_id"].nunique()),
        "fourth_downs": int(len(audit)),
        "go_rate": float(audit["went"].mean()),
        "disagree_rate": float((audit["recommended"] != audit["call"]).mean()),
        "wp_lost_total": float(audit["wp_lost"].sum()),
        "wp_lost_total_ci": [float(np.percentile(league_b, 2.5)), float(np.percentile(league_b, 97.5))],
        "lost_by_kicking_share": float(audit.loc[audit["recommended"] == "go", "wp_lost"].sum()
                                       / max(audit["wp_lost"].sum(), 1e-12)),
        "should_go": int((audit["recommended"] == "go").sum()),
        "should_go_went": float(audit.loc[audit["recommended"] == "go", "went"].mean()),
        "confident_mistakes": int(((audit["wp_lost_lo"] > 0) & (audit["wp_lost"] > 0)).sum()),
        "late_share_of_plays": float(audit["late"].mean()),
        "late_share_of_wp_lost": float(audit.loc[audit["late"], "wp_lost"].sum()
                                       / max(audit["wp_lost"].sum(), 1e-12)),
        "team_rank_corr_excl_late": float(teams[["wp_lost", "wp_lost_excl_late"]]
                                          .corr(method="spearman").iat[0, 1]),
        "aggressiveness_vs_lost_spearman": float(teams[["aggressiveness", "wp_lost"]]
                                                 .corr(method="spearman").iat[0, 1]),
    }
    return audit, teams, coaches, summary
