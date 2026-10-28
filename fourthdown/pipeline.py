"""One command from raw nflverse files to engine, metrics, audit tables, charts and site.

    python -m fourthdown pipeline            # everything
    python -m fourthdown pipeline --fast     # 40 bootstrap replicates instead of 200
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import audit as A, config, data, evaluate as E, models as M
from .engine import Engine

CACHE = config.ARTIFACTS / "cache"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_all(df: pd.DataFrame, n_boot: int):
    train = df[df["season"].isin(config.TRAIN_SEASONS)]
    log(f"fitting all components on {len(config.TRAIN_SEASONS)} seasons with {n_boot} bootstrap replicates")
    comps = M.fit_components(train, n_boot=n_boot, log=log)
    spec = M.engine_spec(comps, M.constants_from(df))
    config.ARTIFACTS.mkdir(exist_ok=True)
    with open(config.ENGINE_PATH, "w") as fh:
        json.dump(spec, fh, separators=(",", ":"))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "models.pkl", "wb") as fh:
        pickle.dump({"comps": comps}, fh)
    log(f"engine written: {config.ENGINE_PATH.stat().st_size / 1e6:.2f} MB")
    return comps


def holdout_metrics(df: pd.DataFrame, comps, engine: Engine) -> dict:
    train = df[df["season"].isin(config.TRAIN_SEASONS)]
    test = df[df["season"] == config.HOLDOUT_SEASON]
    out = {"holdout_season": config.HOLDOUT_SEASON, "train_seasons": config.TRAIN_SEASONS}
    cal = {}

    # -- win probability, against nflverse's vegas_wp on the same plays
    wt = data.wp_frame(test)
    wt = wt[wt["vegas_wp"].notna()]
    ep_hold = M.ep_X(wt) @ np.asarray(comps["point"]["ep"])
    out["ep_rmse"] = float(np.sqrt(np.mean((ep_hold - wt["next_score"]) ** 2)))
    out["ep_corr_nflverse"] = float(np.corrcoef(ep_hold, test.loc[wt.index, "ep"])[0, 1]) if "ep" in test else None
    X, y = M.wp_xy(wt, ep_hold)
    p_mine = M.logit_predict(comps["point"]["wp"], comps["wp_scaler"](X))
    out["wp"] = E.summarise(p_mine, y, wt["game_id"], n_boot=300)
    out["wp_nflverse_vegas_wp"] = E.summarise(wt["vegas_wp"].to_numpy(), y, wt["game_id"], n_boot=300)
    cal["wp"] = E.calibration_table(p_mine, y, 20, "uniform").to_dict("records")
    close = (wt["score_differential"].abs() <= 8).to_numpy()
    gap = np.abs(p_mine - wt["vegas_wp"].to_numpy())
    phases = [("final 2:00", 0, 120), ("2:00-5:00", 120, 300), ("5:00-10:00", 300, 600),
              ("10:00-30:00", 600, 1800), ("first half", 1800, 3601)]
    gsr = wt["game_seconds_remaining"].to_numpy()
    out["reliability"] = [dict(phase=name, n=int(((gsr >= lo) & (gsr < hi) & close).sum()),
                               mad=float(gap[(gsr >= lo) & (gsr < hi) & close].mean()))
                          for name, lo, hi in phases]
    cal["wp_nflverse"] = E.calibration_table(wt["vegas_wp"].to_numpy(), y, 20, "uniform").to_dict("records")
    log(f"WP holdout Brier {out['wp']['brier']:.4f} (nflverse vegas_wp {out['wp_nflverse_vegas_wp']['brier']:.4f})")

    # -- conversion
    ctr, cte = data.conversion_frame(train), data.conversion_frame(test)
    clim = ctr.assign(k=np.minimum(ctr["ydstogo"], 15)).groupby(["down", "k"])["converted"].mean()
    base = clim.reindex(list(zip(cte["down"], np.minimum(cte["ydstogo"], 15)))).to_numpy()
    p_conv = M.logit_predict(comps["point"]["conv"], comps["conv_scaler"](M.conv_X(cte)))
    # comparator: the same logistic family with only yards to go and the down
    Xs = lambda d: np.column_stack([np.log(d["ydstogo"]), d["is_4th"], np.log(d["ydstogo"]) * d["is_4th"]])
    small = LogisticRegression(C=1.0, max_iter=2000).fit(Xs(ctr), ctr["converted"])
    p_small = small.predict_proba(Xs(cte))[:, 1]
    y = cte["converted"].to_numpy()
    for name, m in (("fourth", cte["down"] == 4), ("third", cte["down"] == 3)):
        m = m.to_numpy()
        out[f"conv_{name}"] = E.summarise(p_conv[m], y[m], cte["game_id"][m], baseline=base[m])
        out[f"conv_{name}_ytg_only"] = E.summarise(p_small[m], y[m], cte["game_id"][m], baseline=base[m],
                                                   n_boot=200)
    m4 = (cte["down"] == 4).to_numpy()
    cal["conv_fourth"] = E.calibration_table(p_conv[m4], y[m4], 8).to_dict("records")
    log(f"conversion holdout (4th downs, n={m4.sum()}): Brier {out['conv_fourth']['brier']:.4f} "
        f"vs climatology {out['conv_fourth']['brier_baseline']:.4f}, ytg-only logistic "
        f"{out['conv_fourth_ytg_only']['brier']:.4f}")

    # -- field goals
    ftr, fte = data.fg_frame(train), data.fg_frame(test)
    fclim = ftr.assign(k=(ftr["distance"] // 5)).groupby("k")["made"].mean()
    fbase = fclim.reindex(fte["distance"] // 5).fillna(ftr["made"].mean()).to_numpy()
    p_fg = M.logit_predict(comps["point"]["fg"], comps["fg_scaler"](M.fg_X(fte)))
    out["fg"] = E.summarise(p_fg, fte["made"], fte["game_id"], baseline=fbase)
    cal["fg"] = E.calibration_table(p_fg, fte["made"].to_numpy(), 8).to_dict("records")
    by_dist = fte.assign(p=p_fg, k=(fte["distance"] // 5) * 5).groupby("k").agg(
        n=("made", "size"), made=("made", "mean"), predicted=("p", "mean")).reset_index()
    cal["fg_by_distance"] = by_dist.to_dict("records")
    log(f"FG holdout Brier {out['fg']['brier']:.4f} vs distance-bin baseline {out['fg']['brier_baseline']:.4f}")

    # -- punts: predicted next-possession distribution vs what happened
    pte = data.punt_frame(test)
    pte = pte[pte["next_owner"] == "opp"]
    opp, own, td = comps["point"]["punt"]
    k = pte["yardline_100"].to_numpy(int) - 1
    pm = opp[k] / opp[k].sum(1, keepdims=True)
    cdf = np.cumsum(pm, axis=1)
    actual = pte["next_yardline"].to_numpy(int)
    pred_mean = (pm * np.arange(100)).sum(1)
    lo = (cdf >= 0.05).argmax(1); hi = (cdf >= 0.95).argmax(1)
    out["punt"] = dict(n=int(len(pte)), mae_mean=float(np.mean(np.abs(pred_mean - actual))),
                       bias=float(np.mean(pred_mean - actual)),
                       coverage_90=float(np.mean((actual >= lo) & (actual <= hi))))
    log(f"punt holdout: bias {out['punt']['bias']:+.2f} yd, 90% interval coverage {out['punt']['coverage_90']:.3f}")

    # -- the whole chain: value of the call actually made vs who won
    dec = data.decision_frame(df, [config.HOLDOUT_SEASON])
    dec = dec[~dec["is_tie"]].reset_index(drop=True)
    s = A.situation_from(dec).arrays(len(dec), engine.c["default_season"])
    V = engine.state_values(s)
    sc = engine.score(s, V, dec["ydstogo"].to_numpy(int), dec["yardline_100"].to_numpy(int))
    ev_actual = sc["ev_all"][np.arange(len(dec)), dec["call"].map(A.CALL_INDEX).to_numpy()]
    out["chain"] = E.summarise(ev_actual, dec["posteam_win"], dec["game_id"], n_boot=300)
    cal["chain"] = E.calibration_table(ev_actual, dec["posteam_win"].to_numpy(), 10).to_dict("records")
    log(f"chain check on {len(dec)} 2025 fourth downs: Brier {out['chain']['brier']:.4f}")
    out["components"] = {k: v for k, v in comps["info"].items() if not isinstance(v, dict)}
    out["components"]["conv_cv_logloss"] = {str(k): v for k, v in comps["info"]["conv_cv"].items()}
    out["components"]["wp_cv_logloss"] = {str(k): v for k, v in comps["info"]["wp_cv"].items()}
    out["components"]["fg_cv_logloss"] = {str(k): v for k, v in comps["info"]["fg_cv"].items()}
    return out, cal


def boot_params(comps):
    return [dict(ep=np.asarray(b["ep"]), wp=np.asarray(b["wp"]), conv=np.asarray(b["conv"]), fg=np.asarray(b["fg"]), extra=b["extra"], punt=b["punt"])
            for b in comps["boots"]]


def run(n_boot=config.N_BOOT, refit=True):
    t0 = time.time()
    df = data.load_pbp(config.ALL_SEASONS)
    log(f"loaded {len(df):,} plays, seasons {df['season'].min()}-{df['season'].max()}")
    if refit or not (CACHE / "models.pkl").exists():
        comps = fit_all(df, n_boot)
    else:
        with open(CACHE / "models.pkl", "rb") as fh:
            comps = pickle.load(fh)["comps"]
    engine = Engine.load()
    metrics, cal = holdout_metrics(df, comps, engine)
    boots = boot_params(comps)
    seasons = {}
    for season in config.AUDIT_SEASONS:
        log(f"auditing {season}")
        audit, teams, coaches, summary = A.run_season(engine, df, season, boots)
        audit.to_csv(config.ARTIFACTS / f"decisions_{season}.csv", index=False)
        teams.to_csv(config.ARTIFACTS / f"teams_{season}.csv", index=False)
        coaches.to_csv(config.ARTIFACTS / f"coaches_{season}.csv", index=False)
        seasons[str(season)] = summary
        log(f"  {summary['fourth_downs']} fourth downs, league WP lost {summary['wp_lost_total']:.2f} "
            f"[{summary['wp_lost_total_ci'][0]:.2f}, {summary['wp_lost_total_ci'][1]:.2f}]")
    metrics["seasons"] = seasons
    metrics["n_boot"] = len(boots)
    with open(config.ARTIFACTS / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    with open(config.ARTIFACTS / "calibration.json", "w") as fh:
        json.dump(cal, fh, default=float)
    from . import charts, site
    log("rendering charts")
    charts.render_all(engine)
    log("building site")
    site.build()
    log(f"done in {time.time() - t0:.0f}s")
