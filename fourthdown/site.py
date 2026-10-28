"""Build the static, self-contained dashboard at site/index.html.

Everything is inlined (plotly.js from the installed Python package, the engine, the data),
so the page works offline and can be served by GitHub Pages as a single file.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

PLAY_COLS = ["season", "week", "game_id", "posteam", "defteam", "coach", "qtr",
             "quarter_seconds_remaining", "score_differential", "ydstogo", "yardline_100", "call",
             "recommended", "wp_go", "wp_punt", "wp_fg", "p_convert", "wp_lost", "wp_lost_lo",
             "wp_lost_hi", "agreement", "margin", "late", "desc"]


def _records(df: pd.DataFrame, digits=4):
    out = df.copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(digits)
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


def site_data() -> dict:
    metrics = json.loads((config.ARTIFACTS / "metrics.json").read_text())
    seasons = {}
    for s in config.AUDIT_SEASONS:
        plays = pd.read_csv(config.ARTIFACTS / f"decisions_{s}.csv")
        plays["desc"] = plays["desc"].fillna("").str.slice(0, 160)
        seasons[str(s)] = {
            "summary": metrics["seasons"][str(s)],
            "teams": _records(pd.read_csv(config.ARTIFACTS / f"teams_{s}.csv")),
            "coaches": _records(pd.read_csv(config.ARTIFACTS / f"coaches_{s}.csv")),
            "plays": _records(plays[PLAY_COLS]),
        }
    keep = ["wp", "wp_nflverse_vegas_wp", "conv_fourth", "conv_fourth_ytg_only", "conv_third", "fg",
            "punt", "chain", "reliability", "ep_rmse", "ep_corr_nflverse", "holdout_season", "train_seasons", "n_boot"]
    return {"seasons": seasons, "metrics": {k: metrics[k] for k in keep if k in metrics},
            "default_season": str(config.AUDIT_SEASONS[0])}


def _inline_js(code: str) -> str:
    return code.replace("</script", "<\\/script")


def build(out_path=None) -> str:
    import plotly.offline
    out_path = out_path or (config.SITE_DIR / "index.html")
    src = config.SITE_SRC
    template = (src / "template.html").read_text()
    engine_spec = json.loads(config.ENGINE_PATH.read_text())
    data = site_data()
    payload = ("window.ENGINE_SPEC=" + json.dumps(engine_spec, separators=(",", ":"))
               + ";\nwindow.LEDGER=" + json.dumps(data, separators=(",", ":"), default=float) + ";")
    html = (template
            .replace("/*__STYLE__*/", (src / "style.css").read_text())
            .replace("/*__PLOTLY__*/", _inline_js(plotly.offline.get_plotlyjs()))
            .replace("/*__ENGINE__*/", _inline_js((src / "engine.js").read_text()))
            .replace("/*__DATA__*/", _inline_js(payload))
            .replace("/*__APP__*/", _inline_js((src / "app.js").read_text())))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    (out_path.parent / ".nojekyll").write_text("")
    return str(out_path)
