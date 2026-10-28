"""Static charts for docs/ (PNG, plus a print-ready PDF of the decision chart)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config  # noqa: E402
from .engine import Engine, Situation  # noqa: E402

COLORS = {"go": "#2a78d6", "punt": "#eb6834", "fg": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"go": "Go for it", "punt": "Punt", "fg": "Field goal"}

PRESETS = [
    ("Q1, tied", Situation(score_diff=0, qtr=1, quarter_seconds=900)),
    ("Q4 5:00, down 3", Situation(score_diff=-3, qtr=4, quarter_seconds=300)),
    ("Q4 5:00, up 3", Situation(score_diff=3, qtr=4, quarter_seconds=300)),
    ("Q3 10:00, down 7", Situation(score_diff=-7, qtr=3, quarter_seconds=600)),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _yard_ticks(ax, horizontal=True):
    pos = [99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
    lab = ["own 1", "own 10", "own 20", "own 30", "own 40", "50", "opp 40", "opp 30", "opp 20",
           "opp 10", "opp 1"]
    (ax.set_xticks if horizontal else ax.set_yticks)([100 - p for p in pos])
    (ax.set_xticklabels if horizontal else ax.set_yticklabels)(lab)


def decision_panel(ax, engine: Engine, title: str, sit: Situation, max_ytg=10, tossup=0.005):
    g = engine.grid(sit, max_ytg)
    best = g["best"]
    ev = np.stack([g["go"], g["punt"], g["fg"]], axis=-1)
    srt = -np.sort(-np.nan_to_num(ev, nan=-1), axis=-1)
    margin = srt[..., 0] - srt[..., 1]
    img = np.ones(best.shape + (3,))
    for i, c in enumerate(("go", "punt", "fg")):
        rgb = np.array(to_rgb(COLORS[c]))
        faint = 0.35 * rgb + 0.65 * np.array(to_rgb(SURFACE))
        m = best == i
        img[m] = np.where((margin[m] < tossup)[:, None], faint, rgb)
    img[best < 0] = to_rgb(SURFACE)
    # x axis: field position from own goal line (left) to opponent goal line (right)
    img = img[:, ::-1]
    ax.imshow(img, aspect="auto", origin="upper", extent=[0.5, 99.5, max_ytg + 0.5, 0.5],
              interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    _yard_ticks(ax)
    ax.set_yticks(range(1, max_ytg + 1))
    ax.set_ylabel("yards to go", fontsize=8, color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    for x in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        ax.axvline(x + 0.5, color="white", linewidth=0.8, alpha=0.8)
    return g


def render_decision_chart(engine: Engine, out_png, out_pdf=None):
    fig, axes = plt.subplots(len(PRESETS), 1, figsize=(11, 2.1 * len(PRESETS) + 1.3))
    fig.patch.set_facecolor("white")
    for ax, (title, sit) in zip(axes, PRESETS):
        decision_panel(ax, engine, title, sit)
    handles = [Patch(color=COLORS[c], label=LABEL[c]) for c in ("go", "punt", "fg")]
    handles.append(Patch(color=(0.35 * np.array(to_rgb(COLORS["go"])) + 0.65 * np.array(to_rgb(SURFACE))),
                         label="pale = toss-up (< 0.5 pts of win prob)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Fourth Down Ledger: decision chart (neutral matchup, 3 timeouts each)",
                 x=0.01, y=0.995, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.958, "Each cell is the call with the highest win probability. Offense drives left to right. "
             "Models fit on 2016-2024 nflverse play-by-play.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 0.945])
    fig.savefig(out_png, dpi=160)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)


def render_calibration(cal: dict, metrics: dict, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    panels = [
        ("wp", f"Win probability (Brier {metrics['wp']['brier']:.3f})", "wp_nflverse",
         f"nflverse vegas_wp ({metrics['wp_nflverse_vegas_wp']['brier']:.3f})"),
        ("conv_fourth", f"4th-down conversion (Brier {metrics['conv_fourth']['brier']:.3f})", None, None),
        ("fg", f"Field goal make (Brier {metrics['fg']['brier']:.3f})", None, None),
        ("chain", f"Value of the call made vs win (Brier {metrics['chain']['brier']:.3f})", None, None),
    ]
    for ax, (key, title, ref, ref_label) in zip(axes, panels):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)), label="perfect")
        if ref:
            r = pd.DataFrame(cal[ref])
            ax.plot(r["predicted"], r["observed"], color=COLORS["punt"], linewidth=2, marker="o",
                    markersize=4, label=ref_label)
        t = pd.DataFrame(cal[key])
        ax.errorbar(t["predicted"], t["observed"], yerr=[t["observed"] - t["lo"], t["hi"] - t["observed"]],
                    color=COLORS["go"], linewidth=2, marker="o", markersize=5, capsize=0, label="this project")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("predicted", fontsize=8, color=MUTED)
        ax.set_ylabel("observed (95% CI)", fontsize=8, color=MUTED)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Calibration on the held-out {metrics['holdout_season']} season "
                 f"(models fit on {config.TRAIN_SEASONS[0]}-{config.TRAIN_SEASONS[-1]})",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_fg_curve(engine: Engine, cal: dict, out_png):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    d = np.arange(18, 69)
    s = Situation(0, 1, 900).arrays(len(d), engine.c["default_season"])
    p = engine.p_make(d - 18, s)
    ax.plot(d, p, color=COLORS["go"], linewidth=2, label="model (outdoor)")
    t = pd.DataFrame(cal["fg_by_distance"])
    t = t[t["n"] >= 5]
    ax.scatter(t["k"] + 2.5, t["made"], s=8 + t["n"] / 3, color=COLORS["punt"], zorder=3,
               label="2025 holdout, 5-yd bins (size = attempts)")
    ax.set_xlabel("kick distance (yards)", fontsize=8, color=MUTED)
    ax.set_ylabel("make probability", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.set_title("Field goal make probability by distance", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_punt(engine: Engine, out_png):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    opp = np.asarray(engine.spec["punt"]["opp"])
    for spot, c in ((80, COLORS["go"]), (65, COLORS["punt"]), (50, COLORS["fg"])):
        pmf = opp[spot - 1]
        ax.plot(100 - np.arange(1, 100), pmf[1:], color=c, linewidth=2,
                label=f"punt from {'own ' + str(100 - spot) if spot > 50 else 'midfield'} "
                      f"(mean opp start: own {100 - (pmf * np.arange(100)).sum() / pmf.sum():.0f})")
    ax.set_xlabel("opponent's starting spot (yards from their own goal line)", fontsize=8, color=MUTED)
    ax.set_ylabel("probability", fontsize=8, color=MUTED)
    ax.set_xlim(0, 60)
    ax.set_title("Punt outcome distributions (empirical, 2016-2024)", loc="left", fontsize=10, color=INK,
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_teams(teams: pd.DataFrame, season: int, out_png):
    t = teams.sort_values("wp_lost")
    fig, ax = plt.subplots(figsize=(8, 9))
    _style(ax)
    y = np.arange(len(t))
    ax.barh(y, t["wp_lost"], color=COLORS["go"], height=0.6)
    ax.errorbar(t["wp_lost"], y, xerr=[t["wp_lost"] - t["wp_lost_lo"], t["wp_lost_hi"] - t["wp_lost"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=0)
    ax.set_yticks(y); ax.set_yticklabels(t["posteam"], fontsize=8)
    ax.set_xlabel("expected wins left on the table (sum of win probability lost), 95% CI", fontsize=8, color=MUTED)
    ax.set_title(f"{season}: win probability lost to fourth-down calls, by team", loc="left", fontsize=10,
                 color=INK, fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_aggr(teams: pd.DataFrame, season: int, out_png):
    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.scatter(teams["aggressiveness"], teams["wp_lost"], s=40, color=COLORS["go"], edgecolor=SURFACE,
               linewidth=2, zorder=3)
    for _, r in teams.iterrows():
        ax.annotate(r["posteam"], (r["aggressiveness"], r["wp_lost"]), fontsize=7, color=MUTED,
                    xytext=(4, 2), textcoords="offset points")
    rho = teams[["aggressiveness", "wp_lost"]].corr(method="spearman").iat[0, 1]
    ax.set_xlabel("aggressiveness index: go rate over league expectation (percentage points)",
                  fontsize=8, color=MUTED)
    ax.set_ylabel("expected wins left on the table", fontsize=8, color=MUTED)
    ax.set_title(f"{season}: aggressiveness vs win probability left on the table (Spearman rho {rho:.2f})",
                 loc="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_all(engine: Engine):
    config.DOCS.mkdir(exist_ok=True)
    metrics = json.loads((config.ARTIFACTS / "metrics.json").read_text())
    cal = json.loads((config.ARTIFACTS / "calibration.json").read_text())
    render_decision_chart(engine, config.DOCS / "decision_chart.png", config.DOCS / "decision_chart.pdf")
    render_calibration(cal, metrics, config.DOCS / "calibration.png")
    render_fg_curve(engine, cal, config.DOCS / "fg_curve.png")
    render_punt(engine, config.DOCS / "punt_distribution.png")
    season = config.HOLDOUT_SEASON
    teams = pd.read_csv(config.ARTIFACTS / f"teams_{season}.csv")
    render_teams(teams, season, config.DOCS / f"team_wp_lost_{season}.png")
    render_aggr(teams, season, config.DOCS / f"aggressiveness_{season}.png")
