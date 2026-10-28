"""Feature construction shared by fitting (pandas) and scoring (numpy arrays).

``site_src/engine.js`` re-implements these exact formulas; ``tests/test_js_parity.py`` checks it.
Every model in the project is a (regularised) logistic regression on these designs.
"""

from __future__ import annotations

import numpy as np

YL_KNOTS = (10, 20, 35, 50, 65, 80, 90)
EP_KNOTS = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95)


def _hinges(yardline, knots=YL_KNOTS):
    """Linear regression spline (truncated-power basis) in field position."""
    yl = np.asarray(yardline, float)
    return [yl / 100.0] + [np.maximum(0.0, yl - k) / 100.0 for k in knots]


EP_FEATURES = (
    ["const"] + [f"yl_h{i}" for i in range(len(EP_KNOTS) + 1)]
    + ["down2", "down3", "down4", "down2_yl", "down3_yl", "down4_yl", "down2_yl60", "down3_yl60",
       "down4_yl60", "log_ytg", "down2_log_ytg", "down3_log_ytg", "down4_log_ytg", "goal_to_go",
       "goal_to_go_yl", "end_half", "end_half_yl", "end_half_fg_range"]
)


def ep_matrix(down, ydstogo, yardline, hsr):
    """Design of the expected-points regression: E[next score in the half | state].

    Field position enters through a 12-knot linear spline, the down shifts and bends it,
    yards to go enters on the log scale by down, and an end-of-half decay shrinks EP toward
    a field-goal-or-nothing value as the half runs out."""
    arrs = np.broadcast_arrays(*[np.asarray(a, float) for a in (down, ydstogo, yardline, hsr)])
    down, ytg, yl, hsr = [a.ravel() for a in arrs]
    d = [(down == k) * 1.0 for k in (2, 3, 4)]
    lg = np.log(np.maximum(ytg, 1.0))
    gtg = (ytg >= yl) * 1.0
    e = np.exp(-hsr / 90.0)
    cols = ([np.ones_like(yl)] + _hinges(yl, EP_KNOTS) + d + [x * yl / 100.0 for x in d]
            + [x * np.maximum(0.0, yl - 60) / 100.0 for x in d] + [lg] + [x * lg for x in d]
            + [gtg, gtg * yl / 100.0, e, e * yl / 100.0, e * (yl <= 35)])
    return np.column_stack(cols)


WP_FEATURES = [
    "diff", "diff_r", "ep", "ep_r", "spread_t", "spread_t_r", "rec2h", "rec2h_sq", "home", "pto",
    "dto", "to_diff_r", "lead_late", "tie_late", "trail3_late", "trail8_late", "ep_late",
    "diff_late", "pto_late", "dto_late", "ep_late10", "diff_late10", "lead_late10", "tie_late10",
    "trail8_late10", "pto_late10", "dto_late10", "ep_r_late10",
] + [f"bin{b}_{term}" for b in range(7) for term in ("late10", "ep_late10", "late", "ep_late",
                                                      "pto_late10", "dto_late10")]
# score-margin bins for the late-game interactions: <=-9, -8..-4, -3..-1, 0, 1..3, 4..8, >=9
MARGIN_EDGES = (-8.5, -3.5, -0.5, 0.5, 3.5, 8.5)


def wp_matrix(score_diff, gsr, hsr, pto, dto, rec2h, spread, home, ep):
    """Design of the logistic win-probability model.

    The state enters through two numbers a coach would recognise: the current margin and the
    expected points of the possession (from ``ep_matrix``). Each appears on its own and divided
    by the square root of the time left (r = 1/sqrt(t + 0.01), t = share of the game remaining),
    so the same margin matters more late. The Vegas line decays with the clock. The second-half
    kickoff enters signed (+1 we receive it, -1 they do, 0 after halftime) with a free decay.
    Late-game terms on two time scales (2.5 and 10 minutes) let a lead with little time left
    behave like a kneel-down and let possession matter less once the clock is short. Within
    seven score-margin bins the late value of expected points and timeouts is free, because
    late in a tied game any score wins while late in a two-score game a field goal barely helps.
    """
    arrs = np.broadcast_arrays(*[np.asarray(a, float) for a in
                                 (score_diff, gsr, hsr, pto, dto, rec2h, spread, home, ep)])
    diff, gsr, hsr, pto, dto, rec2h, spread, home, ep = [a.ravel() for a in arrs]
    t = gsr / 3600.0
    r = 1.0 / np.sqrt(t + 0.01)
    late = np.exp(-gsr / 150.0)
    late10 = np.exp(-gsr / 600.0)
    frac = (gsr > 1800) * hsr / 1800.0
    recs = (2 * rec2h - 1) * frac
    cols = [diff, diff * r, ep, ep * r, spread * t, spread * t * r, recs, recs * frac, home, pto,
            dto, (pto - dto) * r, (diff > 0) * late, (diff == 0) * late,
            ((diff < 0) & (diff >= -3)) * late, ((diff < -3) & (diff >= -8)) * late, ep * late,
            np.clip(diff, -17, 17) * late, pto * late, dto * late,
            ep * late10, np.clip(diff, -17, 17) * late10, (diff > 0) * late10, (diff == 0) * late10,
            ((diff < 0) & (diff >= -8)) * late10, pto * late10, dto * late10, ep * r * late10]
    bins = np.searchsorted(np.asarray(MARGIN_EDGES), diff)
    for k in range(7):
        ind = (bins == k) * 1.0
        cols += [ind * late10, ind * ep * late10, ind * late, ind * ep * late, ind * pto * late10,
                 ind * dto * late10]
    return np.column_stack(cols)


LAST_TRAIN_SEASON = 2024


def season_c(season):
    """Season trend term, centred on 2020 and frozen at the last training season so the
    models never extrapolate a trend into seasons they have not seen."""
    return np.minimum(np.asarray(season, float), LAST_TRAIN_SEASON) - 2020.0


CONV_FEATURES = [
    "log_ydstogo", "ydstogo", "yl", "yl_sq", "goal_to_go", "is_4th", "is_4th_x_log_ydstogo",
    "inside_5", "posteam_spread", "total_line", "season_c",
]


def conv_matrix(ydstogo, yardline, is_4th, spread, total, season):
    ydstogo = np.asarray(ydstogo, float)
    yardline = np.asarray(yardline, float)
    is_4th = np.asarray(is_4th, float)
    n = np.broadcast(ydstogo, yardline, is_4th, spread, total, season).shape
    lg = np.log(ydstogo)
    cols = [lg, ydstogo, yardline / 100.0, (yardline / 100.0) ** 2,
            (ydstogo >= yardline).astype(float), is_4th, is_4th * lg,
            (yardline <= 5).astype(float), spread, total, season_c(season)]
    return np.column_stack([np.broadcast_to(np.asarray(c, float), n).ravel() for c in cols])


FG_FEATURES = ["distance", "distance_over_50", "distance_over_58", "indoor", "season_c",
               "distance_x_season"]


def fg_matrix(distance, indoor, season):
    distance = np.asarray(distance, float)
    n = np.broadcast(distance, indoor, season).shape
    s = season_c(season)
    # linear spline in the logit with knots at 50 and 58 yards: long kicks fall off faster
    cols = [distance, np.maximum(0.0, distance - 50) / 10.0, np.maximum(0.0, distance - 58) / 10.0,
            indoor, s, distance * s / 10.0]
    return np.column_stack([np.broadcast_to(np.asarray(c, float), n).ravel() for c in cols])


def ydstogo_bucket(ydstogo):
    """Buckets for the yards-beyond-the-marker distribution on a successful conversion."""
    y = np.asarray(ydstogo)
    return np.select([y <= 1, y <= 2, y <= 3, y <= 5, y <= 10], [0, 1, 2, 3, 4], default=5)


N_BUCKETS = 6
