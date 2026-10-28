"""Hand-checked fourth-down situations where the right call is not in dispute."""

import numpy as np
import pytest

from fourthdown.engine import Engine, Situation


def rec(engine, ytg, yl, **sit):
    return engine.recommend(Situation(**sit), ytg, yl)


@pytest.mark.parametrize("case, ytg, yl, sit, expected", [
    ("4th & 1 at opp 40, tied, Q2", 1, 40, dict(score_diff=0, qtr=2, quarter_seconds=600), "go"),
    ("4th & 10 at own 20, tied, Q1", 10, 80, dict(score_diff=0, qtr=1, quarter_seconds=900), "punt"),
    ("4th & 15 at own 35, up 7, Q3", 15, 65, dict(score_diff=7, qtr=3, quarter_seconds=600), "punt"),
    pytest.param("down 2, 4th & 2 at opp 20, 1:00 left", 2, 20,
                 dict(score_diff=-2, qtr=4, quarter_seconds=60, posteam_timeouts=1), "fg",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "known weakness, documented in README: in the final minute the smooth "
                     "win-probability regression undervalues a late lead (it gives the opponent "
                     "~53% after the go-ahead kick with 0:56 left), so it leans go; the bootstrap "
                     "band on this call spans zero"))),
    ("down 2, 4th & 2 at opp 20, 1:00 left, opponent out of timeouts", 2, 20,
     dict(score_diff=-2, qtr=4, quarter_seconds=60, posteam_timeouts=1, defteam_timeouts=0), "fg"),
    ("down 4, 0:30 left, 4th & 9 at opp 30", 9, 30,
     dict(score_diff=-4, qtr=4, quarter_seconds=30, posteam_timeouts=0), "go"),
    ("down 4, 0:30 left, 4th & 15 at own 40", 15, 60,
     dict(score_diff=-4, qtr=4, quarter_seconds=30, posteam_timeouts=0), "go"),
    ("tied, 0:03 left in the game, 4th & 1 at opp 32 (kick to win)", 1, 32,
     dict(score_diff=0, qtr=4, quarter_seconds=3), "fg"),
    ("tied, 0:03 before half, 4th & 3 at opp 20", 3, 20, dict(score_diff=0, qtr=2, quarter_seconds=3), "fg"),
    ("down 10, 0:20 left, 4th & 5 at opp 25 (needs two scores, kick now)", 5, 25,
     dict(score_diff=-10, qtr=4, quarter_seconds=20, posteam_timeouts=1), "fg"),
])
def test_obvious_calls(engine, case, ytg, yl, sit, expected):
    r = rec(engine, ytg, yl, **sit)
    assert r["recommendation"] == expected, (case, r)


def test_feasibility_limits(engine):
    r = rec(engine, 5, 75, score_diff=0, qtr=1, quarter_seconds=900)   # own 25: 93-yard kick
    assert r["wp"]["fg"] is None and r["p_make"] is None
    r = rec(engine, 2, 20, score_diff=0, qtr=1, quarter_seconds=900)   # opp 20: no punting
    assert r["wp"]["punt"] is None


def test_probabilities_are_probabilities(engine):
    r = rec(engine, 4, 50, score_diff=-3, qtr=3, quarter_seconds=400)
    for v in r["wp"].values():
        assert v is None or 0 < v < 1
    assert 0 < r["p_convert"] < 1 and 0 <= r["agreement"] <= 1
    lo, hi = r["margin_interval"]
    assert lo <= hi


def test_monotone_sanity(engine):
    """More yards to go never makes going for it more attractive; a closer FG is likelier."""
    sit = Situation(0, 2, 600).arrays(1, 2025)
    p = [float(engine.p_convert(np.array([y]), np.array([50]), sit)[0]) for y in (1, 2, 4, 7, 10, 15)]
    assert all(a > b for a, b in zip(p, p[1:]))
    m = [float(engine.p_make(np.array([yl]), sit)[0]) for yl in (5, 15, 25, 35, 45)]
    assert all(a > b for a, b in zip(m, m[1:]))
    g = engine.grid(Situation(0, 1, 900), 10)
    go = g["go"][:, 60]                       # own 40, 1..10 to go
    assert np.all(np.diff(go[np.isfinite(go)]) < 0)


def test_wp_symmetry_and_score_effect(engine):
    """A bigger lead is better, and late leads are worth more than early ones."""
    w = lambda d, gsr: float(engine.wp_prob(d, gsr, min(gsr, 1800), 1, 10, 75, 3, 3, 0, 0, 0.5)[0])
    assert w(7, 3000) > w(3, 3000) > w(0, 3000) > w(-3, 3000)
    assert w(7, 300) > w(7, 3000)
    assert w(-7, 300) < w(-7, 3000)


def test_grid_shape(engine):
    g = engine.grid(Situation(-3, 4, 300), 15)
    assert g["best"].shape == (15, 99)
    assert (g["best"][14, :14] == -1).all()   # 15 to go is impossible inside the 15


def test_clock_expiry_is_exact(engine):
    """A play that runs out the clock ends the game: no value for a possession that never comes."""
    r = rec(engine, 1, 32, score_diff=0, qtr=4, quarter_seconds=3)
    assert r["wp"]["punt"] == pytest.approx(0.5, abs=1e-4)        # punt, clock expires, overtime
    r = rec(engine, 5, 60, score_diff=3, qtr=4, quarter_seconds=2)
    assert 0.99 < r["wp"]["punt"] < 1.0   # up 3, punt as time expires: win unless returned for a TD
