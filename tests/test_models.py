import numpy as np
import pandas as pd

from fourthdown import config, models as M


def test_logit_fit_recovers_signal():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.3 + 1.5 * Z[:, 0])))).astype(float)
    p = M.fit_logit(Z, y, None, 10.0)
    assert len(p) == 3
    assert abs(p[1] - 1.5) < 0.2 and abs(p[2]) < 0.15


def test_weighted_ep_fit_matches_duplication():
    rng = np.random.default_rng(2)
    A = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = A @ [1.0, 2.0] + rng.normal(size=200)
    w = rng.integers(0, 3, 200).astype(float)
    dup = np.repeat(np.arange(200), w.astype(int))
    assert np.allclose(M.fit_ep(A, y, w), M.fit_ep(A[dup], y[dup]), atol=1e-6)


def test_empirical_distributions_sum_to_one():
    conv = pd.DataFrame(dict(converted=[1, 1, 0, 1], ydstogo=[1, 5, 3, 12], extra_yards=[0, 7, 0, 30]))
    pmf = M.extra_yards_pmf(conv)
    assert pmf.shape == (6, 100) and np.allclose(pmf.sum(1), 1)
    punts = pd.DataFrame(dict(yardline_100=[70, 71, 69, 60] * 15,
                              next_owner=["opp", "opp", "own", "opp_td"] * 15,
                              next_yardline=[80, 75, 40, np.nan] * 15))
    opp, own, td = M.punt_pmfs(punts)
    assert opp.shape == (99, 100) and np.allclose(opp.sum(1) + own.sum(1) + td, 1)


def test_engine_spec_shape(engine):
    s = engine.spec
    assert len(s["wp"]["params"]) == len(s["wp"]["features"]) + 1
    assert len(s["ep"]["params"]) == len(s["ep"]["features"])
    assert len(s["conv"]["params"]) == len(s["conv"]["features"]) + 1
    assert len(s["fg"]["params"]) == len(s["fg"]["features"]) + 1
    assert len(s["wp"]["boot"]) == len(s["conv"]["boot"]) == len(s["fg"]["boot"]) >= 20
    assert np.allclose(np.asarray(s["extra"]["pmf"]).sum(1), 1, atol=1e-4)
    assert s["train_seasons"] == config.TRAIN_SEASONS


def test_holdout_calibration_sanity(metrics):
    """Out-of-sample scores beat simple baselines and probabilities are neither too bold nor
    too timid."""
    assert metrics["holdout_season"] not in metrics["train_seasons"]
    for k in ("conv_fourth", "fg"):
        assert metrics[k]["brier"] < metrics[k]["brier_baseline"]
        assert 0.8 <= metrics[k]["calibration_slope"] <= 1.2
        assert metrics[k]["ece"] < 0.05
    assert metrics["wp"]["brier"] < 0.17
    assert abs(metrics["wp"]["brier"] - metrics["wp_nflverse_vegas_wp"]["brier"]) < 0.003
    assert 0.8 <= metrics["wp"]["calibration_slope"] <= 1.2
    assert 0.85 <= metrics["punt"]["coverage_90"] <= 0.95
    assert metrics["ep_corr_nflverse"] > 0.95
