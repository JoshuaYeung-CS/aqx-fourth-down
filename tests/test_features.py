import numpy as np
import pandas as pd

from fourthdown import data, features as F


def test_add_features_perspective(tiny_game):
    df = data.add_features(tiny_game)
    first = df.iloc[0]
    assert first["is_home"] == 0.0                     # AAA is the away team
    assert first["posteam_spread"] == -2.5             # home BBB was favoured by 2.5
    # nflverse `result` = home score minus away score, so -3 means the away team (AAA) won
    assert first["posteam_win"] == 1.0
    assert df.loc[df["posteam"] == "BBB", "posteam_win"].eq(0.0).all()


def test_result_sign_convention():
    df = data.add_features(pd.DataFrame([
        dict(posteam="AAA", home_team="BBB", result=-3, spread_line=0, home_opening_kickoff=1,
             game_half="Half1", home_coach="b", away_coach="a", roof="dome"),
        dict(posteam="BBB", home_team="BBB", result=-3, spread_line=0, home_opening_kickoff=1,
             game_half="Half1", home_coach="b", away_coach="a", roof="dome"),
    ]))
    assert df["posteam_win"].tolist() == [1.0, 0.0]    # away team won by 3
    assert df["coach"].tolist() == ["a", "b"]
    assert df["indoor"].tolist() == [1.0, 1.0]
    # home received the opening kickoff, so the away team gets the second-half kickoff
    assert df["receive_2h_ko"].tolist() == [1.0, 0.0]


def test_conversion_and_next_score(tiny_game):
    df = data.add_next_score(data.add_features(tiny_game))
    conv = data.conversion_frame(df)
    assert conv["down"].tolist() == [3, 4]
    assert conv["converted"].tolist() == [0.0, 1.0]
    assert conv["extra_yards"].tolist() == [0, 4]
    # BBB's touchdown is the next score for every play of AAA's drive: -7 from AAA's view
    aaa = df[(df["posteam"] == "AAA")]
    assert (aaa["next_score"] == -7).all()
    assert df.loc[df["play_id"] == 6, "next_score"].iat[0] == 7


def test_punt_frame_next_possession(tiny_game):
    df = data.add_features(tiny_game)
    p = data.punt_frame(df)
    assert len(p) == 1
    assert p["next_owner"].iat[0] == "opp"
    assert p["next_yardline"].iat[0] == 88


def test_decision_frame_calls(tiny_game):
    df = data.add_features(tiny_game)
    d = data.decision_frame(df, [2024])
    assert d["call"].tolist() == ["go", "punt"]


def test_matrix_shapes_and_finite():
    n = 50
    rng = np.random.default_rng(0)
    yl = rng.integers(1, 100, n); ytg = np.minimum(rng.integers(1, 20, n), yl)
    ep = F.ep_matrix(rng.integers(1, 5, n), ytg, yl, rng.integers(0, 1800, n))
    assert ep.shape == (n, len(F.EP_FEATURES))
    wp = F.wp_matrix(rng.integers(-21, 22, n), rng.integers(0, 3600, n), rng.integers(0, 1800, n),
                     3, 2, rng.integers(0, 2, n), rng.normal(0, 4, n), 0.5, rng.normal(1, 2, n))
    assert wp.shape == (n, len(F.WP_FEATURES))
    conv = F.conv_matrix(ytg, yl, 1, 0, 45, 2025)
    assert conv.shape == (n, len(F.CONV_FEATURES))
    fg = F.fg_matrix(yl + 18, 0, 2025)
    assert fg.shape == (n, len(F.FG_FEATURES))
    for m in (ep, wp, conv, fg):
        assert np.isfinite(m).all()


def test_season_trend_is_frozen():
    assert F.season_c(2026) == F.season_c(2024) == 4.0
    assert F.season_c(2018) == -2.0


def test_signed_second_half_kickoff():
    cols = F.WP_FEATURES.index("rec2h")
    us = F.wp_matrix(0, 3000, 1200, 3, 3, 1, 0, 0.5, 1.0)[0, cols]
    them = F.wp_matrix(0, 3000, 1200, 3, 3, 0, 0, 0.5, 1.0)[0, cols]
    after_half = F.wp_matrix(0, 1500, 1500, 3, 3, 1, 0, 0.5, 1.0)[0, cols]
    assert us == -them > 0
    assert after_half == 0


def test_buckets():
    assert F.ydstogo_bucket([1, 2, 3, 4, 5, 6, 10, 11, 25]).tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 5]
