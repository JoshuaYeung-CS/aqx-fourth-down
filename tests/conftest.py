import json

import numpy as np
import pandas as pd
import pytest

from fourthdown import config
from fourthdown.engine import Engine


@pytest.fixture(scope="session")
def engine():
    if not config.ENGINE_PATH.exists():
        pytest.skip("artifacts/engine.json missing; run `make pipeline`")
    return Engine.load()


@pytest.fixture(scope="session")
def metrics():
    p = config.ARTIFACTS / "metrics.json"
    if not p.exists():
        pytest.skip("artifacts/metrics.json missing; run `make pipeline`")
    return json.loads(p.read_text())


def _play(**kw):
    base = dict(season=2024, season_type="REG", week=1, game_id="2024_01_AAA_BBB", play_id=1,
                posteam="AAA", defteam="BBB", home_team="BBB", away_team="AAA",
                home_coach="Coach B", away_coach="Coach A", down=1.0, ydstogo=10.0,
                yardline_100=75.0, goal_to_go=0.0, qtr=1.0, quarter_seconds_remaining=900.0,
                half_seconds_remaining=1800.0, game_seconds_remaining=3600.0, game_half="Half1",
                score_differential=0.0, posteam_timeouts_remaining=3.0,
                defteam_timeouts_remaining=3.0, play_type="run", first_down=0.0, touchdown=0.0,
                yards_gained=3.0, field_goal_result=None, kick_distance=np.nan,
                extra_point_result=None, return_touchdown=0.0, punt_blocked=0.0, result=-3.0,
                spread_line=2.5, total_line=44.5, roof="outdoors", home_opening_kickoff=1.0,
                drive_start_transition="KICKOFF", fixed_drive=1.0, wp=0.5, vegas_wp=0.5, ep=1.0,
                desc="", td_team=None, interception=0.0, fumble_lost=0.0,
                total_home_score=0.0, total_away_score=0.0)
    base.update(kw)
    return base


@pytest.fixture
def tiny_game():
    """A hand-built five-play drive: AAA (away) drives, converts a 4th down, then punts."""
    rows = [
        _play(play_id=1, down=1, ydstogo=10, yardline_100=75, yards_gained=4),
        _play(play_id=2, down=2, ydstogo=6, yardline_100=71, yards_gained=1),
        _play(play_id=3, down=3, ydstogo=5, yardline_100=70, yards_gained=2, play_type="pass"),
        _play(play_id=4, down=4, ydstogo=3, yardline_100=68, yards_gained=7, first_down=1.0),
        _play(play_id=5, down=4, ydstogo=8, yardline_100=55, play_type="punt", kick_distance=45,
              yards_gained=0),
        _play(play_id=6, posteam="BBB", defteam="AAA", down=1, ydstogo=10, yardline_100=88,
              score_differential=0, yards_gained=88, touchdown=1.0, td_team="BBB",
              total_home_score=6.0),
        _play(play_id=7, posteam="BBB", defteam="AAA", down=np.nan, play_type="extra_point",
              extra_point_result="good", total_home_score=7.0),
    ]
    return pd.DataFrame(rows)
