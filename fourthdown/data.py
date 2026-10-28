"""Loading nflverse play-by-play and turning it into the modelling tables.

Every table here is built from the public nflverse `pbp` release files. Conventions:

* ``yardline_100`` is the distance from the offense to the opponent's end zone (1-99).
* ``posteam_spread`` is the pre-game Vegas line from the offense's point of view
  (positive = the offense was favoured by that many points).
* Overtime is excluded everywhere: the clock and possession rules differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id", "posteam", "defteam",
    "home_team", "away_team", "home_coach", "away_coach", "down", "ydstogo",
    "yardline_100", "goal_to_go", "qtr", "quarter_seconds_remaining",
    "half_seconds_remaining", "game_seconds_remaining", "game_half",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "play_type", "first_down", "touchdown", "yards_gained", "field_goal_result",
    "kick_distance", "extra_point_result", "return_touchdown", "punt_blocked",
    "result", "spread_line", "total_line", "roof", "home_opening_kickoff",
    "drive_start_transition", "fixed_drive", "wp", "vegas_wp", "ep", "desc",
    "td_team", "interception", "fumble_lost", "total_home_score", "total_away_score",
]

SCRIMMAGE = ["run", "pass", "punt", "field_goal"]


def load_pbp(seasons, raw_dir=None, columns=COLUMNS) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW_DIR
    frames = []
    for s in seasons:
        path = raw_dir / f"play_by_play_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `make data` first")
        frames.append(pd.read_parquet(path, columns=columns))
    df = pd.concat(frames, ignore_index=True)
    return add_next_score(add_features(df))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns shared by every model. Safe to call on a raw nflverse frame."""
    df = df.copy()
    home = df["posteam"] == df["home_team"]
    df["is_home"] = home.astype(float)
    df["posteam_spread"] = np.where(home, df["spread_line"], -df["spread_line"])
    first_half = df["game_half"] == "Half1"
    home_received_opening = df["home_opening_kickoff"] == 1
    # The team that did not receive the opening kickoff receives the second-half kickoff.
    df["receive_2h_ko"] = (first_half & (home != home_received_opening)).astype(float)
    df["posteam_win"] = np.where(home, df["result"] > 0, df["result"] < 0).astype(float)
    df["is_tie"] = df["result"] == 0
    df["coach"] = np.where(home, df["home_coach"], df["away_coach"])
    df["indoor"] = df["roof"].isin(["dome", "closed"]).astype(float)
    return df


def add_next_score(df: pd.DataFrame) -> pd.DataFrame:
    """``next_score``: points of the next score in the same half, from the offense's view
    (touchdown 7, field goal 3, safety 2, negative when the other team scores, 0 if the half
    ends first). This is the target of the expected-points regression."""
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    g = df.groupby("game_id")
    home = df["total_home_score"].ffill().fillna(0)
    away = df["total_away_score"].ffill().fillna(0)
    dh = (home - g["total_home_score"].shift(1).ffill().fillna(0).where(g.cumcount() > 0, 0)).clip(lower=0)
    da = (away - g["total_away_score"].shift(1).ffill().fillna(0).where(g.cumcount() > 0, 0)).clip(lower=0)
    scoring = (dh > 0) | (da > 0)
    pts_home = np.where(dh == 6, 7, dh) - np.where(da == 6, 7, da)
    half = df["game_half"].astype(str)
    key = df["game_id"].astype(str) + "|" + half
    nxt = pd.Series(np.where(scoring, pts_home, np.nan), index=df.index).groupby(key).bfill()
    nxt = nxt.fillna(0.0)
    df["next_score"] = np.where(df["posteam"] == df["home_team"], nxt, -nxt)
    return df


def _regulation(df: pd.DataFrame) -> pd.Series:
    return (df["qtr"] <= 4) & df["posteam"].notna() & df["down"].notna()


def wp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-snap states with a known final result: the win-probability training set."""
    m = _regulation(df) & df["play_type"].isin(SCRIMMAGE) & ~df["is_tie"]
    m &= df["spread_line"].notna() & df["posteam_timeouts_remaining"].notna()
    return df.loc[m].copy()


def conversion_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Third- and fourth-down run/pass plays. ``converted`` = first down or touchdown.

    Third downs are included (with a fourth-down indicator) because fourth-down attempts
    alone are a selected sample: coaches go for it mostly when they expect to convert.
    """
    m = _regulation(df) & df["down"].isin([3, 4]) & df["play_type"].isin(["run", "pass"])
    m &= df["spread_line"].notna() & df["ydstogo"].between(1, config.MAX_YDSTOGO)
    out = df.loc[m].copy()
    gained = out["yards_gained"].fillna(0)
    out["converted"] = (
        (out["first_down"] == 1) | (out["touchdown"] == 1) & (out["td_team"] == out["posteam"])
        | (gained >= out["ydstogo"])
    ).astype(float)
    # A turnover is never a conversion even when the yardage was reached first.
    out.loc[(out["interception"] == 1) | (out["fumble_lost"] == 1), "converted"] = 0.0
    out["is_4th"] = (out["down"] == 4).astype(float)
    out["extra_yards"] = np.clip(gained - out["ydstogo"], 0, 99)
    return out


def fg_frame(df: pd.DataFrame) -> pd.DataFrame:
    m = _regulation(df) & (df["play_type"] == "field_goal") & df["kick_distance"].notna()
    out = df.loc[m].copy()
    out["made"] = (out["field_goal_result"] == "made").astype(float)
    out["distance"] = out["yardline_100"] + 18
    return out


def punt_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Punts with the state of the next possession attached.

    ``next_owner`` is ``"opp"`` (normal), ``"own"`` (muff recovered by the punting team) or
    ``"opp_td"`` (return touchdown). ``next_yardline`` is the next snap's ``yardline_100``
    from the perspective of whoever owns the ball.
    """
    cols = ["game_id", "play_id", "posteam", "yardline_100", "play_type", "down", "qtr"]
    seq = df.loc[df["posteam"].notna() & df["down"].notna(), cols + ["season", "return_touchdown",
                                                                    "punt_blocked"]]
    seq = seq.sort_values(["game_id", "play_id"])
    g = seq.groupby("game_id")
    seq = seq.assign(next_posteam=g["posteam"].shift(-1), next_yl=g["yardline_100"].shift(-1))
    p = seq[(seq["play_type"] == "punt") & (seq["qtr"] <= 4)].copy()
    p["next_owner"] = np.where(p["return_touchdown"] == 1, "opp_td",
                               np.where(p["next_posteam"] == p["posteam"], "own", "opp"))
    p["next_yardline"] = p["next_yl"]
    p = p[(p["next_owner"] == "opp_td") | p["next_yardline"].notna()]
    return p[["season", "game_id", "play_id", "yardline_100", "next_owner", "next_yardline"]]


def kickoff_start(df: pd.DataFrame, season: int) -> float:
    """Mean yardline_100 of drives that start from a kickoff in a season (rules change it)."""
    d = df[(df["season"] == season) & _regulation(df)]
    first = d.groupby(["game_id", "fixed_drive"]).head(1)
    k = first.loc[first["drive_start_transition"] == "KICKOFF", "yardline_100"]
    return float(k.mean())


def xp_rate(df: pd.DataFrame, seasons) -> float:
    x = df[df["season"].isin(seasons) & (df["play_type"] == "extra_point")]
    return float((x["extra_point_result"] == "good").mean())


def classify_call(play_type: pd.Series) -> pd.Series:
    return play_type.map({"run": "go", "pass": "go", "punt": "punt", "field_goal": "fg"})


def decision_frame(df: pd.DataFrame, seasons) -> pd.DataFrame:
    """Every regulation, regular-season fourth down with a real decision (no penalties
    wiping the play, no kneels or spikes). Fakes count as 'go'."""
    m = _regulation(df) & (df["down"] == 4) & df["season"].isin(seasons)
    m &= (df["season_type"] == "REG") & df["play_type"].isin(SCRIMMAGE)
    m &= df["spread_line"].notna() & df["ydstogo"].between(1, config.MAX_YDSTOGO)
    out = df.loc[m].copy()
    out["call"] = classify_call(out["play_type"])
    return out
