"""Paths, season windows and modelling constants in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
SITE_DIR = ROOT / "site"
SITE_SRC = ROOT / "site_src"
ENGINE_PATH = ARTIFACTS / "engine.json"

FIRST_SEASON = 2016
TRAIN_SEASONS = list(range(2016, 2025))   # every model is fit on these
HOLDOUT_SEASON = 2025                      # never seen in fitting; used for evaluation
AUDIT_SEASONS = [2025, 2026]               # decisions audited (2026 = season to date)
ALL_SEASONS = list(range(2016, 2027))

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

# Game clock consumed by each branch, in seconds, measured from 2016-2024 medians and rounded.
# A converted attempt keeps the clock running to the next snap; a failed one stops it.
CLOCK_GO_SUCCESS = 25
CLOCK_GO_FAIL = 6
CLOCK_PUNT = 9
CLOCK_FG = 4

# Feasibility limits for the recommendation (no model is asked to extrapolate wildly).
MAX_FG_DISTANCE = 66          # kick distance in yards (yardline_100 + 18); longer is not credible
MIN_PUNT_YARDLINE = 30        # no punts from inside the opponent 30
MAX_YDSTOGO = 30

# Bootstrap replicates for the play-level audit (game-clustered) and for the calculator.
N_BOOT = 200
N_BOOT_CALC = 100
SEED = 2026

PUNT_WINDOW = 3               # yards either side of the punt spot pooled for the punt distribution
