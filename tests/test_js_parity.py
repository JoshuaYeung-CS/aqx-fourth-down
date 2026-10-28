"""The browser engine (site_src/engine.js) must give the same numbers as Python."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from fourthdown import config
from fourthdown.engine import Situation

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_matches_python(engine):
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(25):
        yl = int(rng.integers(1, 100)); ytg = int(min(rng.integers(1, 16), yl))
        cases.append(dict(scoreDiff=int(rng.integers(-17, 18)), qtr=int(rng.integers(1, 5)),
                          quarterSeconds=int(rng.integers(0, 901)), timeouts=int(rng.integers(0, 4)),
                          oppTimeouts=int(rng.integers(0, 4)), spread=float(rng.normal(0, 4)),
                          indoor=bool(rng.integers(0, 2)), receive2h=bool(rng.integers(0, 2)),
                          ytg=ytg, yl=yl))
    # clock-expiry cases: end of game and end of first half during the play
    for qtr, secs, diff in ((4, 3, 0), (4, 20, -3), (2, 2, 0), (2, 20, 4), (4, 0, 1)):
        cases.append(dict(scoreDiff=diff, qtr=qtr, quarterSeconds=secs, timeouts=1, oppTimeouts=2,
                          spread=0.0, indoor=False, receive2h=qtr == 2, ytg=3, yl=30))
    script = """
const {Engine} = require(process.argv[1]);
const fs = require('fs');
const e = new Engine(JSON.parse(fs.readFileSync(process.argv[2])));
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => e.recommend(c, c.ytg, c.yl))));
"""
    out = subprocess.run([NODE, "-e", script, str(config.SITE_SRC / "engine.js"), str(config.ENGINE_PATH)],
                         input=json.dumps(cases), capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    for c, j in zip(cases, js):
        sit = Situation(score_diff=c["scoreDiff"], qtr=c["qtr"], quarter_seconds=c["quarterSeconds"],
                        posteam_timeouts=c["timeouts"], defteam_timeouts=c["oppTimeouts"],
                        spread=c["spread"], indoor=int(c["indoor"]), receive_2h_ko=int(c["receive2h"]))
        p = engine.recommend(sit, c["ytg"], c["yl"])
        assert p["recommendation"] == j["recommendation"], c
        for k in ("go", "punt", "fg"):
            if p["wp"][k] is None:
                assert j["wp"][k] is None
            else:
                assert abs(p["wp"][k] - j["wp"][k]) < 1e-9, (c, k)
        assert abs(p["p_convert"] - j["p_convert"]) < 1e-9
        assert abs(p["agreement"] - j["agreement"]) < 1e-9
