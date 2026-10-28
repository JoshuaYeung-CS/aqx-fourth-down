"""The decision engine: price GO, PUNT and FIELD GOAL in win probability.

For a fourth down the engine builds the win probability of every state each call can lead
to, then weights those states by the model probabilities:

* GO      p(convert) * E[WP | first down, yards beyond the marker drawn from the empirical
          distribution; touchdown if the gain reaches the end zone]
          + (1 - p) * WP(opponent ball at the spot)
* FG      p(make) * WP(opponent receives kickoff, +3) + (1 - p) * WP(opponent ball at the
          spot of the kick, or its 20 if the kick was inside the 20)
* PUNT    sum over the empirical next-possession distribution for that punt spot
          (opponent ball at y, muff recovered at y, or return touchdown)

Every possession change flips perspective (WP_us = 1 - WP_them). If the clock runs out during
the play, the resulting state is replaced by its exact value: the final margin decides the game
(a tie goes to overtime at 0.5), and at the end of the first half the value is that of the
second-half kickoff. Touchdowns are worth 7
with the league extra-point rate and 6 otherwise. The engine is pure numpy and reads the
JSON spec produced by ``models.engine_spec``; ``site_src/engine.js`` is a line-for-line port.
All four components are logistic regressions or empirical distributions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, features as F

CALLS = ("go", "punt", "fg")


@dataclass
class Situation:
    """Game state at the fourth-down snap, from the offense's point of view. Arrays or scalars."""
    score_diff: np.ndarray
    qtr: np.ndarray
    quarter_seconds: np.ndarray
    posteam_timeouts: np.ndarray = 3
    defteam_timeouts: np.ndarray = 3
    receive_2h_ko: np.ndarray = 0
    spread: np.ndarray = 0.0          # pre-game line, positive = offense favoured
    total: np.ndarray = 45.0
    is_home: np.ndarray = 0.5
    season: np.ndarray = None
    indoor: np.ndarray = 0.0

    def arrays(self, n: int, default_season: int) -> dict:
        season = default_season if self.season is None else self.season
        out = {}
        for k in ("score_diff", "qtr", "quarter_seconds", "posteam_timeouts", "defteam_timeouts",
                  "receive_2h_ko", "spread", "total", "is_home", "indoor"):
            out[k] = np.broadcast_to(np.asarray(getattr(self, k), float), (n,)).copy()
        out["season"] = np.broadcast_to(np.asarray(season, float), (n,)).copy()
        q = out["qtr"]
        out["gsr"] = (4 - q) * 900 + out["quarter_seconds"]
        out["hsr"] = np.where(q <= 2, (2 - q) * 900, (4 - q) * 900) + out["quarter_seconds"]
        out["first_half"] = (q <= 2).astype(float)
        out["receive_2h_ko"] = out["receive_2h_ko"] * out["first_half"]
        return out


class Engine:
    def __init__(self, spec: dict):
        self.spec = spec
        self.c = spec["constants"]
        self.wp_mean = np.asarray(spec["wp"]["mean"]); self.wp_scale = np.asarray(spec["wp"]["scale"])
        self.conv_mean = np.asarray(spec["conv"]["mean"]); self.conv_scale = np.asarray(spec["conv"]["scale"])
        self.fg_mean = np.asarray(spec["fg"]["mean"]); self.fg_scale = np.asarray(spec["fg"]["scale"])
        self.point = dict(ep=np.asarray(spec["ep"]["params"]), wp=np.asarray(spec["wp"]["params"]), conv=np.asarray(spec["conv"]["params"]), fg=np.asarray(spec["fg"]["params"]),
                          extra=np.asarray(spec["extra"]["pmf"]),
                          punt=(np.asarray(spec["punt"]["opp"]), np.asarray(spec["punt"]["own"]),
                                np.asarray(spec["punt"]["td"])))

    @classmethod
    def load(cls, path: Path | str = config.ENGINE_PATH) -> "Engine":
        with open(path) as fh:
            return cls(json.load(fh))

    # -------------------------------------------------------------- building blocks
    def kickoff_yl(self, season):
        ko = self.c["kickoff_start"]
        latest = ko[max(ko, key=int)]
        return np.array([ko.get(str(int(s)), latest) for s in np.atleast_1d(season)])

    def expected_points(self, down, ydstogo, yardline, hsr, params=None):
        coef = self.point["ep"] if params is None else np.asarray(params["ep"])
        return F.ep_matrix(down, ydstogo, yardline, hsr) @ coef

    def wp_prob(self, score_diff, gsr, hsr, down, ydstogo, yardline, pto, dto, rec2h, spread, home,
                params=None):
        """Win probability of the team with the ball. ``params`` = {"ep": ..., "wp": ...}."""
        wp = self.point["wp"] if params is None else np.asarray(params["wp"])
        shape = np.broadcast(*[np.asarray(a) for a in (score_diff, gsr, hsr, down, ydstogo, yardline,
                                                        pto, dto, rec2h, spread, home)]).shape
        b = lambda a: np.broadcast_to(np.asarray(a, float), shape)
        ep = self.expected_points(b(down), b(ydstogo), b(yardline), b(hsr), params).reshape(shape)
        X = F.wp_matrix(b(score_diff), b(gsr), b(hsr), b(pto), b(dto), b(rec2h), b(spread), b(home), ep)
        z = (X - self.wp_mean) / self.wp_scale
        return 1.0 / (1.0 + np.exp(-(wp[0] + z @ wp[1:])))

    def p_convert(self, ydstogo, yardline, s, params=None):
        params = self.point["conv"] if params is None else np.asarray(params)
        X = F.conv_matrix(ydstogo, yardline, 1.0, s["spread"], s["total"], s["season"])
        z = (X - self.conv_mean) / self.conv_scale
        return 1.0 / (1.0 + np.exp(-(params[0] + z @ params[1:])))

    def p_make(self, yardline, s, params=None):
        params = self.point["fg"] if params is None else np.asarray(params)
        X = F.fg_matrix(np.asarray(yardline, float) + 18.0, s["indoor"], s["season"])
        z = (X - self.fg_mean) / self.fg_scale
        return 1.0 / (1.0 + np.exp(-(params[0] + z @ params[1:])))

    def state_values(self, s: dict, wp_params=None) -> dict:
        """``wp_params``: optional {"ep": ..., "wp": ...} replicate; default = point fit."""
        """Win probability (offense's view) of every state a fourth-down call can lead to.

        Arrays are (n, 100) indexed by the new line of scrimmage's yardline_100 for whoever
        has the ball; column 0 of ``own_succ`` holds the touchdown value."""
        n = len(s["score_diff"])
        y = np.arange(1, 100, dtype=float)
        col = lambda a: np.asarray(a, float)[:, None]
        rec_opp = s["first_half"] * (1 - s["receive_2h_ko"])
        xp = self.c["xp_rate"]
        ko = self.kickoff_yl(s["season"])
        clock = self.c["clock"]
        diff, pto, dto = s["score_diff"], s["posteam_timeouts"], s["defteam_timeouts"]
        sp, home, rec = s["spread"], s["is_home"], s["receive_2h_ko"]

        def after(dt):
            return np.maximum(s["gsr"] - dt, 0), np.maximum(s["hsr"] - dt, 0)

        def ends(dt):
            """Masks: the game ends during the play / the first half ends during the play."""
            g, h = after(dt)
            over = g <= 0
            return over, (s["first_half"] == 1) & (h <= 0) & ~over

        def terminal(d):  # exact value when the clock runs out; a tie goes to overtime (0.5)
            return np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))

        def halftime(d):  # first half over: the second half starts with a kickoff, timeouts reset
            ours = self.wp_prob(d, 1800, 1800, 1, 10, ko, 3, 3, 0, sp, home, wp_params)
            theirs = 1 - self.wp_prob(-d, 1800, 1800, 1, 10, ko, 3, 3, 0, -sp, 1 - home, wp_params)
            return np.where(rec == 1, ours, theirs)

        def settle(values, dt, d):
            """Replace values of states that cannot happen because time ran out."""
            over, half = ends(dt)
            v = np.asarray(values, float)
            if v.ndim == 2:
                v = np.where(over[:, None], terminal(d)[:, None], v)
                return np.where(half[:, None], halftime(d)[:, None], v)
            return np.where(over, terminal(d), np.where(half, halftime(d), v))

        def own_ball(dt):  # offense keeps the ball, 1st & 10 (or goal) at y
            g, h = after(dt)
            p = self.wp_prob(col(diff), col(g), col(h), 1, np.minimum(10, y), y, col(pto), col(dto),
                             col(rec), col(sp), col(home), wp_params).reshape(n, 99)
            return np.hstack([np.zeros((n, 1)), settle(p, dt, diff)])

        def opp_ball(dt):  # opponent ball 1st & 10 at y (its own yardline_100)
            g, h = after(dt)
            p = self.wp_prob(col(-diff), col(g), col(h), 1, np.minimum(10, y), y, col(dto), col(pto),
                             col(rec_opp), col(-sp), col(1 - home), wp_params).reshape(n, 99)
            return np.hstack([np.full((n, 1), np.nan), settle(1 - p, dt, diff)])

        def opp_receives(dt, pts):  # we scored `pts`, they receive the kickoff
            g, h = after(dt)
            v = 1 - self.wp_prob(-(diff + pts), g, h, 1, 10, ko, dto, pto, rec_opp, -sp, 1 - home,
                                 wp_params)
            return settle(v, dt, diff + pts)

        def we_receive(dt, pts):  # they scored `pts`, we receive the kickoff
            g, h = after(dt)
            v = self.wp_prob(diff - pts, g, h, 1, 10, ko, pto, dto, rec, sp, home, wp_params)
            return settle(v, dt, diff - pts)

        own_succ = own_ball(clock["go_success"])
        own_succ[:, 0] = (xp * opp_receives(clock["go_success"], 7)
                          + (1 - xp) * opp_receives(clock["go_success"], 6))
        return {
            "own_succ": own_succ,
            "opp_fail": opp_ball(clock["go_fail"]),
            "opp_punt": opp_ball(clock["punt"]),
            "own_punt": own_ball(clock["punt"]),
            "opp_fg": opp_ball(clock["fg"]),
            "fg_make": opp_receives(clock["fg"], 3),
            "td_against": xp * we_receive(clock["punt"], 7) + (1 - xp) * we_receive(clock["punt"], 6),
        }

    # -------------------------------------------------------------- the decision
    def gather_go(self, V: dict, ydstogo, yardline):
        """(n, 100) matrix of success-state values for e = 0..99 extra yards, and fail value."""
        ydstogo = np.asarray(ydstogo, int); yardline = np.asarray(yardline, int)
        rows = np.arange(len(yardline))[:, None]
        idx = np.clip(yardline[:, None] - ydstogo[:, None] - np.arange(100)[None, :], 0, 99)
        succ = V["own_succ"][rows, idx]
        fail = V["opp_fail"][rows[:, 0], np.clip(100 - yardline, 1, 99)]
        return succ, fail

    def score(self, s: dict, V: dict, ydstogo, yardline, params=None, gathered=None) -> dict:
        params = params or self.point
        ydstogo = np.asarray(ydstogo, int); yardline = np.asarray(yardline, int)
        n = len(yardline)
        rows = np.arange(n)
        succ, fail = gathered if gathered is not None else self.gather_go(V, ydstogo, yardline)
        pmf = params["extra"][F.ydstogo_bucket(ydstogo)]
        p_conv = self.p_convert(ydstogo, yardline, s, params["conv"])
        ev_succ = (succ * pmf).sum(axis=1)
        go = p_conv * ev_succ + (1 - p_conv) * fail

        p_make = self.p_make(yardline, s, params["fg"])
        miss_spot = np.clip(np.minimum(80, 93 - yardline), 1, 99)
        fg = p_make * V["fg_make"] + (1 - p_make) * V["opp_fg"][rows, miss_spot]

        opp_pmf, own_pmf, td = params["punt"]
        k = np.clip(yardline, 1, 99) - 1
        punt = ((opp_pmf[k] * np.nan_to_num(V["opp_punt"])).sum(1) + (own_pmf[k] * V["own_punt"]).sum(1)
                + td[k] * V["td_against"])

        ev_all = np.column_stack([go, punt, fg])          # every call priced, even if unusual
        avail = np.column_stack([np.ones(n, bool), yardline >= self.c["min_punt_yardline"],
                                 yardline + 18 <= self.c["max_fg_distance"]])
        ev = np.where(avail, ev_all, np.nan)              # only sensible calls are recommended
        best = np.nanargmax(ev, axis=1)
        return {"go": go, "punt": ev[:, 1], "fg": ev[:, 2], "p_convert": p_conv, "p_make": p_make,
                "ev": ev, "ev_all": ev_all, "best": best}

    def evaluate(self, situation: Situation, ydstogo, yardline) -> dict:
        ydstogo = np.atleast_1d(np.asarray(ydstogo, int))
        yardline = np.atleast_1d(np.asarray(yardline, int))
        n = max(len(ydstogo), len(yardline))
        ydstogo = np.broadcast_to(ydstogo, (n,)); yardline = np.broadcast_to(yardline, (n,))
        s = situation.arrays(n, self.c["default_season"])
        V = self.state_values(s)
        return self.score(s, V, ydstogo, yardline)

    def grid(self, situation: Situation, max_ydstogo: int = 15) -> dict:
        """Decision chart for one game situation: every (yards to go, yardline) cell.

        State values depend only on the situation, so they are computed once and shared."""
        yl = np.arange(1, 100)
        ytg = np.arange(1, max_ydstogo + 1)
        YL, YTG = np.meshgrid(yl, ytg)
        yl_f, ytg_f = YL.ravel(), YTG.ravel()
        s1 = situation.arrays(1, self.c["default_season"])
        V1 = self.state_values(s1)
        m = len(yl_f)
        s = {k: np.repeat(v, m) for k, v in s1.items()}
        V = {k: np.repeat(v, m, axis=0) for k, v in V1.items()}
        out = self.score(s, V, ytg_f, yl_f)
        valid = ytg_f <= yl_f      # yards to go can never exceed the distance to the goal line
        shape = YL.shape
        res = {k: np.where(valid, out[k], np.nan).reshape(shape) for k in ("go", "punt", "fg")}
        res["best"] = np.where(valid, out["best"], -1).reshape(shape)
        res["ydstogo"], res["yardline"] = ytg, yl
        return res

    def recommend(self, situation: Situation, ydstogo: int, yardline: int) -> dict:
        """One situation, with an uncertainty band from the bootstrap replicates of the
        win-probability, conversion and field-goal models (punt and gain distributions are
        held at their point estimates here; the season audit resamples them too)."""
        s = situation.arrays(1, self.c["default_season"])
        V = self.state_values(s)
        ytg, yl = np.array([ydstogo]), np.array([yardline])
        gathered = self.gather_go(V, ytg, yl)
        pt = self.score(s, V, ytg, yl, gathered=gathered)
        evs = []
        for eb, wb, cb, fb in zip(self.spec["ep"]["boot"], self.spec["wp"]["boot"],
                                  self.spec["conv"]["boot"], self.spec["fg"]["boot"]):
            params = dict(self.point, ep=np.asarray(eb), wp=np.asarray(wb), conv=np.asarray(cb),
                          fg=np.asarray(fb))
            Vb = self.state_values(s, params)
            evs.append(self.score(s, Vb, ytg, yl, params=params)["ev"][0])
        best = int(pt["best"][0])
        evs = np.array(evs) if evs else pt["ev"].copy()   # no replicates: degenerate band
        ev = pt["ev"][0]
        order = np.argsort(-np.nan_to_num(ev, nan=-1))
        second = int(order[1]) if np.isfinite(ev[order[1]]) else None
        margin_b = evs[:, best] - (evs[:, second] if second is not None else 0)
        agree = float(np.mean(np.nanargmax(evs, axis=1) == best))
        return {
            "recommendation": CALLS[best],
            "wp": {c: (None if not np.isfinite(ev[i]) else float(ev[i])) for i, c in enumerate(CALLS)},
            "p_convert": float(pt["p_convert"][0]),
            "p_make": float(pt["p_make"][0]) if np.isfinite(ev[2]) else None,
            "margin": float(ev[best] - ev[second]) if second is not None else None,
            "margin_interval": ([float(np.percentile(margin_b, 5)), float(np.percentile(margin_b, 95))]
                                if second is not None else None),
            "runner_up": CALLS[second] if second is not None else None,
            "agreement": agree,
        }
