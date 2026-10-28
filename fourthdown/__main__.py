"""Command line.

    python -m fourthdown recommend --ydstogo 2 --yardline 45 --score-diff -3 --qtr 4 --seconds-left 300
    python -m fourthdown pipeline [--fast] [--no-refit]
    python -m fourthdown site
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config

NAMES = {"go": "GO FOR IT", "punt": "PUNT", "fg": "FIELD GOAL"}


def parse_yardline(text: str) -> int:
    """'45' = 45 yards from the opponent's end zone; 'own 30' / 'opp 30' also accepted."""
    t = text.lower().replace(" ", "").replace("_", "")
    if t.startswith("own"):
        val = 100 - int(t[3:])
    elif t.startswith("opp"):
        val = int(t[3:])
    else:
        val = int(t)
    if not 1 <= val <= 99:
        raise argparse.ArgumentTypeError("yardline must be between 1 and 99")
    return val


def describe_spot(yl: int) -> str:
    if yl == 50:
        return "midfield"
    return f"opponent {yl}" if yl < 50 else f"own {100 - yl}"


def cmd_recommend(a) -> int:
    from .engine import Engine, Situation
    if a.ydstogo > a.yardline:
        print("yards to go cannot exceed the distance to the goal line", file=sys.stderr)
        return 2
    engine = Engine.load(a.engine)
    sit = Situation(score_diff=a.score_diff, qtr=a.qtr, quarter_seconds=a.seconds_left,
                    posteam_timeouts=a.timeouts, defteam_timeouts=a.opp_timeouts,
                    receive_2h_ko=int(a.receive_2h_kickoff), spread=a.spread, total=a.total,
                    is_home=a.home, indoor=int(a.indoor), season=a.season)
    r = engine.recommend(sit, a.ydstogo, a.yardline)
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    mm, ss = divmod(int(a.seconds_left), 60)
    print(f"4th & {a.ydstogo} at the {describe_spot(a.yardline)}, Q{a.qtr} {mm}:{ss:02d}, "
          f"score diff {a.score_diff:+d}")
    print()
    for call in ("go", "fg", "punt"):
        wp = r["wp"][call]
        tag = "  <- recommended" if call == r["recommendation"] else ""
        print(f"  {NAMES[call]:<11} {'n/a' if wp is None else f'{100 * wp:5.1f}% win prob'}{tag}")
    print()
    print(f"  conversion chance {100 * r['p_convert']:.0f}%" +
          (f", field goal make chance {100 * r['p_make']:.0f}%" if r["p_make"] is not None else ""))
    if r["margin"] is not None:
        lo, hi = r["margin_interval"]
        print(f"  {NAMES[r['recommendation']]} beats {NAMES[r['runner_up']]} by "
              f"{100 * r['margin']:.1f} pts of win prob (90% band {100 * lo:+.1f} to {100 * hi:+.1f}); "
              f"same call in {100 * r['agreement']:.0f}% of bootstrap refits")
    strength = ("clear" if r["agreement"] >= 0.9 and (r["margin"] or 0) >= 0.01
                else "lean" if r["agreement"] >= 0.7 else "toss-up")
    print(f"  verdict: {NAMES[r['recommendation']]} ({strength})")
    if a.qtr in (2, 4) and a.seconds_left < 120:
        print("  caution: final two minutes of a half, where the win-probability model is least "
              "reliable (see README, reliability map)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fourthdown", description="NFL fourth-down decision audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recommend", help="what should they do in this situation?")
    r.add_argument("--ydstogo", type=int, required=True, help="yards to go for a first down")
    r.add_argument("--yardline", type=parse_yardline, required=True,
                   help="yards from the OPPONENT's end zone (45 = opp 45); 'own 30' also works")
    r.add_argument("--score-diff", type=int, default=0, help="offense score minus defense score")
    r.add_argument("--qtr", type=int, default=1, choices=[1, 2, 3, 4])
    r.add_argument("--seconds-left", type=int, default=900, help="seconds left in the quarter")
    r.add_argument("--timeouts", type=int, default=3, choices=[0, 1, 2, 3])
    r.add_argument("--opp-timeouts", type=int, default=3, choices=[0, 1, 2, 3])
    r.add_argument("--spread", type=float, default=0.0,
                   help="pre-game Vegas line for the offense (positive = offense favoured)")
    r.add_argument("--total", type=float, default=45.0, help="pre-game over/under")
    r.add_argument("--home", type=float, default=0.5, help="1 home, 0 away, 0.5 neutral")
    r.add_argument("--receive-2h-kickoff", action="store_true",
                   help="offense receives the second-half kickoff (first half only)")
    r.add_argument("--indoor", action="store_true", help="dome or closed roof")
    r.add_argument("--season", type=int, default=None)
    r.add_argument("--engine", default=str(config.ENGINE_PATH))
    r.add_argument("--json", action="store_true")

    pl = sub.add_parser("pipeline", help="rebuild models, audit, charts and site")
    pl.add_argument("--fast", action="store_true", help="40 bootstrap replicates instead of 200")
    pl.add_argument("--no-refit", action="store_true", help="reuse cached model fits")
    sub.add_parser("site", help="rebuild site/index.html from artifacts")
    sub.add_parser("charts", help="re-render docs/ charts from artifacts")

    a = p.parse_args(argv)
    if a.cmd == "recommend":
        return cmd_recommend(a)
    if a.cmd == "pipeline":
        from . import pipeline
        pipeline.run(n_boot=40 if a.fast else config.N_BOOT, refit=not a.no_refit)
        return 0
    if a.cmd == "site":
        from . import site
        site.build()
        return 0
    if a.cmd == "charts":
        from . import charts
        from .engine import Engine
        charts.render_all(Engine.load())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
