/* Fourth Down Ledger decision engine: a line-for-line port of fourthdown/engine.py and
 * fourthdown/features.py. It reads the same engine.json, so the browser calculator, the
 * command line and the season audit produce identical numbers (checked in the test suite). */
(function (root) {
  "use strict";
  const EP_KNOTS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95];
  const LAST_TRAIN_SEASON = 2024;
  const CALLS = ["go", "punt", "fg"];

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const seasonC = (s) => Math.min(s, LAST_TRAIN_SEASON) - 2020;

  function epRow(down, ytg, yl, hsr) {
    const d = [down === 2 ? 1 : 0, down === 3 ? 1 : 0, down === 4 ? 1 : 0];
    const lg = Math.log(Math.max(ytg, 1));
    const gtg = ytg >= yl ? 1 : 0;
    const e = Math.exp(-hsr / 90);
    const row = [1, yl / 100];
    for (const k of EP_KNOTS) row.push(Math.max(0, yl - k) / 100);
    row.push(...d);
    for (const x of d) row.push(x * yl / 100);
    for (const x of d) row.push(x * Math.max(0, yl - 60) / 100);
    row.push(lg);
    for (const x of d) row.push(x * lg);
    row.push(gtg, gtg * yl / 100, e, e * yl / 100, e * (yl <= 35 ? 1 : 0));
    return row;
  }

  const MARGIN_EDGES = [-8.5, -3.5, -0.5, 0.5, 3.5, 8.5];
  function marginBin(diff) {  // same as numpy searchsorted(edges, diff, side="left")
    let k = 0;
    while (k < MARGIN_EDGES.length && MARGIN_EDGES[k] < diff) k++;
    return k;
  }

  function wpRow(diff, gsr, hsr, pto, dto, rec, spread, home, ep) {
    const t = gsr / 3600;
    const r = 1 / Math.sqrt(t + 0.01);
    const late = Math.exp(-gsr / 150);
    const late10 = Math.exp(-gsr / 600);
    const frac = (gsr > 1800 ? 1 : 0) * hsr / 1800;
    const recs = (2 * rec - 1) * frac;
    const row = [diff, diff * r, ep, ep * r, spread * t, spread * t * r, recs, recs * frac, home, pto, dto,
      (pto - dto) * r, (diff > 0 ? 1 : 0) * late, (diff === 0 ? 1 : 0) * late,
      (diff < 0 && diff >= -3 ? 1 : 0) * late, (diff < -3 && diff >= -8 ? 1 : 0) * late, ep * late,
      clip(diff, -17, 17) * late, pto * late, dto * late,
      ep * late10, clip(diff, -17, 17) * late10, (diff > 0 ? 1 : 0) * late10, (diff === 0 ? 1 : 0) * late10,
      (diff < 0 && diff >= -8 ? 1 : 0) * late10, pto * late10, dto * late10, ep * r * late10];
    const bin = marginBin(diff);
    for (let k = 0; k < 7; k++) {
      const ind = bin === k ? 1 : 0;
      row.push(ind * late10, ind * ep * late10, ind * late, ind * ep * late, ind * pto * late10, ind * dto * late10);
    }
    return row;
  }

  function convRow(ytg, yl, is4th, spread, total, season) {
    const lg = Math.log(ytg);
    return [lg, ytg, yl / 100, (yl / 100) ** 2, ytg >= yl ? 1 : 0, is4th, is4th * lg,
      yl <= 5 ? 1 : 0, spread, total, seasonC(season)];
  }

  function fgRow(distance, indoor, season) {
    const s = seasonC(season);
    return [distance, Math.max(0, distance - 50) / 10, Math.max(0, distance - 58) / 10, indoor, s, distance * s / 10];
  }

  function bucket(ytg) {
    if (ytg <= 1) return 0;
    if (ytg <= 2) return 1;
    if (ytg <= 3) return 2;
    if (ytg <= 5) return 3;
    if (ytg <= 10) return 4;
    return 5;
  }

  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  function standardLogit(row, mean, scale, params) {
    let s = params[0];
    for (let i = 0; i < row.length; i++) s += ((row[i] - mean[i]) / scale[i]) * params[i + 1];
    return sigmoid(s);
  }

  class Engine {
    constructor(spec) {
      this.spec = spec;
      this.c = spec.constants;
      this.point = {
        ep: spec.ep.params, wp: spec.wp.params, conv: spec.conv.params, fg: spec.fg.params,
        extra: spec.extra.pmf, punt: [spec.punt.opp, spec.punt.own, spec.punt.td],
      };
    }

    situation(o) {
      const q = o.qtr, qs = o.quarterSeconds;
      const firstHalf = q <= 2 ? 1 : 0;
      return {
        diff: o.scoreDiff, qtr: q, gsr: (4 - q) * 900 + qs,
        hsr: (q <= 2 ? (2 - q) * 900 : (4 - q) * 900) + qs, firstHalf,
        pto: o.timeouts ?? 3, dto: o.oppTimeouts ?? 3, rec: (o.receive2h ? 1 : 0) * firstHalf,
        spread: o.spread ?? 0, total: o.total ?? 45, home: o.home ?? 0.5, indoor: o.indoor ? 1 : 0,
        season: o.season ?? this.c.default_season,
      };
    }

    kickoffYl(season) {
      const ko = this.c.kickoff_start;
      const keys = Object.keys(ko).map(Number);
      const latest = ko[String(Math.max(...keys))];
      return ko[String(season)] ?? latest;
    }

    wpProb(diff, gsr, hsr, down, ytg, yl, pto, dto, rec, spread, home, prm) {
      const ep = dot(epRow(down, ytg, yl, hsr), prm.ep);
      return standardLogit(wpRow(diff, gsr, hsr, pto, dto, rec, spread, home, ep),
        this.spec.wp.mean, this.spec.wp.scale, prm.wp);
    }

    stateValues(s, prm) {
      prm = prm || this.point;
      const recOpp = s.firstHalf * (1 - s.rec);
      const xp = this.c.xp_rate, ko = this.kickoffYl(s.season), clock = this.c.clock;
      const after = (dt) => [Math.max(s.gsr - dt, 0), Math.max(s.hsr - dt, 0)];
      const terminal = (d) => (d > 0 ? 1 : d < 0 ? 0 : 0.5);
      const halftime = (d) => (s.rec === 1
        ? this.wpProb(d, 1800, 1800, 1, 10, ko, 3, 3, 0, s.spread, s.home, prm)
        : 1 - this.wpProb(-d, 1800, 1800, 1, 10, ko, 3, 3, 0, -s.spread, 1 - s.home, prm));
      // value of a state that cannot happen because the clock ran out, else null
      const settled = (dt, d) => {
        const [g, h] = after(dt);
        if (g <= 0) return terminal(d);
        if (s.firstHalf === 1 && h <= 0) return halftime(d);
        return null;
      };
      const ownBall = (dt) => {
        const [g, h] = after(dt);
        const a = new Float64Array(100);
        const fixed = settled(dt, s.diff);
        for (let y = 1; y < 100; y++)
          a[y] = fixed !== null ? fixed
            : this.wpProb(s.diff, g, h, 1, Math.min(10, y), y, s.pto, s.dto, s.rec, s.spread, s.home, prm);
        return a;
      };
      const oppBall = (dt) => {
        const [g, h] = after(dt);
        const a = new Float64Array(100);
        a[0] = NaN;
        const fixed = settled(dt, s.diff);
        for (let y = 1; y < 100; y++)
          a[y] = fixed !== null ? fixed
            : 1 - this.wpProb(-s.diff, g, h, 1, Math.min(10, y), y, s.dto, s.pto, recOpp, -s.spread, 1 - s.home, prm);
        return a;
      };
      const oppReceives = (dt, pts) => {
        const [g, h] = after(dt);
        const fixed = settled(dt, s.diff + pts);
        return fixed !== null ? fixed
          : 1 - this.wpProb(-(s.diff + pts), g, h, 1, 10, ko, s.dto, s.pto, recOpp, -s.spread, 1 - s.home, prm);
      };
      const weReceive = (dt, pts) => {
        const [g, h] = after(dt);
        const fixed = settled(dt, s.diff - pts);
        return fixed !== null ? fixed
          : this.wpProb(s.diff - pts, g, h, 1, 10, ko, s.pto, s.dto, s.rec, s.spread, s.home, prm);
      };
      const ownSucc = ownBall(clock.go_success);
      ownSucc[0] = xp * oppReceives(clock.go_success, 7) + (1 - xp) * oppReceives(clock.go_success, 6);
      return {
        ownSucc, oppFail: oppBall(clock.go_fail), oppPunt: oppBall(clock.punt),
        ownPunt: ownBall(clock.punt), oppFg: oppBall(clock.fg), fgMake: oppReceives(clock.fg, 3),
        tdAgainst: xp * weReceive(clock.punt, 7) + (1 - xp) * weReceive(clock.punt, 6),
      };
    }

    score(s, V, ytg, yl, prm) {
      prm = prm || this.point;
      const pmf = prm.extra[bucket(ytg)];
      let evSucc = 0;
      for (let e = 0; e < 100; e++) evSucc += V.ownSucc[clip(yl - ytg - e, 0, 99)] * pmf[e];
      const fail = V.oppFail[clip(100 - yl, 1, 99)];
      const pConv = standardLogit(convRow(ytg, yl, 1, s.spread, s.total, s.season),
        this.spec.conv.mean, this.spec.conv.scale, prm.conv);
      const go = pConv * evSucc + (1 - pConv) * fail;

      const pMake = standardLogit(fgRow(yl + 18, s.indoor, s.season), this.spec.fg.mean,
        this.spec.fg.scale, prm.fg);
      const miss = clip(Math.min(80, 93 - yl), 1, 99);
      const fg = pMake * V.fgMake + (1 - pMake) * V.oppFg[miss];

      const [opp, own, td] = prm.punt;
      const k = clip(yl, 1, 99) - 1;
      let punt = td[k] * V.tdAgainst;
      for (let y = 0; y < 100; y++) {
        const vo = V.oppPunt[y];
        punt += opp[k][y] * (Number.isNaN(vo) ? 0 : vo) + own[k][y] * V.ownPunt[y];
      }
      const avail = [true, yl >= this.c.min_punt_yardline, yl + 18 <= this.c.max_fg_distance];
      const ev = [go, punt, fg].map((v, i) => (avail[i] ? v : NaN));
      let best = 0;
      for (let i = 1; i < 3; i++) if (!Number.isNaN(ev[i]) && ev[i] > ev[best]) best = i;
      return { go, punt: ev[1], fg: ev[2], pConvert: pConv, pMake, ev, best };
    }

    grid(o, maxYtg = 15) {
      const s = this.situation(o);
      const V = this.stateValues(s);
      const cells = [];
      for (let ytg = 1; ytg <= maxYtg; ytg++) {
        const row = [];
        for (let yl = 1; yl < 100; yl++) row.push(ytg <= yl ? this.score(s, V, ytg, yl) : null);
        cells.push(row);
      }
      return cells;
    }

    recommend(o, ytg, yl) {
      const s = this.situation(o);
      const V = this.stateValues(s);
      const pt = this.score(s, V, ytg, yl);
      const best = pt.best;
      const order = [0, 1, 2].filter((i) => !Number.isNaN(pt.ev[i])).sort((a, b) => pt.ev[b] - pt.ev[a]);
      const second = order.length > 1 ? order[1] : null;
      const margins = [];
      let agree = 0;
      const B = this.spec.wp.boot.length;
      for (let b = 0; b < B; b++) {
        const prm = Object.assign({}, this.point, {
          ep: this.spec.ep.boot[b], wp: this.spec.wp.boot[b], conv: this.spec.conv.boot[b], fg: this.spec.fg.boot[b],
        });
        const r = this.score(s, this.stateValues(s, prm), ytg, yl, prm);
        if (r.best === best) agree++;
        if (second !== null) margins.push(r.ev[best] - r.ev[second]);
      }
      margins.sort((a, b) => a - b);
      const pct = (p) => {
        const i = (margins.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
        return margins[lo] + (margins[hi] - margins[lo]) * (i - lo);
      };
      return {
        recommendation: CALLS[best],
        wp: { go: pt.ev[0], punt: Number.isNaN(pt.ev[1]) ? null : pt.ev[1], fg: Number.isNaN(pt.ev[2]) ? null : pt.ev[2] },
        p_convert: pt.pConvert, p_make: Number.isNaN(pt.ev[2]) ? null : pt.pMake,
        margin: second === null ? null : pt.ev[best] - pt.ev[second],
        margin_interval: second === null ? null : [pct(0.05), pct(0.95)],
        runner_up: second === null ? null : CALLS[second], agreement: agree / B,
      };
    }
  }

  const api = { Engine, CALLS, epRow, wpRow, convRow, fgRow };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FourthDown = api;
})(typeof window !== "undefined" ? window : globalThis);
