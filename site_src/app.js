/* Fourth Down Ledger dashboard. Data: window.LEDGER; engine: window.FourthDown + ENGINE_SPEC. */
(function () {
  "use strict";
  const L = window.LEDGER;
  const engine = new FourthDown.Engine(window.ENGINE_SPEC);
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const NAME = { go: "Go for it", punt: "Punt", fg: "Field goal" };
  const state = { season: L.default_season, team: "" };

  const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "n/a" : (100 * x).toFixed(d) + "%");
  const pts = (x, d = 1) => (x == null ? "n/a" : (100 * x).toFixed(d));
  const num = (x, d = 2) => (x == null || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
  const signed = (x, d = 1) => (x == null ? "n/a" : (x >= 0 ? "+" : "") + Number(x).toFixed(d));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const spot = (yl) => (yl === 50 ? "50" : yl < 50 ? "opp " + yl : "own " + (100 - yl));
  const clockStr = (s) => Math.floor(s / 60) + ":" + String(Math.round(s % 60)).padStart(2, "0");
  const pill = (c) => `<span class="pill ${c}">${NAME[c]}</span>`;
  function parseClock(v) {
    const m = String(v).trim().match(/^(\d{1,2})(?::(\d{1,2}))?$/);
    if (!m) return 300;
    const s = (+m[1]) * 60 + (+(m[2] || 0));
    return Math.min(Math.max(s, 0), 900);
  }

  // ------------------------------------------------------------------ plot theming
  function baseLayout(extra) {
    return Object.assign({
      paper_bgcolor: css("--surface-1"), plot_bgcolor: css("--surface-1"),
      font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 12, color: css("--text-secondary") },
      margin: { l: 56, r: 16, t: 10, b: 50 },
      hoverlabel: { bgcolor: css("--surface-1"), bordercolor: css("--border"), font: { color: css("--text-primary") } },
    }, extra);
  }
  const config = { displayModeBar: false, responsive: true };

  function mix(hex, amt) {
    const s = css("--surface-1");
    const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const a = p(hex), b = p(s.length === 7 ? s : "#fcfcfb");
    return "rgb(" + a.map((v, i) => Math.round(v * amt + b[i] * (1 - amt))).join(",") + ")";
  }

  // ------------------------------------------------------------------ selectors
  function initSelectors() {
    const ss = $("season");
    Object.keys(L.seasons).sort().reverse().forEach((s) => {
      const sum = L.seasons[s].summary;
      const o = document.createElement("option");
      o.value = s;
      o.textContent = s + (sum.weeks < 18 ? ` (through week ${sum.weeks})` : "");
      ss.appendChild(o);
    });
    ss.value = state.season;
    ss.onchange = () => { state.season = ss.value; fillTeams(); renderData(); };
    $("team").onchange = () => { state.team = $("team").value; renderData(); };
    fillTeams();
  }
  function fillTeams() {
    const sel = $("team");
    const teams = L.seasons[state.season].teams.map((t) => t.posteam).sort();
    sel.innerHTML = '<option value="">All teams</option>' + teams.map((t) => `<option>${t}</option>`).join("");
    if (!teams.includes(state.team)) state.team = "";
    sel.value = state.team;
  }

  // ------------------------------------------------------------------ KPIs
  function renderKpis() {
    const S = L.seasons[state.season];
    const box = (label, value, sub) => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
    let html;
    if (state.team) {
      const t = S.teams.find((r) => r.posteam === state.team);
      const rank = S.teams.slice().sort((a, b) => b.wp_lost - a.wp_lost).findIndex((r) => r.posteam === state.team) + 1;
      html = box("Expected wins left on the table", num(t.wp_lost), `95% CI ${num(t.wp_lost_lo)} to ${num(t.wp_lost_hi)} · rank ${rank} of ${S.teams.length} (1 = most)`)
        + box("Without the final minutes", num(t.wp_lost_excl_late), "expected wins lost outside Q4 under 5:00 and Q2 under 2:00")
        + box("Go rate on 4th down", pct(t.go_rate, 0), `${t.fourth_downs} fourth downs in ${t.games} games`)
        + box("Aggressiveness index", signed(t.aggressiveness), `pts above league expectation · CI ${signed(t.aggressiveness_lo)} to ${signed(t.aggressiveness_hi)}`)
        + box("Followed clear go calls", pct(t.clear_go_followed, 0), `${t.clear_go_spots} spots where going was clearly right`);
    } else {
      const s = S.summary;
      html = box("League-wide expected wins lost", num(s.wp_lost_total, 1), `95% CI ${num(s.wp_lost_total_ci[0], 1)} to ${num(s.wp_lost_total_ci[1], 1)} · ${s.fourth_downs.toLocaleString()} fourth downs`)
        + box("Teams rank the same without the final minutes", num(s.team_rank_corr_excl_late, 2), "Spearman correlation of team totals with and without late-game plays")
        + box("Share lost by kicking", pct(s.lost_by_kicking_share, 0), "of the total came from punting or kicking when the model said go")
        + box("Went for it when the model said go", pct(s.should_go_went, 0), `${s.should_go.toLocaleString()} go-recommended spots`)
        + box("League go rate", pct(s.go_rate, 0), `${s.games} games, through week ${s.weeks}`);
    }
    $("kpis").innerHTML = html;
  }

  // ------------------------------------------------------------------ decision chart
  function gridSituation() {
    return {
      qtr: +$("g-qtr").value, quarterSeconds: parseClock($("g-clock").value), scoreDiff: +$("g-diff").value || 0,
      timeouts: +$("g-to").value, oppTimeouts: +$("g-oto").value, spread: +$("g-spread").value || 0,
    };
  }
  function renderGrid() {
    const sit = gridSituation();
    const cells = engine.grid(sit, 15);
    const C = [css("--go"), css("--punt"), css("--fg")];
    const x = []; for (let i = 1; i < 100; i++) x.push(i);  // yards from own goal line
    const y = []; for (let i = 1; i <= 15; i++) y.push(i);
    const z = [], text = [];
    cells.forEach((row) => {
      const zr = new Array(99).fill(null), tr = new Array(99).fill("");
      row.forEach((c, j) => {
        if (!c) return;
        const yl = j + 1, col = 99 - yl;  // column index by yards from own goal
        const ev = c.ev.map((v) => (Number.isNaN(v) ? -1 : v)).slice().sort((a, b) => b - a);
        const margin = ev[0] - ev[1];
        zr[col] = c.best + (margin < 0.005 ? 3 : 0);
        const best = FourthDown.CALLS[c.best];
        tr[col] = `<b>${NAME[best]}</b> at the ${spot(yl)}<br>Go ${pct(c.ev[0])} · Punt ${pct(c.ev[1])} · FG ${pct(c.ev[2])}`
          + `<br>conversion ${pct(c.pConvert, 0)}${Number.isNaN(c.ev[2]) ? "" : " · FG make " + pct(c.pMake, 0)}<br>margin ${pts(margin)} pts`;
      });
      z.push(zr); text.push(tr);
    });
    const stops = [C[0], C[1], C[2], mix(C[0], 0.35), mix(C[1], 0.35), mix(C[2], 0.35)];
    const colorscale = [];
    stops.forEach((c, i) => { colorscale.push([i / 6, c], [(i + 1) / 6, c]); });
    const trace = {
      type: "heatmap", x, y, z, text, hoverinfo: "text", hovertemplate: "%{text}<br>4th & %{y}<extra></extra>",
      colorscale, zmin: -0.5, zmax: 5.5, showscale: false, xgap: 0, ygap: 1,
    };
    const legend = ["go", "punt", "fg"].map((k, i) => ({
      type: "scatter", x: [null], y: [null], mode: "markers", name: NAME[k],
      marker: { color: C[i], size: 12, symbol: "square" }, hoverinfo: "skip",
    }));
    const ticks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99];
    const tickText = ticks.map((t) => (t === 50 ? "50" : t < 50 ? "own " + t : "opp " + (100 - t)));
    Plotly.react("grid", [trace, ...legend], baseLayout({
      xaxis: { tickvals: ticks, ticktext: tickText, title: { text: "field position (offense drives left to right)" }, showgrid: false, zeroline: false, range: [0.5, 99.5] },
      yaxis: { autorange: "reversed", title: { text: "yards to go" }, dtick: 1, showgrid: false, zeroline: false },
      legend: { orientation: "h", y: 1.08, x: 0 }, margin: { l: 56, r: 16, t: 36, b: 56 },
    }), config);
  }

  // ------------------------------------------------------------------ calculator
  function renderCalc() {
    const ytg = Math.max(1, +$("c-ytg").value || 1);
    const n = Math.min(50, Math.max(1, +$("c-yl").value || 50));
    const yl = $("c-side").value === "opp" ? n : 100 - n;
    const out = $("calc-out");
    if (ytg > yl) { out.innerHTML = '<p class="note">Yards to go cannot be more than the distance to the goal line.</p>'; return; }
    const r = engine.recommend({
      qtr: +$("c-qtr").value, quarterSeconds: parseClock($("c-clock").value), scoreDiff: +$("c-diff").value || 0,
      timeouts: +$("c-to").value, oppTimeouts: +$("c-oto").value, spread: +$("c-spread").value || 0,
      indoor: $("c-indoor").checked, receive2h: $("c-rec").checked,
    }, ytg, yl);
    const strength = r.agreement >= 0.9 && (r.margin ?? 0) >= 0.01 ? "clear call" : r.agreement >= 0.7 ? "lean" : "toss-up";
    const max = Math.max(...Object.values(r.wp).filter((v) => v != null));
    const rows = ["go", "fg", "punt"].map((k) => {
      const v = r.wp[k];
      if (v == null) return `<div class="bar-row"><span>${NAME[k]}</span><span class="note">not a realistic option here</span><span></span></div>`;
      const w = Math.max(2, 100 * v / Math.max(max, 1e-6));
      return `<div class="bar-row"><span>${NAME[k]}</span><div class="bar-track"><div class="bar" style="width:${w}%;background:var(--${k})"></div></div><span class="num">${pct(v)}</span></div>`;
    }).join("");
    out.innerHTML = `<div class="label note">Recommended</div><div class="verdict" style="color:var(--${r.recommendation})">${NAME[r.recommendation]}</div>`
      + `<p class="note">${strength}: ${r.margin == null ? "" : `better than ${NAME[r.runner_up].toLowerCase()} by ${pts(r.margin)} points of win probability (90% band ${signed(100 * r.margin_interval[0])} to ${signed(100 * r.margin_interval[1])}); `}same call in ${pct(r.agreement, 0)} of bootstrap refits.</p>`
      + rows
      + (([2, 4].includes(+$("c-qtr").value) && parseClock($("c-clock").value) < 120)
        ? '<p class="note"><b>Caution:</b> final two minutes of a half, where the win-probability model is least reliable (see the reliability map).</p>' : "")
      + `<p class="note">Conversion chance ${pct(r.p_convert, 0)}${r.p_make == null ? "" : `, field goal make chance ${pct(r.p_make, 0)}`}. Win probability for the offense after each call.</p>`;
  }

  // ------------------------------------------------------------------ team plot
  function renderTeamsPlot() {
    const S = L.seasons[state.season];
    const t = S.teams.slice().sort((a, b) => a.wp_lost - b.wp_lost);
    const go = css("--go");
    const colors = t.map((r) => (!state.team || r.posteam === state.team ? go : mix(go, 0.3)));
    Plotly.react("teams-plot", [{
      type: "bar", orientation: "h", y: t.map((r) => r.posteam), x: t.map((r) => r.wp_lost),
      marker: { color: colors },
      error_x: { type: "data", symmetric: false, array: t.map((r) => r.wp_lost_hi - r.wp_lost), arrayminus: t.map((r) => r.wp_lost - r.wp_lost_lo), color: css("--text-secondary"), thickness: 1.2, width: 0 },
      customdata: t.map((r) => [r.wp_lost_lo, r.wp_lost_hi, r.go_rate, r.aggressiveness, r.fourth_downs]),
      hovertemplate: "<b>%{y}</b><br>%{x:.2f} expected wins lost (CI %{customdata[0]:.2f} to %{customdata[1]:.2f})<br>go rate %{customdata[2]:.0%} · aggressiveness %{customdata[3]:+.1f}<br>%{customdata[4]} fourth downs<extra></extra>",
    }], baseLayout({
      xaxis: { title: { text: "expected wins left on the table" }, gridcolor: css("--grid"), zeroline: false, rangemode: "tozero" },
      yaxis: { automargin: true, tickfont: { size: 11 } }, bargap: 0.35, margin: { l: 48, r: 16, t: 6, b: 50 },
    }), config);
    $("teams-title").textContent = `Win probability left on the table, ${state.season}${S.summary.weeks < 18 ? " (to date)" : ""}`;
  }

  // ------------------------------------------------------------------ tables
  function renderTop() {
    const S = L.seasons[state.season];
    const exclLate = $("excl-late").checked;
    let plays = S.plays.filter((p) => (!state.team || p.posteam === state.team) && !(exclLate && p.late));
    plays = plays.slice().sort((a, b) => b.wp_lost - a.wp_lost).slice(0, 10);
    $("top-title").textContent = `Costliest decisions${state.team ? " — " + state.team : ""}, ${state.season}`;
    $("top").innerHTML = "<thead><tr><th>Wk</th><th>Team</th><th>Situation</th><th>Called</th><th>Model</th><th class='num'>WP lost</th><th class='num'>95% CI</th><th class='num'>Refits agree</th><th>Play</th></tr></thead><tbody>"
      + plays.map((p) => `<tr><td>${p.week}</td><td>${p.posteam} v ${p.defteam}</td>`
        + `<td>4th & ${p.ydstogo} at ${spot(p.yardline_100)}${p.late ? '<span class="tag">late</span>' : ""}<br><span class="note">Q${p.qtr} ${clockStr(p.quarter_seconds_remaining)}, ${p.score_differential > 0 ? "up " + p.score_differential : p.score_differential < 0 ? "down " + (-p.score_differential) : "tied"}</span></td>`
        + `<td>${pill(p.call)}</td><td>${pill(p.recommended)}</td><td class="num">${pts(p.wp_lost)}</td><td class="num">${pts(p.wp_lost_lo)} to ${pts(p.wp_lost_hi)}</td>`
        + `<td class="num">${pct(p.agreement, 0)}</td><td class="desc">${esc(p.desc)}</td></tr>`).join("") + "</tbody>";
  }

  const TEAM_COLS = [
    ["posteam", "Team", (r) => r.posteam],
    ["wp_lost", "Wins lost", (r) => num(r.wp_lost), true],
    ["wp_lost_lo", "95% CI", (r) => `${num(r.wp_lost_lo)} to ${num(r.wp_lost_hi)}`, true],
    ["wp_lost_per_game", "Per game", (r) => num(r.wp_lost_per_game, 3), true],
    ["fourth_downs", "4th downs", (r) => r.fourth_downs, true],
    ["go_rate", "Go rate", (r) => pct(r.go_rate, 0), true],
    ["aggressiveness", "Aggressiveness", (r) => `${signed(r.aggressiveness)} <span class="note">(${signed(r.aggressiveness_lo)}, ${signed(r.aggressiveness_hi)})</span>`, true],
    ["lost_kicking", "Lost by kicking", (r) => num(r.lost_kicking), true],
    ["lost_going", "Lost by going", (r) => num(r.lost_going), true],
    ["clear_go_followed", "Clear-go follow rate", (r) => pct(r.clear_go_followed, 0), true],
  ];
  const sortState = {};
  function renderTable(id, rows, cols, keyCol) {
    const st = sortState[id] || { key: "wp_lost", dir: -1 };
    const sorted = rows.slice().sort((a, b) => {
      const va = a[st.key], vb = b[st.key];
      if (typeof va === "string") return st.dir * va.localeCompare(vb);
      return st.dir * ((va ?? -Infinity) - (vb ?? -Infinity));
    });
    const el = $(id);
    el.innerHTML = "<thead><tr>" + cols.map(([k, h, , n]) => `<th data-k="${k}" class="${n ? "num" : ""}" aria-sort="${st.key === k ? (st.dir > 0 ? "ascending" : "descending") : "none"}">${h}${st.key === k ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr></thead><tbody>"
      + sorted.map((r) => `<tr class="${state.team && r.team === state.team ? "hl" : ""}">` + cols.map(([, , f, n]) => `<td class="${n ? "num" : ""}">${f(r)}</td>`).join("") + "</tr>").join("") + "</tbody>";
    el.querySelectorAll("th").forEach((th) => th.onclick = () => {
      const k = th.dataset.k;
      sortState[id] = { key: k, dir: st.key === k ? -st.dir : (k === keyCol ? 1 : -1) };
      renderTable(id, rows, cols, keyCol);
    });
  }

  function renderMetrics() {
    const M = L.metrics;
    const m = (title, body) => `<div class="metric"><b>${title}</b>${body}</div>`;
    const ci = (a) => `${num(a[0], 4)} to ${num(a[1], 4)}`;
    $("metrics").innerHTML = `<p class="note">Every model was fit on ${M.train_seasons[0]}-${M.train_seasons[M.train_seasons.length - 1]} and scored on the untouched ${M.holdout_season} season. Lower Brier is better; a calibration slope of 1 means the probabilities are neither too bold nor too timid. Intervals resample whole games.</p><div class="metrics-grid">`
      + m("Win probability", `Brier ${num(M.wp.brier, 4)} (CI ${ci(M.wp.brier_ci)}) on ${M.wp.n.toLocaleString()} plays; nflverse's own model scores ${num(M.wp_nflverse_vegas_wp.brier, 4)}. Calibration slope ${num(M.wp.calibration_slope)}, expected calibration error ${pct(M.wp.ece)}.`)
      + m("Fourth-down conversion", `Brier ${num(M.conv_fourth.brier, 4)} on ${M.conv_fourth.n} attempts vs ${num(M.conv_fourth.brier_baseline, 4)} for a by-distance lookup table (skill ${signed(100 * M.conv_fourth.brier_skill)}%, CI ${signed(100 * M.conv_fourth.brier_skill_ci[0])} to ${signed(100 * M.conv_fourth.brier_skill_ci[1])}). Calibration slope ${num(M.conv_fourth.calibration_slope)}.`)
      + m("Field goals", `Brier ${num(M.fg.brier, 4)} on ${M.fg.n.toLocaleString()} kicks vs ${num(M.fg.brier_baseline, 4)} for 5-yard distance bins. Calibration slope ${num(M.fg.calibration_slope)}.`)
      + m("Punts", `Mean predicted starting spot is off by ${num(M.punt.bias, 2)} yards on ${M.punt.n.toLocaleString()} punts; the predicted 90% range covers ${pct(M.punt.coverage_90)} of them.`)
      + m("Whole chain", `Win probability of the call actually made, against who won: Brier ${num(M.chain.brier, 4)} on ${M.chain.n.toLocaleString()} fourth downs, calibration slope ${num(M.chain.calibration_slope)}.`)
      + "</div>";
    const R = M.reliability || [];
    $("reliability").innerHTML = "<thead><tr><th>Time left in game</th><th class='num'>Plays</th><th class='num'>Mean gap vs nflverse</th></tr></thead><tbody>"
      + R.map((r) => `<tr><td>${esc(r.phase)}</td><td class="num">${r.n.toLocaleString()}</td><td class="num">${pts(r.mad)} pts</td></tr>`).join("") + "</tbody>";
  }

  function renderData() {
    const S = L.seasons[state.season];
    renderKpis(); renderTeamsPlot(); renderTop();
    renderTable("teams-table", S.teams, TEAM_COLS, "posteam");
    const coachCols = [["coach", "Coach", (r) => esc(r.coach)], ["team", "Team", (r) => r.team]].concat(TEAM_COLS.slice(1));
    renderTable("coaches-table", S.coaches, coachCols, "coach");
  }

  function renderAll() { renderData(); renderGrid(); renderCalc(); renderMetrics(); }

  // ------------------------------------------------------------------ wiring
  ["g-qtr", "g-clock", "g-diff", "g-to", "g-oto", "g-spread"].forEach((id) => $(id).addEventListener("change", renderGrid));
  $("g-preset-q1").onclick = () => { $("g-qtr").value = "1"; $("g-clock").value = "15:00"; $("g-diff").value = 0; renderGrid(); };
  $("calc").addEventListener("input", renderCalc);
  $("calc").addEventListener("change", renderCalc);
  $("print").onclick = () => window.print();
  $("excl-late").onchange = renderTop;
  $("theme").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    $("theme").textContent = dark ? "Dark mode" : "Light mode";
    renderData(); renderGrid();
  };
  initSelectors();
  renderAll();
})();
