# Demo script (optional, under 3 minutes)

The Data Bowl does not require a video. If I record one, this is the shot list.

| Time | Shot | What I say |
|---|---|---|
| 0:00-0:15 | Dashboard header and KPI row, season 2025, all teams | "Every NFL fourth down in 2025, priced in win probability. The league left about this many expected wins on the table, and most of it came from kicking when the numbers said go." |
| 0:15-0:45 | Decision chart. Change the situation from Q4 5:00 down 3 to Q1 tied with the preset button | "This is the chart a staff can print. Each cell is the call with the best win probability for yards to go and field position. It redraws live in the browser for any score, clock and timeouts." |
| 0:45-1:10 | Print chart button, then show `docs/decision_chart.pdf` | "One click gives a printable sheet with four common game situations." |
| 1:10-1:40 | Calculator: 4th & 2 at the opp 45, down 3, 5:00 left | "For any situation: the three options, the conversion and field-goal odds, the margin with a bootstrap band, and how often the call survives refitting the models on resampled games." |
| 1:40-2:10 | Team picker: pick the team at the top of the bar chart, then the costliest-decisions table | "By team and by head coach: wins left on the table with 95% intervals, an aggressiveness index against what the league would do in the same spots, and the ten costliest calls." |
| 2:10-2:40 | Methods section: holdout metrics and the reliability map; `docs/calibration.png` | "Everything was fit on 2016 to 2024 and tested on 2025. The win-probability model scores within a hair of nflverse's own, the conversion and field-goal models beat lookup tables, and I show where the model is weakest: the final two minutes." |
| 2:40-2:55 | Terminal: `python -m fourthdown recommend --ydstogo 2 --yardline 45 --score-diff -3 --qtr 4 --seconds-left 300` | "The same engine runs from the command line, and the whole pipeline rebuilds with one `make all`." |
