import json

import pytest

from fourthdown import __main__ as cli, config, site


def test_site_build(tmp_path, engine):
    if not (config.ARTIFACTS / "metrics.json").exists():
        pytest.skip("run the pipeline first")
    out = tmp_path / "index.html"
    site.build(out)
    html = out.read_text()
    for marker in ("window.ENGINE_SPEC=", "window.LEDGER=", "Plotly", "Decision chart",
                   "What should they do?", "Costliest decisions", 'id="team"'):
        assert marker in html
    assert "cdn." not in html.split("<script>")[0]           # no external stylesheet or script
    assert "<script src=" not in html
    assert out.stat().st_size < 16e6
    data = site.site_data()
    for s in config.AUDIT_SEASONS:
        assert len(data["seasons"][str(s)]["teams"]) == 32


def test_cli_recommend_json(capsys, engine):
    rc = cli.main(["recommend", "--ydstogo", "2", "--yardline", "45", "--score-diff", "-3",
                   "--qtr", "4", "--seconds-left", "300", "--json"])
    assert rc == 0
    r = json.loads(capsys.readouterr().out)
    assert r["recommendation"] in ("go", "punt", "fg")


def test_cli_text_and_yardline_parsing(capsys, engine):
    assert cli.parse_yardline("own 30") == 70
    assert cli.parse_yardline("opp30") == 30
    assert cli.parse_yardline("45") == 45
    rc = cli.main(["recommend", "--ydstogo", "1", "--yardline", "own 45", "--qtr", "1"])
    assert rc == 0
    assert "verdict" in capsys.readouterr().out
    assert cli.main(["recommend", "--ydstogo", "12", "--yardline", "5"]) == 2
