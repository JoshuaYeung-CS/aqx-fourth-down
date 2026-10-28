# Fourth Down Ledger: one command from public data to dashboard.
PY      := .venv/bin/python
SEASONS := 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
URL     := https://github.com/nflverse/nflverse-data/releases/download/pbp
export LOKY_MAX_CPU_COUNT := 4
export MPLCONFIGDIR := $(CURDIR)/.venv/.mpl

.PHONY: all venv data pipeline fast site charts test clean

all: venv data pipeline test

venv: .venv/bin/python
.venv/bin/python:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

data: $(foreach s,$(SEASONS),data/raw/play_by_play_$(s).parquet)
data/raw/play_by_play_%.parquet:
	@mkdir -p data/raw
	curl -sSfL --retry 2 -o $@ $(URL)/play_by_play_$*.parquet

pipeline:
	$(PY) -m fourthdown pipeline

fast:
	$(PY) -m fourthdown pipeline --fast

site:
	$(PY) -m fourthdown site

charts:
	$(PY) -m fourthdown charts

test:
	$(PY) -m pytest -q

clean:
	rm -rf artifacts/cache
