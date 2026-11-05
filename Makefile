# Regenerate everything published in this repo from data/ (no simulator needed).
PY ?= .venv/bin/python

.PHONY: all figures import
all:            ## number check, runs.csv, ULog CSVs, figures/, slides/ (incl. 3D replay, ~5 min)
	$(PY) make_results.py
figures:        ## same, without the video
	$(PY) make_results.py --no-video
import:         ## author only: copy results/ and PX4 ULogs into data/
	$(PY) tools/import_runs.py
