PY := ./.venv/bin/python
JUPYTEXT := ./.venv/bin/jupytext
# librosa/numba and matplotlib both need writable cache directories; without
# these the feature extractors fail per-file rather than at import.
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

STAGES := 01_data_cleaning 02_eda 03_feature_engineering \
          04_modelling_evaluation 05_negative_controls 06_results

.PHONY: all test notebooks figures sync clean-derived $(STAGES)

# Fast path: run the pipeline as plain scripts, no kernel needed.
all: test $(STAGES)

test:
	$(PY) -m pytest tests/ -q

# Headless only when run as scripts. Notebook execution deliberately keeps the
# default inline backend so figures are embedded in the .ipynb.
$(STAGES): export MPLBACKEND := Agg
$(STAGES):
	@echo "=== $@ ==="
	$(PY) scripts/$@.py

# Stage dependencies. 01 produces the metadata everything else reads.
02_eda 03_feature_engineering: 01_data_cleaning
04_modelling_evaluation: 03_feature_engineering
05_negative_controls: 04_modelling_evaluation
06_results: 05_negative_controls

figures: 02_eda 04_modelling_evaluation 05_negative_controls 06_results

# Execute every cell and write the paired notebooks/*.ipynb with outputs.
# Slower than `make all` because each stage starts a Jupyter kernel.
notebooks:
	for stage in $(STAGES); do \
	  echo "=== $$stage ==="; \
	  $(JUPYTEXT) --sync --execute scripts/$$stage.py || exit 1; \
	done

# Propagate edits between scripts/*.py and notebooks/*.ipynb without running.
sync:
	$(JUPYTEXT) --sync scripts/*.py

# Removes generated artefacts only. Never touches data/raw.
clean-derived:
	rm -rf data/processed results figures .cache
