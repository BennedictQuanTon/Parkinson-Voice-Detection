PY := ./.venv/bin/python
JUPYTEXT := ./.venv/bin/jupytext
# librosa/numba and matplotlib both need writable cache directories; without
# these the feature extractors fail per-file rather than at import.
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export MPLBACKEND := Agg

PHASE0_STAGES := 01_data_cleaning 02_eda 03_feature_engineering \
                 04_modelling_evaluation 05_negative_controls 06_results

PHASE1_STAGES := 10_acquire_mdvr_kcl 11_harmonize_corpora 12_eda_cross_corpus \
                 13_channel_invariance 14_feature_engineering_v2 15_cross_corpus_model \
                 16_generalization_audit 17_results_phase1

.PHONY: all test phase0 phase1 notebooks figures sync clean-derived $(PHASE0_STAGES) $(PHASE1_STAGES)

all: test phase0 phase1

test:
	$(PY) -m pytest tests/ -q

phase0: $(PHASE0_STAGES)

phase1: $(PHASE1_STAGES)

# Phase 0 stages
$(PHASE0_STAGES):
	@echo "=== Phase 0: $@ ==="
	$(PY) scripts/phase0/$@.py

# Phase 1 stages
$(PHASE1_STAGES):
	@echo "=== Phase 1: $@ ==="
	$(PY) scripts/phase1/$@.py

# Phase 0 Stage dependencies
02_eda 03_feature_engineering: 01_data_cleaning
04_modelling_evaluation: 03_feature_engineering
05_negative_controls: 04_modelling_evaluation
06_results: 05_negative_controls

# Phase 1 Stage dependencies
11_harmonize_corpora: 10_acquire_mdvr_kcl
12_eda_cross_corpus: 11_harmonize_corpora
13_channel_invariance: 12_eda_cross_corpus
14_feature_engineering_v2: 13_channel_invariance
15_cross_corpus_model: 14_feature_engineering_v2
16_generalization_audit: 15_cross_corpus_model
17_results_phase1: 16_generalization_audit

figures:
	$(PY) scripts/phase0/02_eda.py
	$(PY) scripts/phase0/04_modelling_evaluation.py
	$(PY) scripts/phase0/05_negative_controls.py
	$(PY) scripts/phase0/06_results.py

notebooks:
	@echo "Syncing and executing Phase 0 notebooks..."
	@for stage in $(PHASE0_STAGES); do \
	  echo "=== Phase 0: $$stage ==="; \
	  $(JUPYTEXT) --sync --execute scripts/phase0/$$stage.py || exit 1; \
	done
	@echo "Syncing and executing Phase 1 notebooks..."
	@for stage in $(PHASE1_STAGES); do \
	  if [ -f scripts/phase1/$$stage.py ]; then \
	    echo "=== Phase 1: $$stage ==="; \
	    $(JUPYTEXT) --sync --execute scripts/phase1/$$stage.py || exit 1; \
	  fi \
	done

sync:
	@for f in scripts/phase0/*.py; do base=$$(basename $$f .py); $(JUPYTEXT) --to notebook --output "notebooks/phase0/$${base}.ipynb" "$$f"; done
	@for f in scripts/phase1/*.py; do base=$$(basename $$f .py); $(JUPYTEXT) --to notebook --output "notebooks/phase1/$${base}.ipynb" "$$f"; done

clean-derived:
	rm -rf data/processed results figures .cache
