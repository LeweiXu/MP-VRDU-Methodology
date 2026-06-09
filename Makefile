# MP-VRDU common commands. Activate the venv first:
#   source ~/venvs/cits4010-venv/bin/activate
.PHONY: test smoke subset grid grid-dry report compare audit clean help

help:
	@grep -E '^[a-zA-Z_-]+:.*?# ' $(MAKEFILE_LIST) | sed 's/:.*# / - /'

test:                 # run the offline test suite
	python -m pytest

smoke:                # end-to-end offline smoke (synthetic data, mocked generator)
	python -m mpvrdu.pipeline --config configs/smoke.yaml

subset:               # download the small real subset (3 docs / 25 questions)
	python scripts/download_subset.py --out data/mmlongbench_subset --docs 3

grid-dry:             # preview the local grid without running
	python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --dry-run

grid:                 # run the full local grid (3B); resumable. SLICE/MAXQ overridable
	python scripts/run_grid.py --suite experiments/grid_local_3b.yaml \
		$(if $(SLICE),--slice $(SLICE),) $(if $(MAXQ),--max-questions $(MAXQ),)

report:               # build the analysis report (tables + figures) from results/grid
	python scripts/report.py --results results/grid

audit:                # contamination-audit fingerprints of the eval docs
	python scripts/contamination_audit.py --slice full

clean:                # remove generated outputs (keeps data/ + .cache/)
	rm -rf results fig
