"""Stage 7: consolidate JSONL outputs into tables + figures.

Everything here reads ONLY the JSONL result files (which embed their config in
the meta header), so the whole results section regenerates from raw outputs with
one command — the reproducibility requirement.
"""

from .aggregate import (CONDITION_FIELDS, TIER1_FIELDS, aggregate_dir,
                        bootstrap_ci, paired_diff_test, summarize_run,
                        to_markdown_table)  # noqa: F401
from .report import (abstention_table, build_report, cost_table,
                     evidence_source_table, oracle_gap, pairwise_significance,
                     question_type_table, recall_accuracy_correlation,
                     sanity_checks, seed_variance, tier1_table,
                     topk_series)  # noqa: F401
