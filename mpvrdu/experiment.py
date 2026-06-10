"""Experiment suites: expand a compact axes spec into many RunConfigs.

A suite YAML has two parts:

    defaults:        # a partial RunConfig applied to every run (incl. the
                     # generation block — the ONE place to edit for Kaya)
      data: {name: mmlongbench-doc, slice: full}
      generation: {generator: local_small_vlm, model_id: Qwen/Qwen2.5-VL-3B-Instruct}
      ...
    substudies:
      RQ1_retrieval:
        rq: RQ1                            # research-question framing metadata
        subset: question_type              # the discriminating subset (RQ spec §4)
        control: {kind: floor_ceiling}     # what to measure the effect against
        hypotheses:                        # rendered + verdicted by the reporter
          - {id: H1b, text: "...", expect: {...}}
        axes:                              # cross-producted into runs
          retrieval.method:    [bm25, dense, colpali]
          generation.modality: [image, text]
          retrieval.top_k:     [4]

Each axes combination is deep-merged onto `defaults`, given an auto name, and
validated. One suite -> a list of (substudy, RunConfig). The metadata keys
(everything but `axes`) are IGNORED by the expander and consumed only by the
analysis layer (``study_metadata`` / docs/research_questions.md §4). Switching the
whole grid to Kaya is a single edit to `defaults.generation` (generator + model_id).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig, dict_to_config


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _short(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "")


def expand_suite(suite: dict) -> list[tuple[str, RunConfig]]:
    """Expand a loaded suite dict into [(substudy, RunConfig), ...]."""
    defaults = suite.get("defaults", {})
    substudies = suite.get("substudies", {})
    runs: list[tuple[str, RunConfig]] = []

    for sub_name, spec in substudies.items():
        axes = spec.get("axes", {})
        keys = list(axes)
        value_lists = [axes[k] for k in keys]
        for combo in itertools.product(*value_lists):
            override: dict = {}
            for k, v in zip(keys, combo):
                _set_dotted(override, k, v)
            merged = _deep_merge(defaults, override)
            # auto name: substudy + each varying axis's value
            name_bits = [sub_name] + [_short(v) for v in combo]
            merged["name"] = "__".join(name_bits)
            runs.append((sub_name, dict_to_config(merged)))
    return runs


def study_metadata(suite: dict) -> dict[str, dict]:
    """Per-study RQ metadata (everything in a study spec except ``axes``).

    Keyed by sub-study name == the results subdirectory the runner writes to, so
    the reporter can join a run's directory to its research question, hypotheses,
    discriminating subset and declared control. Studies with no metadata (only
    ``axes``) are omitted.
    """
    out: dict[str, dict] = {}
    for name, spec in suite.get("substudies", {}).items():
        meta = {k: v for k, v in (spec or {}).items() if k != "axes"}
        if meta:
            out[name] = meta
    return out


def load_suite(path: str | Path) -> list[tuple[str, RunConfig]]:
    with Path(path).open() as fh:
        suite = yaml.safe_load(fh)
    return expand_suite(suite)


def load_suite_metadata(path: str | Path) -> dict[str, dict]:
    """Load just the per-study RQ metadata from a suite YAML (for the reporter)."""
    with Path(path).open() as fh:
        suite = yaml.safe_load(fh)
    return study_metadata(suite)
