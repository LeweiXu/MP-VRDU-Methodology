#!/usr/bin/env python3
"""Run a whole experiment grid (many configs) sequentially. RESUMABLE.

Expands a suite YAML into runs, then executes each as a normal pipeline run with
a deterministic output path (results/grid/<substudy>/<name>__<hash>.jsonl). A run
is skipped if its output already exists and is complete, so you can stop/restart
freely and chip away at a long local grid.

    # local 3B (as configured in the suite):
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml
    # preview what would run, without running:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --dry-run
    # quick machinery check: only a few questions per run
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --max-questions 4
    # only one sub-study:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --only A_retrieval
    # later, override the generator for Kaya without editing the suite:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml \
        --generator kaya_vlm --model-id Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import aggregate_dir, to_markdown_table
from mpvrdu.data.load import load_dataset
from mpvrdu.experiment import load_suite
from mpvrdu.logging_utils import add_file_logging, file_logging, get_logger
from mpvrdu.pipeline import run
from mpvrdu.results import read_rows

log = get_logger("run_grid")


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:        # ImportError, or a CUDA error after an OOM
        pass


def _apply_overrides(cfg, args):
    if args.slice:
        cfg.data.slice = args.slice
    if args.max_questions is not None:
        cfg.data.max_questions = args.max_questions
    if args.generator:
        cfg.generation.generator = args.generator
    if args.model_id:
        cfg.generation.model_id = args.model_id
    if args.load_in_4bit:
        cfg.generation.load_in_4bit = True
    return cfg.validate()


def _out_path(root: Path, substudy: str, cfg) -> Path:
    return root / substudy / f"{cfg.name}__{cfg.hash()}.jsonl"


def _run_log_path(logs_root: Path, suite_name: str, substudy: str, cfg) -> Path:
    return logs_root / suite_name / substudy / f"{cfg.name}__{cfg.hash()}.log"


def _write_run_header(fh, cfg, path: Path) -> None:
    fh.write(f"run_name: {cfg.name}\n")
    fh.write(f"config_hash: {cfg.hash()}\n")
    fh.write(f"results_path: {path}\n")
    fh.write("config:\n")
    yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    fh.write("\n--- process output ---\n")
    fh.flush()


def _run_isolated(cfg, path: Path, run_log: Path) -> bool:
    """Run one config in a FRESH subprocess so a hard CUDA OOM (which corrupts the
    process's CUDA context) can't cascade into every later run. Each run gets a
    clean GPU; the trade-off is reloading the dataset per run (cheap on a subset).
    Returns True on success (exit 0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="gridcfg_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg.to_dict(), f)
        with run_log.open("w", encoding="utf-8") as fh:
            _write_run_header(fh, cfg, path)
            proc = subprocess.run(
                [sys.executable, "-m", "mpvrdu.pipeline", "--config", tmp,
                 "--out", str(path), "--no-file-log"], env=os.environ.copy(),
                stdout=fh, stderr=subprocess.STDOUT)
            fh.write(f"\n--- exit code: {proc.returncode} ---\n")
        return proc.returncode == 0
    finally:
        os.unlink(tmp)


def _run_in_process(cfg, dataset, path: Path, run_log: Path) -> None:
    """Run with structured logs plus third-party stdout/stderr persisted."""
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("w", encoding="utf-8") as fh:
        _write_run_header(fh, cfg, path)
    with run_log.open("a", encoding="utf-8") as fh:
        with file_logging(run_log), redirect_stdout(fh), redirect_stderr(fh):
            run(cfg, dataset=dataset, out_path=path)


def _is_complete(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(read_rows(path)) >= expected_n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--results-root", default="results/grid")
    ap.add_argument("--logs-root", default="logs",
                    help="grid and per-run logs root (default: logs)")
    ap.add_argument("--only", action="append", default=None,
                    help="run only these substudies (repeatable)")
    ap.add_argument("--slice", default=None, help="override data.slice for all runs")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--generator", default=None, help="override generator (e.g. kaya_vlm)")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run each condition once per seed (variance estimate)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore existing results")
    ap.add_argument("--isolate", action="store_true",
                    help="run each config in its own subprocess (OOM-isolated; "
                         "a clean CUDA context per run). Recommended on one GPU.")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.results_root)
    logs_root = Path(args.logs_root)
    suite_name = Path(args.suite).stem
    stamp = time.strftime("%Y%m%d-%H%M%S")
    grid_log = logs_root / f"{suite_name}__{stamp}.log"
    add_file_logging(grid_log, mode="w")
    log.info("grid log: %s", grid_log)

    if not args.aggregate_only:
        runs = load_suite(args.suite)
        if args.only:
            runs = [(s, c) for s, c in runs if s in set(args.only)]
        for _, cfg in runs:
            _apply_overrides(cfg, args)
        if args.seeds:
            import copy
            expanded = []
            for sub, cfg in runs:
                for s in args.seeds:
                    c = copy.deepcopy(cfg)
                    c.seed = s
                    expanded.append((sub, c.validate()))
            runs = expanded

        log.info("suite expands to %d runs across %d sub-studies",
                 len(runs), len({s for s, _ in runs}))

        # dataset cache keyed by data signature (avoid reloading 1091 q each run)
        ds_cache: dict = {}

        def get_ds(cfg):
            key = (cfg.data.name, cfg.data.slice, cfg.data.split, cfg.data.max_questions)
            if key not in ds_cache:
                ds_cache[key] = load_dataset(cfg.data)
            return ds_cache[key]

        if args.dry_run:
            for sub, cfg in runs:
                path = _out_path(root, sub, cfg)
                state = "done" if _is_complete(path, 1) else "TODO"
                print(f"[{state:4s}] {sub:16s} {cfg.name:45s} {cfg.hash()}")
            print(f"\n{len(runs)} runs total. (dry-run; nothing executed)")
            return

        done = failed = skipped = 0
        for idx, (sub, cfg) in enumerate(runs, 1):
            path = _out_path(root, sub, cfg)
            run_log = _run_log_path(logs_root, suite_name, sub, cfg)
            ds = get_ds(cfg)
            n = len(ds)
            if not args.rerun and _is_complete(path, n):
                log.info("[%d/%d] SKIP (done): %s", idx, len(runs), cfg.name)
                skipped += 1
                continue
            log.info("[%d/%d] RUN %s/%s (n=%d) log=%s",
                     idx, len(runs), sub, cfg.name, n, run_log)
            try:
                if args.isolate:
                    if _run_isolated(cfg, path, run_log):
                        done += 1
                    else:
                        failed += 1
                        log.error("FAILED (subprocess) %s/%s; see %s",
                                  sub, cfg.name, run_log)
                else:
                    _run_in_process(cfg, ds, path, run_log)
                    done += 1
            except Exception:
                failed += 1
                tb = traceback.format_exc()
                run_log.parent.mkdir(parents=True, exist_ok=True)
                with run_log.open("a", encoding="utf-8") as fh:
                    fh.write("\n--- parent exception ---\n")
                    fh.write(tb)
                log.error("FAILED %s/%s; see %s\n%s",
                          sub, cfg.name, run_log, tb)
            finally:
                _free_cuda()
        log.info("grid finished: %d ran, %d skipped, %d failed", done, skipped, failed)

    # ---- aggregate everything under the results root ----
    summaries = aggregate_dir(root, pattern="**/*.jsonl")
    if summaries:
        table = to_markdown_table(summaries)
        out_md = root / "summary.md"
        out_md.write_text(table + "\n", encoding="utf-8")
        print("\n" + table)
        print(f"\n-> wrote {out_md}")
    else:
        print("no completed results to aggregate yet.")


if __name__ == "__main__":
    main()
