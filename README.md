# MP-VRDU Component-Analysis Harness

A controlled-comparison study on **Multi-Page Visually Rich Document
Understanding** (MP-VRDU): answering questions over long PDFs where the evidence
is sparse and scattered across pages. This repo is the experiment harness — it
isolates the effect of each pipeline component (retrieval method, document
representation, generation modality) on the **MMLongBench-Doc** benchmark.

- **`context.md`** — the study design (goals, sub-studies, tool inventory).
- **`agent_build_plan.md`** — the staged engineering plan (Stages 0–7).
- This README — how the code is organised and how to run everything.

Everything runs locally for development with a tiny 3B VLM; switching to the
real 7B/32B on Kaya is a one-line change (see [Local vs Kaya](#local-vs-kaya)).

---

## 1. Quick start

```bash
# activate the project venv
source ~/venvs/cits4010-venv/bin/activate          # alias: mpvrdu-venv

# run the test suite (offline, instant)
python -m pytest

# offline smoke run (synthetic data, mocked generator) -> writes a JSONL
python -m mpvrdu.pipeline --config configs/smoke.yaml

# inspect a question + its rendered evidence page
python scripts/inspect_question.py --dataset synthetic --qid c1
```

## 2. Install

Two dependency tiers:

```bash
pip install -r requirements-light.txt   # data layer, sparse+TF-IDF retrieval,
                                         # metrics, analysis, tests (CPU only)
pip install -r requirements-gpu.txt      # + real VLMs, dense + visual retrievers
```

What's already installed in `~/venvs/cits4010-venv` on this box: the full stack
(torch+CUDA, transformers, qwen-vl-utils, sentence-transformers, colpali-engine,
rank-bm25, scikit-learn, pytesseract, matplotlib). `magic-pdf` (MinerU) is **not**
installed — it's a heavy optional parser; configs that request it fail-soft.

## 3. How it works (architecture)

One run = one **config** → one **JSONL** of per-question results. The pipeline
(`mpvrdu/pipeline.py`) wires four swappable stages:

```
DataConfig ──load_dataset──► Dataset  (Questions + Documents)
                                  │
   PHASE 1 (retrieval)            ▼   for each question:
   EvidenceSelector.select(q, doc) ─► ranked 0-based PAGE indices   [retrieve/]
        none · oracle · bm25 · tfidf · dense · colpali · colqwen · hybrid
                                  │
            selector.unload()  ◄──┘  free the retriever's GPU model
                                  │
   PHASE 2 (generation)           ▼
   InputBuilder.build(...) ─► images and/or text                 [generate/, represent/]
   Generator.answer(q, images, text) ─► prediction              [generate/]
        mock · local_small_vlm (Qwen2.5-VL-3B) · kaya_vlm (7B/32B)
                                  │
   Judge.score(pred, gold, fmt) ──► correct / abstained          [eval/]
        rule (deterministic) · llm (stub)
                                  │
   ResultsWriter ────────────────▼  results/<name>__<confighash>__<ts>.jsonl
```

**Two-phase execution** (key for the 12GB box): all retrieval selections are
computed first (visual-retriever model resident), then that model is freed, then
the generator loads. So only **one** large model is in GPU memory at a time, and
a ColPali+VLM run fits in 12GB. Phase 1 visits questions in document order so the
per-document index is built once and reused.

### Module map

| Path | Stage | Responsibility |
|------|-------|----------------|
| `mpvrdu/config.py` | 0 | `RunConfig` dataclass schema, YAML loader, validation, config hash |
| `mpvrdu/env.py` | 0 | repo-local cache config (`HF_HOME`/`TORCH_HOME`) |
| `mpvrdu/experiment.py` | – | expand a grid suite (axes) into many `RunConfig`s |
| `mpvrdu/results.py` | 0 | JSONL writer/reader; filenames encode the config hash |
| `mpvrdu/logging_utils.py` | 0 | logger + `set_seed` |
| `mpvrdu/pipeline.py` | all | wires stages per config; two-phase; emits JSONL |
| `mpvrdu/data/dataset.py` | 1 | `Question`/`Document`/`Dataset`, parquet + JSON loaders |
| `mpvrdu/data/render.py` | 1 | PDF → cached per-page PNG (PyMuPDF), 0-based |
| `mpvrdu/data/slice.py` | 1 | dev-slice carving |
| `mpvrdu/data/synthetic.py` | 1 | offline synthetic fixture (tests / smoke) |
| `mpvrdu/data/load.py` | 1 | `DataConfig` → `Dataset` dispatch |
| `mpvrdu/represent/` | 2/5 | parsers (PyMuPDF4LLM/MinerU/Tesseract) + chunking |
| `mpvrdu/retrieve/` | 1/4 | selectors (none/oracle) + retrievers + hybrid + recall eval; Tier-1 post-processing (`postprocess.py` adaptive-k/expansion, `rerank.py`) |
| `mpvrdu/generate/` | 3 | input builder + mock/local/Kaya VLM generators |
| `mpvrdu/eval/` | 2 | answer normalisation, metrics (acc/F1/recall@k), rule + LLM judge |
| `mpvrdu/analysis/` | 7 | aggregate JSONL → tables, CIs, oracle-gap, figures, report |
| `mpvrdu/data/audit.py` | – | contamination-audit fingerprints (context.md §10) |
| `mpvrdu/experiment.py` | – | expand an experiment suite into many `RunConfig`s |

### Key concepts

- **Config-driven.** A `RunConfig` fully specifies a condition (data slice,
  representation, retrieval, generation, judge, seed). Two configs that differ in
  any field get different hashes → different result files. See §6 for all fields.
- **Stage entanglement is enforced.** Visual retrievers (ColPali/ColQwen) consume
  **page images** and ignore the text parser (`Retriever.modality = "visual"`).
  Text retrievers consume the parser's text. Generation modality is independent
  of retrieval modality — you can retrieve visually and feed the generator text.
- **Page indexing.** `evidence_pages` from the dataset are **1-based**; the
  renderer is 0-based. Conversion happens in exactly one place
  (`Question.evidence_pages_zero_based`). Recall is always computed page-based,
  even when chunking is sub-page (each retrieval `Unit` records its source page).
- **Unanswerable** is determined by the **answer** ("Not answerable" /
  `answer_format: None`), NOT by empty evidence pages (real data has both
  answerable-with-no-evidence and unanswerable-with-evidence cases).

## 4. Getting the data

MMLongBench-Doc ships as one HF parquet (**1091 questions** — the benchmark is
evaluation-only) plus ~662 MB of PDFs.

```bash
# (a) SMALL subset — only a few small PDFs (<1 MB). Great for the 12GB box.
python scripts/download_subset.py --out data/mmlongbench_subset --docs 3

# (b) FULL benchmark — all 1091 questions + all 135 PDFs (662 MB).
python scripts/download_data.py --out data/mmlongbench
```

Both produce a dir with `samples.json`/parquet + `documents/`. Point a config at
it with `data: {name: mmlongbench-doc, slice: <full | path>}`. `data/` and
`results/` are git-ignored.

## 5. Running things

### A single condition
```bash
python -m mpvrdu.pipeline --config configs/oracle.yaml
python -m mpvrdu.pipeline --config configs/local_subset_tiny_vlm.yaml   # real 3B VLM
```

### Retrieval recall only (fast, no generator)
Validate a retriever on recall@k before spending compute on generation:
```bash
python scripts/eval_retrieval.py --config configs/subA/bm25_image.yaml --ks 1 2 4 8
```

### Inspect what a run produced (selected pages vs gold)
After a run, eyeball its per-question output — prediction vs gold, and the
**selected pages vs the gold evidence pages** — to see whether a wrong answer was
a *retrieval* miss (evidence never selected) or a *generation* miss (evidence
selected, answered wrong). The footer tallies both failure modes.
```bash
python scripts/inspect_run.py results/grid/A_retrieval/<run>.jsonl            # all questions
python scripts/inspect_run.py <run>.jsonl --errors-only                       # only mistakes
python scripts/inspect_run.py <run>.jsonl --retrieval-misses                  # gold page not selected
python scripts/inspect_run.py <run>.jsonl --qid <qid> --full                  # one question, untruncated
python scripts/inspect_run.py results/grid                                    # list runs to inspect
```

### The full grid (all combinations)
The grid is defined once in `experiments/grid_local_3b.yaml` and run by
`scripts/run_grid.py`. It expands the suite into ~30 runs across the sub-studies
and executes them sequentially. **It is resumable** — completed runs are skipped,
so you can stop/restart and chip away (a full local grid is many GPU-hours).

```bash
# preview the run list without executing
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --dry-run

# run the whole grid locally with the 3B
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml

# quick machinery check: a few questions per run
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --max-questions 4

# only one sub-study
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --only A_retrieval

# run on the small subset instead of the full benchmark
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml \
    --slice data/mmlongbench_subset
```

Results land in `results/grid/<substudy>/<name>__<hash>.jsonl` and a
`results/grid/summary.md` table is (re)written at the end.

### Consolidate results into tables (Stage 7)
```bash
python scripts/analyze.py --results results/grid --out results/grid/summary.md
```
Tables are a pure function of the JSONL files (each embeds its own config), so the
results section regenerates from raw outputs with one command.

## 6. The sub-studies (one variable each)

Defined as `substudies` in the grid suite; each is an axes cross-product.

| Sub-study | Varies | Held fixed |
|-----------|--------|-----------|
| **baselines** | retrieval ∈ {none, oracle} × modality | everything else — the floor & ceiling |
| **A_retrieval** | retrieval method × modality (k=4) | representation (PyMuPDF4LLM), generator |
| **A_topk** | top-k ∈ {1,2,4,8} on bm25 + colpali | method/modality |
| **B_representation** | parser ∈ {pymupdf4llm, mineru, tesseract} | retriever (dense), modality (text) |
| **C_modality** | modality ∈ {image, text, both} | retrieval (ColPali), representation |

Sub-study B uses **text retrievers only** (visual retrievers don't consume the
parser). For a visual retriever in text/both modality (sub-study C), the
retrieved pages are parsed by the fixed default parser, stated in the config.

## 7. Local vs Kaya

The grid suite's `defaults.generation` block is the **single switch**. To run the
identical grid on Kaya with the real model:

```yaml
# experiments/grid_local_3b.yaml -> defaults.generation
generator: kaya_vlm                       # was: local_small_vlm
model_id:  Qwen/Qwen2.5-VL-7B-Instruct    # was: Qwen/Qwen2.5-VL-3B-Instruct
# you can also raise/remove max_pixels (more GPU on Kaya)
```

Or override at the command line without editing the file:
```bash
python scripts/run_grid.py --suite experiments/grid_local_3b.yaml \
    --generator kaya_vlm --model-id Qwen/Qwen2.5-VL-7B-Instruct
```
Everything else — retrievers, parsers, modalities, top-k, scoring — is identical,
so a result validated locally is the same pipeline at scale. For the real headline
numbers, also switch the judge from `rule` to a fixed `llm` judge and declare it.

## 7b. Where downloads go (cache)

All model weights and dataset caches go into a **repo-local** `.cache/`
(git-ignored), not `$HOME` — configured by `mpvrdu/env.py` which sets `HF_HOME`
and `TORCH_HOME` on import. Override the location with `MPVRDU_CACHE`, or set
`HF_HOME` directly (respected as-is — this is how Kaya points at `/group`).
The dataset PDFs live under `data/` (also git-ignored).

| What | Default location | Override |
|------|------------------|----------|
| HF models + dataset cache | `.cache/huggingface` | `HF_HOME` or `MPVRDU_CACHE` |
| torch hub | `.cache/torch` | `TORCH_HOME` or `MPVRDU_CACHE` |
| dataset PDFs | `data/mmlongbench[_subset]` | `--out` / `MPVRDU_MMLB_DIR` |
| page-render cache | `data/renders` | `MPVRDU_RENDER_CACHE` |

## 7c. Running on Kaya (HPC, SLURM)

The code is Kaya-ready: it respects `HF_HOME`/`MPVRDU_*` env vars, loads models
fully offline, and the grid runner is resumable. Scripts live in `scripts/kaya/`.
See `kaya_cheatsheet.md` for cluster mechanics.

**One-time setup (LOGIN node — it has internet):**
```bash
# 1. point the scripts at your project: edit scripts/kaya/env.sh
#    (set MPVRDU_GROUP=/group/<yourproject>, MPVRDU_CUDA=cuda/<version>)
# 2. create the conda env under /group + install deps
bash scripts/kaya/setup_conda_env.sh
# 3. pre-download the dataset + all model weights into the /group HF cache
bash scripts/kaya/prestage.sh
mkdir -p logs
```

**Smoke test then run (COMPUTE node, via SLURM):**
```bash
sbatch scripts/kaya/gpu_test.sbatch        # confirms GPU+CUDA before any model
sbatch scripts/kaya/run_grid.sbatch        # full grid with the real 7B (offline)
sbatch scripts/kaya/run_config.sbatch configs/oracle.yaml   # a single condition
squeue -u $USER                            # watch it
```

`run_grid.sbatch` runs the full `experiments/grid_kaya_7b.yaml` suite (64
conditions) on the real 7B (set `MPVRDU_MODEL` for 32B) and writes the analysis
report at the end. The local `grid_local_3b.yaml` is the same design at smaller
scale — a grid validated locally with the 3B runs unchanged at scale; the only
difference is that one flag. Compute nodes have no internet; the sbatch scripts
set `HF_HUB_OFFLINE=1` and load everything from the `/group` cache staged in
step 3. Results are written under `$MPVRDU_RESULTS` (on `/group`).

## 8. Config reference

```yaml
name: my-run                 # used in the results filename
seed: 0
data:
  name: mmlongbench-doc      # or "synthetic"
  slice: full                # "dev" | "full" | path to a dataset dir
  max_questions: null        # cap (e.g. for smoke runs)
representation:
  parser: pymupdf4llm        # pymupdf4llm | pymupdf | mineru | tesseract | none
  chunking: page             # page | chunk | section
  dpi: 144                   # page-render resolution
  text_format: markdown
retrieval:
  method: bm25               # none | oracle | grep | bm25 | tfidf | dense | colpali | colqwen | hybrid
  top_k: 4
  no_retrieval_pages: 10     # N for the first-N no-retrieval baseline
  embedding_model: null      # dense text encoder
  visual_model: null         # ColPali/ColQwen checkpoint
  hybrid_methods: [bm25, dense]
  rrf_k: 60
  # --- Tier-1 post-processing (wraps ANY retriever; docs/corpus_techniques.md) ---
  k_strategy: fixed          # fixed | gmm | kmeans   (adaptive top-k, #1)
  candidate_k: 0             # pool depth for adaptive cut (0 -> auto)
  rerank: none               # none | llm             (retrieve->rerank->read, #3)
  rerank_candidates: 0       # pages to rerank (0 -> auto)
  rerank_model: null         # LLM checkpoint for the reranker
  expand: none               # none | parent_page | parent_section | adjacent (#4/#5)
  expand_window: 1           # adjacent: pages on each side
generation:
  modality: image            # image | text | both
  generator: local_small_vlm # mock | local_small_vlm | kaya_vlm
  model_id: Qwen/Qwen2.5-VL-3B-Instruct
  max_new_tokens: 128
  temperature: 0.0
  max_pixels: 602112         # cap vision tokens/page (fit 12GB); null = model default
  load_in_4bit: false
judge:
  type: rule                 # rule | llm
  model_id: null             # required when type: llm (the fixed judge model)
```

`data.name` also accepts `synthetic` (offline fixture) and `longdocurl` (the
secondary benchmark, S8 — same loader interface, accuracy-only). Validation is
strict: unknown keys and invalid enum values raise `ConfigError`.

## 9. Metrics & analysis

Per-question metrics:
- **Accuracy** — mean per-question correctness via the `answer_format`-aware rule
  comparison (Int/Float/Str/List/None), abstention handled explicitly
  (unanswerable questions are correct iff the model abstains).
- **F1** — binary F1 of the *answerability* decision (answerable vs abstained).
- **recall@k** — fraction of gold evidence pages in the top-k retrieved pages.
- **cost** — per question, the generator logs `input_tokens`, `output_tokens`,
  `n_images`, `gen_seconds`, `n_llm_calls` (context.md §8 cost proxy).

Aggregate analysis (`mpvrdu/analysis/`, surfaced by `scripts/report.py`). The
report (`report.md`) is comprehensive:
- **per-condition table** + **one table per sub-study** (grouped by results subdir).
- **bootstrap 95% CIs** on accuracy + a **paired significance test** between two
  runs (`paired_diff_test`) — so method gaps come with intervals, not just points.
- **oracle-gap decomposition** — per method, lift over the no-retrieval floor and
  gap to the oracle ceiling (matched per modality).
- **accuracy by question type** (single / cross / unanswerable) and **by gold
  evidence source** (text / layout / chart / table / image).
- **abstention behaviour** — hallucination (answered an unanswerable) vs
  over-abstention (abstained on an answerable), per condition.
- **recall → accuracy correlation** — does better retrieval actually help answers?
- **Tier-1 post-processing breakdown** — adaptive-k / rerank / expansion runs with
  accuracy, recall@k, and **mean pages fed** (variable under adaptive-k/expansion).
- **cost** — mean tokens to the generator + wall-clock per condition.
- **sanity checks** — flags e.g. oracle < no-retrieval, or recall≈0 retrievers.
- **figures** — top-k accuracy/recall curves, method bars with floor/ceiling.

```bash
python scripts/report.py --results results/grid          # -> report.md + figures/
python scripts/contamination_audit.py --slice full       # fingerprint eval docs (§10)
```

### Judge

The deterministic **rule judge** (`type: rule`) needs no model and is used for all
local plumbing. For headline numbers use the **LLM judge** (`type: llm`), which
mirrors MMLongBench-Doc's official scoring — an LLM *extracts* the concise answer,
then the same rule comparison scores it. Enable per run:

```yaml
judge: {type: llm, model_id: Qwen/Qwen2.5-7B-Instruct}   # fixed + declared
```

The judge model is loaded offline from the HF cache (stage it on Kaya like any
other model). It is a fixed, declared experimental variable (context.md §6,§10).

## 10. Gotchas / constraints

- **12GB VRAM.** Many-page image inputs (no-retrieval) can OOM; the `max_pixels`
  cap bounds vision tokens per page. Two-phase keeps one model resident at a time.
- **Tiny VLM ≠ reportable.** The local 3B is a code-path stand-in for the 7B/32B
  on Kaya. Never report numbers from the 3B (or from mock / synthetic).
- **MinerU** needs `magic-pdf` (heavy, not installed) — its configs fail-soft.
  Tesseract works locally (needs the `tesseract` binary, present here).
- **PyMuPDF is AGPL-3.0** — note for any code release; prefer MinerU (Apache) where
  licensing matters.
- The HF dataset's recorded sizes for 2 PDFs are stale; `download_subset.py` /
  `download_data.py` handle the common files, and a direct fetch was used for the
  2 mismatched ones.

## 11. Repository tree

```
mpvrdu/            the package (see module map in §3)
configs/           hand-written single-condition configs (smoke, baselines, subA/B/C)
experiments/       grid suites (grid_local_3b.yaml = the full local grid)
scripts/           CLI entry points (download, build slice, inspect_question,
                   inspect_run, eval_retrieval, run_grid, report)
tests/             pytest suite (offline, synthetic fixture)
results/           JSONL outputs (git-ignored)
data/              dataset + render cache (git-ignored)
context.md         study design   ·   agent_build_plan.md   staged plan
```
