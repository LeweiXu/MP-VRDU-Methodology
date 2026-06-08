# Project Context: MP-VRDU Component Analysis Study

> This file orients any contributor (human or AI coding agent) to the project's
> goals, current state, and key decisions. Keep it updated as the project evolves.
> Last updated: 2026-06-08.

## 1. What this project is

An **honours-year analysis study** on Multi-Page Visually Rich Document
Understanding (MP-VRDU). The goal is to determine **which pipeline components
most affect MP-VRDU performance**, through controlled experiments, and to write
a dissertation reporting the findings.

MP-VRDU = answering questions over documents that span tens to hundreds of pages,
where answer-relevant evidence is sparse, distributed across pages, and often
exceeds the backbone model's context window. A completed survey paper on this
topic exists (see the survey `.tex` / project root) and defines the framing.
The survey's central thesis: **evidence management** (deciding what to retain,
discard, retrieve, or acquire) is the core design problem.

## 2. Objectives

**Primary (required):** A controlled "analysis" study isolating the effect of
individual pipeline components on MP-VRDU accuracy. Produce a passable
dissertation. Treat this as an ablation / factorial study: hold the generator
fixed, vary one component at a time, measure effect on a benchmark.

**Secondary (stretch, not required):** Assemble the best-performing component
combination from the analysis into a proposed method and attempt to beat
existing systems on MMLongBench-Doc and/or LongDocURL.

Timeline: ~5 months remaining.

## 3. Components under study (the experimental factors)

The study isolates pipeline components. Primary and secondary axes, decided
after reviewing the surveyed literature:

**Primary axis — retrieval modality** (the survey's named binding constraint;
literature genuinely conflicts here, which is the opening):
- Sparse / lexical (BM25)
- Dense text (single-vector text embedder, e.g. a SentenceTransformers model)
- Visual late-interaction (ColPali / ColQwen — the MP-VRDU default)
- (Optional) hybrid / joint fusion of the above

**Secondary axis — retrieval granularity or structure-awareness:**
- Granularity: chunk vs page vs section vs element
- Structure-aware (tree/outline, cross-reference following) vs flat chunking

**Swept within every condition (nearly free, produces the best figures):**
- top-k retrieved evidence. NOTE from literature: top-k is NOT monotonic —
  more pages raises hit-rate but can lower F1 / accuracy via attention dilution
  (observed in SimpleDoc). Always report the accuracy-vs-recall-vs-cost curve.

**Focused sub-study (text conditions only, if time allows) — KB representation:**
- raw OCR-with-layout vs Markdown vs HTML serialisation.
- IMPORTANT ENTANGLEMENT: a *visual* retriever (ColPali) never sees the text
  serialisation — it works on page pixels. So KB format only varies meaningfully
  for text-based retrieval conditions. Not all (modality × format) cells are
  valid; mapping which cells are valid is itself a contribution.

## 4. Fixed experimental scaffolding (do NOT vary these between conditions)

- **Generator (reader):** ONE frozen VLM. Develop on `Qwen/Qwen2.5-VL-7B-Instruct`
  (~16.6 GB, Apache-2.0) for engineering velocity; scale to a larger backbone
  (e.g. Qwen2.5-VL-32B) for final reported numbers if compute allows. The
  generator must NOT change between conditions, or differences can't be
  attributed to the retrieval component.
- **Primary benchmark:** MMLongBench-Doc (accuracy + F1). Start here.
- **Secondary benchmark:** LongDocURL (accuracy only). Add after harness works.
- **LLM-as-judge (if used for accuracy scoring):** must be held FIXED across all
  conditions and declared in methodology (reproducibility + contamination risk).

## 5. Required baselines (the analysis lives between these two numbers)

- **No-retrieval lower bound:** feed truncated / first-N pages to the generator.
- **Oracle-retrieval upper bound:** feed the gold evidence pages to the generator.
  This is the single most useful number — it separates *retrieval failure* from
  *generation failure*, and gives the ceiling each retrieval method is chasing.

Every real retrieval method should land between these two. The gap IS the analysis.

## 6. Metrics

- Answer accuracy (benchmark-native).
- Answer F1 (MMLongBench-Doc).
- **Retrieval / evidence-page recall@k** — does the retrieved set contain the gold
  pages? Report this ALONGSIDE accuracy. Answer-only accuracy masks retrieval
  failure; reporting both cleanly is a methodological contribution the survey
  flags as currently unstandardised.
- Cost proxy: tokens fed to generator, wall-clock, number of LLM calls.

Validate each retrieval method against retrieval recall BEFORE looking at
downstream accuracy. A retriever with low recall@k when oracle says the evidence
exists is broken — catch it at the retrieval stage, not after it pollutes the
accuracy table.

## 7. Build order (each step de-risks the next — do not skip)

1. **Eval harness first**, against the no-retrieval baseline. Get ONE benchmark
   loading, ONE frozen generator answering, the official metric computing a
   number. Do not proceed until any honest end-to-end number is produced.
2. **Reproduce one published number** within a reasonable margin (sanity anchor).
3. **Add the oracle upper bound** (feed gold evidence pages).
4. **Implement retrieval methods one at a time**, validating each on retrieval
   recall before downstream accuracy.
5. **Vary KB representation / granularity last**, on the best-working retrieval.

## 8. Known risks / things to watch

- **Contamination:** the survey flags this as a live risk. Several retrieval
  encoders (ColPali/ColQwen) and benchmarks share document pools, and
  MMLongBench-Doc draws documents from DUDE/SlideVQA/ChartQA. Check that neither
  the generator nor the encoder was trained on eval documents; document this
  check in methodology. Supervisor will likely ask.
- **NAME COLLISION:** "MMLongBench-Doc" (the doc-VQA benchmark, arXiv 2407.01523,
  by mayubo2333) is DIFFERENT from "MMLongBench" (arXiv 2505.10610, by
  ZhaoweiWang/EdinburghNLP). We want the former. Don't download the wrong one.
- **top-k non-monotonicity** (see §3) — never assume more retrieval = better.
- **LLM-judge as hidden variable** (see §4) — fix and declare it.

## 9. Key facts about the primary benchmark (MMLongBench-Doc)

- ~1,062 expert-annotated questions over ~130 PDFs, avg ~49 pages / ~21k tokens.
- Answer evidence from 5 sources: text, layout, chart, table, image.
- Question types: single-page (~467), cross-page (~353, the MP-distinctive ones),
  unanswerable (~242 — these test hallucination / abstention).
- Each sample has an `evidence_pages` field (e.g. `[3, 5]`) — this is what makes
  the oracle baseline and retrieval-recall metric possible.
- Official repo: github.com/mayubo2333/MMLongBench-Doc
- HF dataset: `yubo2333/MMLongBench-Doc`
- Also integrated into VLMEvalKit (a possible ready-made harness).

## 10. Compute environment

UWA Kaya HPC cluster, Linux, SLURM scheduler. A100-class GPUs available.
See `kaya_slurm_cheatsheet.md` for usage. Key constraints:
- Never compute on the login (head) node.
- Compute nodes have NO internet — download models/datasets on the LOGIN node
  into `/group/<project>/` first, then run offline with `HF_HUB_OFFLINE=1`.
- Install Conda envs under `/group` (space), not `$HOME`.
