# Project Context: MP-VRDU Component Analysis Study

> Orients any contributor (human or AI coding agent) to the project's goals,
> current state, and decisions. Keep updated as the project evolves.
> Last updated: 2026-06-08.

## 1. What this project is

An **honours-year analysis study** on Multi-Page Visually Rich Document
Understanding (MP-VRDU): answering questions over documents spanning tens to
hundreds of pages, where answer-relevant evidence is sparse, distributed across
pages, and often exceeds the backbone model's context window.

A completed survey paper on this topic exists (survey `.tex` in project root)
and defines the framing. Its central thesis: **evidence management** (deciding
what to retain, discard, retrieve, or acquire) is the core design problem.

This study uses **existing, off-the-shelf tools and techniques** to compare and
analyse which pipeline components most affect MP-VRDU performance. It is an
ablation / controlled-comparison study, NOT a from-scratch model-building effort.

## 2. Goals (primary vs secondary)

### PRIMARY (required for the dissertation)
A controlled analysis isolating the effect of individual pipeline components on
MP-VRDU performance, on the **MMLongBench-Doc** benchmark, producing a passable
dissertation. One variable per sub-study; everything else held fixed.

### SECONDARY (nice-to-have, only if time allows — in rough priority order)
1. Extend to **LongDocURL** (second benchmark, accuracy-only).
2. **Cross-domain generalisation**: repeat key comparisons on other document
   domains — slides (SlideVQA), medical reports, natural-scene/text-rich images,
   text-only documents — to test whether component rankings hold across domains
   or are domain-specific. (The survey notes domain scope is inherited from
   single-page VRDU and underexplored.)
3. **Proposed method**: assemble the best component combination found in the
   analysis and attempt to beat existing systems on MMLongBench-Doc / LongDocURL.

Timeline: ~5 months remaining. Primary goal first; do not start secondary work
until the primary pipeline produces clean, defensible numbers.

## 3. The pipeline and its three stages (the experimental factors)

The study is organised around a three-stage pipeline. Each stage is one axis.

```
PDF --> [Stage 2: representation] --> indexable units --> [Stage 1: retrieval]
        parsers / OCR / chunking                          BM25, ColPali, ...
    --> top-k evidence --> [Stage 3: generation modality] --> generator --> answer
                            image vs text vs both
```

CRITICAL DESIGN POINT — these stages are partly entangled, which dictates which
combinations are valid:
- **Text-based retrievers** (BM25, TF-IDF, dense text) consume the TEXT produced
  by Stage 2, so representation (parser/OCR) fully matters for them.
- **Visual retrievers** (ColPali/ColQwen) consume PAGE IMAGES and never see the
  Stage-2 text output — so parser/OCR choice does NOT apply to visual retrieval.
- **Generation modality is independent of retrieval modality**: you can retrieve
  visually but feed the generator text, or vice versa. Must be controlled, not
  left free, or "ColPali beat BM25" becomes uninterpretable (did visual
  retrieval win, or did feeding images to the generator win?).
- Mapping which (stage x stage) cells are valid is itself a contribution.

## 4. Experimental structure: sub-studies (one variable each)

Run as separate sub-studies sharing ONE harness, not a single giant grid
(grid is too wide and half its cells are invalid).

- **Sub-study A — Retrieval method.** Fix representation AND generation modality.
  Compare retrieval methods. Run as two blocks (generation=image, generation=text)
  so the retrieval comparison is clean within each block.
- **Sub-study B — Document representation.** Text retrievers ONLY (visual
  retrievers don't participate, by definition). Fix the retriever (best from A),
  vary the parser/OCR. Answers "how much does extraction quality matter?"
- **Sub-study C — Generation modality.** Fix retrieval + representation at best
  settings; flip image vs text vs image+text. Isolates the "both" comparison.
- **Cross-cutting sweeps** (within conditions): top-k, chunking/granularity.

## 5. Tools and techniques to test

### STAGE 1 — Retrieval method

PRIMARY (test these):
- **BM25** — sparse lexical baseline. (Lit note: UDA benchmark found BM25 beats
  dense on keyword-heavy financial docs — a real contender, not just a floor.)
- **TF-IDF** — sparse; treat as a second point on the sparse sub-axis with BM25.
- **Dense text (single-vector)** — a SentenceTransformers-class embedder
  (e.g. all-MiniLM, all-mpnet, bge, e5). Pick 1-2.
- **ColPali** — visual late-interaction, the MP-VRDU default (PaliGemma-based).
- **ColQwen2.5** — visual late-interaction (Qwen2.5-VL-based); current default
  in surveyed systems (M3DocRAG, SimpleDoc, MDocAgent).
- **Joint / hybrid** — fuse sparse+dense (e.g. RRF) or text+visual. Implement
  LAST, only once components work individually.

ADDITIONAL (if time allows):
- **grep / exact substring** — not ranked retrieval (returns hits, not top-k);
  include as a dumb FLOOR baseline, not a main condition.
- **ColBERT / ColBERTv2** — text late-interaction (used by MDocAgent's text path).
- **ColNomic-embed-multimodal (3b/7b)** — stronger newer visual late-interaction.
- **ColSmol / ColModernVBERT** — lightweight visual retrievers (250-500M),
  good for an accuracy-vs-cost axis.
- **ColQwen2 (v0.1)** — older version, for a version-sensitivity check.
- **VisRAG** — alternative visual retriever appearing in the surveyed corpus.
- **Single-vector multimodal** (e.g. a CLIP-style or unified vision embedder) —
  contrast against multi-vector late interaction.

### STAGE 2 — Document representation (parser / OCR / format)

PRIMARY (test these):
- **PyMuPDF4LLM** — fast, light, native-text-layer -> Markdown. Good default for
  born-digital PDFs. NOTE: AGPL-3.0 license (matters for code release).
- **MinerU** — layout/table-aware, ML-based; strong on scientific/financial and
  CJK; PaddleOCR backend; -> Markdown/JSON. Apache.
- **Tesseract** — classic OCR (works from rendered images, needed for scanned
  docs; noisier). The OCR comparison point.

ADDITIONAL (if time allows):
- **Docling** (IBM; MIT; layout via DocLayNet, tables via TableFormer; multi-OCR
  backends) — strong enterprise-RAG-oriented parser.
- **Marker** (Apache; Surya OCR; fast; Markdown/JSON/HTML) — "Swiss-army-knife".
- **MarkItDown** (Microsoft; MIT) — versatile file->Markdown.
- **pdfplumber / PyPDF2** — older/simpler text extractors (weaker baselines).
- **olmOCR** — VLM-based OCR.
- **pdf-craft** — specialised for scanned books.
- Format sub-factor: where a tool supports it, compare raw-text vs Markdown vs
  HTML/JSON serialisation (esp. for tables: HTML = complete but verbose;
  Markdown = concise but can't express row/col spans; see DocOwl in project).

### STAGE 3 — Generation modality (what the frozen generator reads)
- **Image** — feed retrieved pages as rendered images.
- **Text** — feed parsed text of retrieved pages (re-invokes Stage 2 even for
  visual retrieval — note in methodology; fix one default parser here).
- **Both** — feed images + text together.

### CROSS-CUTTING (sweep within any condition)
- **top-k** — number of retrieved units. NON-MONOTONIC: more raises hit-rate but
  can lower F1/accuracy via attention dilution (observed in SimpleDoc). Always
  report the accuracy-vs-recall-vs-cost curve.
- **chunking / granularity** — chunk vs page vs section vs element. Interacts
  with Stage 2 (structure-aware parsers enable section-level chunks).

## 6. Fixed scaffolding (do NOT vary between conditions)

- **Generator:** ONE frozen VLM. Develop on `Qwen/Qwen2.5-VL-7B-Instruct`
  (~16.6 GB, Apache-2.0). Scale to larger (e.g. Qwen2.5-VL-32B) for final numbers
  if compute allows. Generator must NOT change between conditions.
- **Primary benchmark:** MMLongBench-Doc (accuracy + F1).
- **LLM-as-judge** (if used for scoring): FIXED across all conditions, declared
  in methodology (reproducibility + contamination risk).

## 7. Required baselines (the analysis lives between these)

- **No-retrieval lower bound:** feed truncated / first-N pages.
- **Oracle-retrieval upper bound:** feed gold evidence pages (`evidence_pages`).
  Separates retrieval failure from generation failure; the ceiling each method
  chases. The single most useful number in the study.

## 8. Metrics

- Answer accuracy (benchmark-native) and F1 (MMLongBench-Doc).
- **Retrieval / evidence-page recall@k** — report ALONGSIDE accuracy. Validate
  each retriever on recall BEFORE looking at downstream accuracy (low recall when
  oracle says evidence exists = broken retriever, catch it early).
- Cost proxy: tokens to generator, wall-clock, number of LLM calls.

## 9. Build order (each step de-risks the next — do not skip)

1. **Eval harness first**, against no-retrieval baseline. One benchmark loading,
   one frozen generator answering, official metric producing a number.
2. **Reproduce one published number** within reasonable margin (sanity anchor).
3. **Add oracle upper bound** (gold evidence pages).
4. **Retrieval methods one at a time** (Sub-study A), recall-validated first.
5. **Representation** (Sub-study B), then **generation modality** (Sub-study C).
6. Only then: secondary goals (LongDocURL, cross-domain, proposed method).

## 10. Known risks / things to watch

- **Contamination:** ColPali/ColQwen encoders and benchmarks share document
  pools; MMLongBench-Doc draws docs from DUDE/SlideVQA/ChartQA. Check generator
  and encoder weren't trained on eval docs; document the check. Supervisor will
  ask.
- **NAME COLLISION:** "MMLongBench-Doc" (doc-VQA, arXiv 2407.01523, mayubo2333 —
  WANT THIS) vs "MMLongBench" (arXiv 2505.10610, EdinburghNLP/ZhaoweiWang — NOT
  this). Don't download the wrong one.
- **AGPL licensing:** PyMuPDF / pymupdf4llm are AGPL-3.0 — note for any code
  release. Prefer Apache/MIT tools (MinerU, Docling, Marker) where it matters.
- **top-k non-monotonicity** (S5) — never assume more retrieval = better.
- **LLM-judge as hidden variable** (S6) — fix and declare.
- **Stage entanglement** (S3) — don't run invalid cells (e.g. ColPali x Tesseract);
  control generation modality so retrieval comparisons stay clean.

## 11. Primary benchmark facts (MMLongBench-Doc)

- ~1,062 expert-annotated questions over ~130 PDFs, avg ~49 pages / ~21k tokens.
- Evidence from 5 sources: text, layout, chart, table, image.
- Question types: single-page (~467), cross-page (~353, MP-distinctive),
  unanswerable (~242, test hallucination/abstention).
- Each sample has `evidence_pages` (e.g. `[3, 5]`) -> enables oracle baseline and
  retrieval-recall metric.
- Scoring is NOT exact-match: a model extracts/normalises answers then compares;
  official results used GPT-4.1 / GPT-4o as judge.
- Repo: github.com/mayubo2333/MMLongBench-Doc ; HF: `yubo2333/MMLongBench-Doc` ;
  also in VLMEvalKit (a possible ready-made harness — read its scoring impl).

## 12. Compute environment

UWA Kaya HPC, Linux, SLURM, A100-class GPUs. See `kaya_slurm_cheatsheet.md`.
- Never compute on the login node.
- Compute nodes have NO internet: download models/data on the LOGIN node into
  `/group/<project>/`, then run offline (`HF_HUB_OFFLINE=1`).
- Install Conda envs under `/group` (space), not `$HOME`.