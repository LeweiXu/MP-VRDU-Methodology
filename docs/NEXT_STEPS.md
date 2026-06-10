# Research roadmap & methodological review

A standing review of what's built, what's still fundamentally missing, and which
extensions are worth the compute. Companion to `context.md` (study design) and
`agent_build_plan.md` (staged plan).

## Built (harness is dissertation-grade for these)

- Full pipeline + config system + resumable grid runner (local 3B / Kaya 7B-32B
  via one switch), 116 passing tests.
- Baselines (no-retrieval floor, oracle ceiling); retrievers BM25, TF-IDF, dense,
  ColPali, ColQwen, hybrid (RRF), plus a grep dumb-floor.
- Representation: PyMuPDF4LLM, Tesseract (MinerU optional/heavy); chunking
  page/chunk/section.
- Generation modality image/text/both; two-phase execution to fit 12GB.
- **Metrics:** accuracy, answerability-F1, recall@k, **per-question cost**
  (tokens/latency/calls).
- **Judge:** deterministic rule judge + a faithful **LLM judge** (LLM extracts →
  rule compares), pluggable backend.
- **Analysis:** bootstrap CIs, paired significance + pairwise method matrix,
  oracle-gap decomposition, recall→accuracy correlation, hallucination/
  over-abstention rates, seed variance, sanity checks, auto-report + figures
  (top-k curves, method bars w/ floor+ceiling, efficiency frontier).
- **Audit:** contamination fingerprinting of eval docs.
- Second-dataset interface (alias-tolerant loader; LongDocURL dispatch).

## Still fundamentally missing (do before headline claims)

1. **Reproduce a published MMLongBench-Doc number** (context.md §9 step 2) — the
   credibility anchor. Needs the real 7B + the LLM judge on the full set; check a
   known condition lands within a reasonable margin. Depends on #2.
2. **Run the LLM judge for real** — it's implemented + tested with a fake backend,
   but never run against a real judge model. Stage e.g. Qwen2.5-7B-Instruct on
   Kaya, run key conditions with `judge: {type: llm, model_id: ...}`, and report
   rule-vs-LLM-judge deltas (shows how much the scorer matters).
3. **Run MinerU on Kaya** — the current MinerU parser has passed a real local
   GPU parse. Install its separate environment and stage its models on Kaya as
   documented in `docs/kaya_cheatsheet.md` to complete sub-study B at scale.

## High-value extensions (ranked)

1. **Generator-size scaling** (3B → 7B → 32B): one `--model-id` axis. Do the
   component rankings hold across model size? Cheap, very citable.
2. **Oracle-gap as the headline framing** — already computed; lead the results
   with "retrieval recovers X% of the no-retrieval→oracle gap" per method.
3. **Abstention/hallucination vs top-k** on the 244 unanswerable Qs — does more
   retrieved context induce more false answers? (rates already logged.)
4. **LongDocURL** (S8): stage data, verify the loader's field aliases, re-run key
   conditions (accuracy-only).
5. **Cross-domain** (S9): SlideVQA / medical / scene-text — does the ranking hold?
   Note parser availability shifts (scans force OCR).
6. **Proposed method** (S10): assemble best (repr × retrieval × modality × k) and
   benchmark vs published systems.

## Design caveats to state in the methodology

- **Greedy "fix-best-then-vary"** can miss component interactions (parser×retriever,
  modality×retriever). Add one confirmatory run at the chosen best combination.
- **image-vs-text confounds:** image accuracy depends on dpi/max_pixels (vision
  tokens), text on parser quality. Hold fixed and report the settings.
- **Retrieval granularity:** visual retrievers score whole pages; text retrievers
  can use sub-page chunks. "ColPali vs BM25" partly conflates this — control via
  the chunking axis and note it.
- **Judge is a hidden variable** — fix it, declare it, report rule-vs-LLM deltas.
