# Repository Pivot Guide: from Component-Swapping to Mechanism Analysis

> FOR THE CODING AGENT. This document steers the EXISTING repository
> (see README) toward the research-question framing in
> `docs/research_questions.md` (RQ spec). Read the RQ spec FIRST; this guide is
> the migration plan that reconciles the current code with it.
> Last updated: 2026-06-08.

## 0. What this pivot IS and IS NOT

The study's framing changed from **products** ("ColPali vs MinerU") to
**mechanisms** ("similarity vs relation-aware retrieval", "sequential vs parallel
reasoning"). See RQ spec §0. The harness ALREADY implements most of the
machinery; what is missing is the mechanism FRAMING and two whole axes.

**This pivot IS:**
- Relabelling existing sub-studies (A/B/C) as RQ-framed studies (RQ1–RQ4).
- Adding subset-stratified, hypothesis-tested REPORTING (the RQ spec §4 bar).
- Building TWO genuinely new axes: RQ3 (reasoning) and RQ4 (coarse-to-fine).
- Making the reference pipeline (RQ spec §1) an explicit, named control.

**This pivot IS NOT:**
- A rewrite. Do NOT delete working retrievers, parsers, post-processing, metrics,
  two-phase execution, the config system, the grid runner, or the 116 tests.
- A change to the harness's mechanics. The config-driven, one-JSONL-per-run,
  resumable-grid design is correct and stays.
- License to break the 12GB/two-phase constraints or the offline-Kaya path.

**PRIME DIRECTIVE — do not over-correct.** Most existing code is KEPT and
RELABELLED, not removed. When in doubt, preserve and wrap rather than rewrite.
Every existing test must still pass after the pivot (plus new tests for new axes).

## 1. Keep / Reframe / Build — the master map

| Repo element (README) | Verdict | Action |
|-----------------------|---------|--------|
| config system, hashing, JSONL, grid runner, two-phase | **KEEP** | untouched |
| retrievers: bm25, tfidf, dense, colpali, colqwen, hybrid | **KEEP** | become RQ1 instantiations |
| Tier-1 post-proc: adaptive-k, expansion, rerank | **KEEP** | split across RQ1 (expansion=relation-aware) & RQ4 (rerank) |
| parsers: pymupdf4llm, mineru, tesseract; chunking | **KEEP** | become RQ2 instantiations |
| generation modality image/text/both | **KEEP** | demote to cross-cutting (RQ spec §3), not a principal RQ |
| metrics: acc, F1, recall@k, cost | **KEEP** | untouched |
| analysis: subset breakdowns, oracle-gap, CIs, paired tests | **KEEP** | the RQ-reporting backbone already exists |
| sub-study **A_retrieval / A_topk** | **REFRAME** | -> RQ1 (add relation-aware level + per-subset hypotheses) |
| sub-study **B_representation** | **REFRAME** | -> RQ2 (add structure-chunking contrast + hypotheses) |
| sub-study **C_modality** | **REFRAME** | -> cross-cutting axis, not a principal study |
| sub-study **baselines** (none/oracle) | **KEEP** | the RQ1 control + universal floor/ceiling |
| **reasoning axis** | **BUILD** | RQ3 — does not exist yet |
| **coarse-to-fine narrowing** | **BUILD** | RQ4 — partially (rerank exists), assemble & frame |
| reference-pipeline as named control | **BUILD** | one canonical config all RQs perturb |

## 2. Step-by-step migration (ordered; each step ends green)

### Step 1 — Establish the reference pipeline as a first-class object
The RQ spec §1 defines ONE control config every RQ perturbs. Make it concrete.
- Add `configs/reference.yaml`: dense text retriever (pinned embedder), one
  pinned default parser, plain CoT prompt, fixed modality, fixed LLM judge,
  Qwen2.5-VL-7B (3B locally). This is THE control.
- Document in the README that non-tested components are pinned to `reference.yaml`
  values in every study.
- **Test:** `reference.yaml` validates and runs end-to-end on the dev slice.

### Step 2 — Reframe sub-studies into RQ studies (mostly renaming + reporting)
Rename the grid `substudies` keys and add the hypothesis metadata. The axes
themselves barely change; the FRAMING and REPORTING do.
- `A_retrieval` + `A_topk` -> `RQ1_retrieval`. Add the relation-aware level
  (expansion + traversal, see Step 4) alongside the existing similarity levels.
- `B_representation` -> `RQ2_representation`. Add the flat-vs-structure chunking
  contrast as the headline (RQ spec H2b).
- `C_modality` -> move under a `crosscutting` group (RQ spec §3).
- Keep `baselines`; it is RQ1's control (no-retrieval floor + oracle ceiling).
- In the suite, attach to each study: its RQ id, the hypotheses, the
  discriminating subset(s), and the declared control. (A metadata block per
  study; the runner ignores it, the reporter consumes it.)
- **Test:** grid `--dry-run` lists RQ-named studies; existing runs still expand.

### Step 3 — Upgrade reporting to the RQ §4 bar (this is where "analysis" lives)
The analysis module already computes subset breakdowns, oracle-gap, CIs, paired
tests. Wire them into a per-RQ report that ENFORCES the §4 checklist.
- For each RQ, the report must show: the hypothesis; the PER-SUBSET result on the
  discriminating subset (not just the grand mean); the effect vs the declared
  control with a CI/paired test; and a HELD / REFUTED / MIXED verdict per
  hypothesis.
- Add a `report_rqN.md` section generator per RQ that pulls the right subset axis
  (e.g. RQ1 -> by evidence-source AND single/cross-page; RQ3 -> single/multi-hop).
- A predicted NULL (e.g. H3c ToT) must render as a first-class reported finding,
  not an empty cell.
- **Test:** report renders a hypothesis verdict table for a synthetic RQ1 run;
  per-subset columns are populated.

### Step 4 — RQ1: add the relation-aware LEVEL (the survey's central split)
RQ1's similarity side exists (bm25/tfidf/dense/colpali/colqwen/fusion). The
RELATION-AWARE side is under-built — it is currently only Tier-1 expansion.
- Promote `expand: parent_page|parent_section|adjacent` to a labelled
  relation-aware CONDITION in RQ1 (not just a post-proc toggle): these test the
  cheap end of relation-aware retrieval.
- Add the structural-traversal condition (tree/section navigation) as the richer
  relation-aware level. Reuse parser-derived structure (shared with RQ2). Keep
  any traversal budget FIXED for comparability (RQ spec RQ1).
- These target H1b (relation-aware wins on multi-hop / low-similarity-page
  questions; ~zero on single-page). Ensure the reporter slices by hop/cross-page.
- **Test:** a relation-aware condition selects pages a pure-similarity run misses
  on a constructed multi-hop dev-slice item; recall measured per subset.

### Step 5 — RQ3: BUILD the reasoning axis (new; does not exist)
Generation is currently "answer the question". Add a reasoning-strategy axis over
a FIXED evidence buffer (so the delta is reasoning alone — RQ spec RQ3 control).
- New config field `generation.reasoning: direct | cot | self_reflection |
  self_consistency | tot`. (Add to schema + validation.)
- Implement in `generate/` as prompt-template + decode-strategy variants:
  - `direct` — current behaviour (the control).
  - `cot` — single-pass chain-of-thought, parse final answer.
  - `self_reflection` — CoT + one revision pass against a critic prompt.
  - `self_consistency` — sample N (temp>0), aggregate (vote/adjudicate); REUSE
    the existing sampling/aggregate plumbing if `corpus_techniques.md` #9 landed.
  - `tot` — bounded breadth/depth branch-and-select (predicted null H3c; keep
    cheap, it exists to TEST the null, not to win).
- CRITICAL: every reasoning condition must receive the IDENTICAL retrieved pages
  (freeze Phase-1 selection across the reasoning sweep) or the comparison
  confounds reasoning with retrieval. Add a guard/assertion.
- Cost axis matters here (parallel methods multiply calls) — already logged.
- **Test:** all five reasoning levels run on the dev slice over the SAME selected
  pages; outputs parse to a scoreable answer; cost logs reflect N samples.

### Step 6 — RQ4: ASSEMBLE coarse-to-fine (rerank exists; frame + extend)
RQ4 reuses RQ1 retrievers and RQ3 prompts (dependency-last). Build minimally.
- `single_pass` (reference) vs `retrieve_rerank_read` (existing `rerank: llm`) vs
  `page_to_region` (retrieve page -> crop/zoom -> read) vs `plan_execute`
  (decompose -> fetch per step). Start with the first two; the latter two are
  if-time.
- Enforce MATCHED FINAL BUDGET (same #pages into the reader) so the contrast is
  narrowing-vs-not, not more-compute (RQ spec RQ4 control, H4c).
- Slice the report by document-length bins (H4b) and over/under-retrieval regime.
- **Test:** a coarse-to-fine run and a single-pass run feed the SAME number of
  final pages to the reader; cost logs show the extra stage's calls.

### Step 7 — Update the docs to the new framing
- README §6 sub-study table -> RQ table (RQ1–RQ4 + crosscutting + baselines),
  each row citing its RQ-spec hypotheses and discriminating subset.
- `context.md`: replace the A/B/C sub-study framing with the RQ framing; keep the
  tool inventory (now labelled as "instantiations of mechanisms").
- Add a one-line pointer at the top of README and context.md to
  `docs/research_questions.md` as the governing spec.
- **Test:** docs reference RQ ids consistently; no dangling A/B/C names except in
  a "formerly known as" note.

## 3. Per-RQ control reminder (do not get this wrong)
The fixed control is NOT always the dense reference retriever:
- RQ1 (retrieval is the variable): control = no-retrieval floor + oracle ceiling.
- RQ2/RQ3/RQ4 (retrieval not the variable): control = reference pipeline, with
  the SAME retrieved pages held fixed across the swept axis.
Encode this per-study in the suite metadata so the reporter compares against the
right baseline automatically.

## 4. Acceptance criteria for the whole pivot
- All pre-existing tests pass; new tests for RQ3/RQ4 and RQ-reporting added.
- `configs/reference.yaml` exists and is the documented control.
- Grid suite uses RQ1–RQ4 + crosscutting + baselines naming.
- Report emits, per RQ, a hypothesis verdict table with per-subset results vs the
  declared control. Predicted nulls render as findings.
- RQ3 reasoning axis runs over frozen retrieved pages (guarded).
- RQ4 enforces matched final budget.
- Docs (README, context.md) reframed; research_questions.md cited as governing.
- Nothing in the KEEP column was deleted or behaviourally changed.

## 5. Scope discipline (from RQ spec §2)
Depth-first. Land RQ1 + RQ2 fully (SPINE) before RQ3; RQ3 before RQ4. If time
runs short, stop at a clean RQ boundary — two RQs done to the §4 bar beat four
done shallowly. Do NOT attempt to grid all RQs at once; build and validate each
on the dev slice with the 3B before any Kaya run.