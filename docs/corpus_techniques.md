# Corpus Techniques: Candidate Extensions from the MP-VRDU Literature

> For the coding agent. Companion to `context.md`, `agent_build_plan.md`, README.
> Source: the surveyed MP-VRDU corpus (CSV summaries in project dir).
> Purpose: catalogue techniques from existing systems that can be ADDED to this
> analysis harness as new conditions/toggles, with where each plugs in and how
> hard it is. Last updated: 2026-06-08.

## How to read this file

This study tests **existing, off-the-shelf, training-free techniques** behind a
**frozen generator**, one variable at a time (see context.md). So the corpus is
filtered hard:

- **ADDABLE** — training-free (or uses an off-the-shelf frozen model we already
  load), slots into an existing stage as a new method or a toggle. Implement.
- **TECHNIQUE-ONLY** — the full system needs training or a proprietary model, but
  one *component idea* is extractable training-free. Implement the idea, cite the
  system, do NOT claim to reproduce the system.
- **OUT OF SCOPE** — requires training a model or reproducing a training pipeline.
  Belongs to the S10 stretch goal, not the analysis. Cite as frontier only.

Every ADDABLE item below states: the source system(s), what it changes, which
module/config field it touches, and effort. Implement them as new enum values or
boolean toggles on the existing `RunConfig`, never as code forks.

------------------------------------------------------------------------------
## TIER 1 — implement these (best effort-to-payoff; each resolves a real gap)

> **STATUS (2026-06-09): all five Tier-1 items IMPLEMENTED + tested.** They are
> config toggles on `RetrievalConfig` / `RepresentationConfig`, wrapping any
> retriever — no code forks (README §8 config reference). Where they live:
> - #1 adaptive top-k → `retrieval.k_strategy: {fixed|gmm|kmeans}` (+ `candidate_k`),
>   `mpvrdu/retrieve/postprocess.py::apply_k_strategy`.
> - #2 structure-aware chunking → `representation.chunking: section` now splits
>   long sections (heading kept) per MultiDocFusion, `mpvrdu/represent/chunking.py`.
> - #3 LLM rerank → `retrieval.rerank: llm` (+ `rerank_candidates`, `rerank_model`),
>   `mpvrdu/retrieve/rerank.py`.
> - #4 parent expansion + #5 adjacent → `retrieval.expand:
>   {parent_page|parent_section|adjacent}` (+ `expand_window`),
>   `postprocess.py::expand_pages`.
>
> Example conditions: `configs/tier1/*.yaml`. Tests: `tests/test_postprocess.py`,
> `tests/test_rerank.py`, plus section-chunking cases in `tests/test_chunking.py`.
> Note on #4 `parent_page`: in this page-granular harness the generator already
> feeds whole pages, so `parent_page` coincides with the default page selection —
> kept as an explicit control. `parent_section` is the genuinely new parent mode.

### 1. Adaptive top-k selection  [Stage 1 / retrieve]
- **Source:** ViDoRAG (GMM over score distribution), AVIR (k-means k=2 for long
  docs + probability threshold for short docs, Top-8 cap).
- **What it changes:** replaces the fixed `top_k` cut with a per-query adaptive
  cut derived from the retrieval-score distribution. Keeps the high-scoring
  cluster instead of a constant k.
- **Why it matters:** directly converts the existing `A_topk` SWEEP into a
  FINDING — fixed-k vs adaptive-k, retriever held constant. The corpus disagrees
  on the method (GMM vs k-means), so measuring which wins is a contribution. Hits
  the top-k non-monotonicity already documented (more pages → attention dilution).
- **Where:** new `retrieval.top_k` mode. Add `top_k: adaptive_gmm` and
  `top_k: adaptive_kmeans` as accepted values (alongside the integer), or a
  separate `retrieval.k_strategy: {fixed|gmm|kmeans_threshold}` field. Implement
  as a post-processing step on the retriever's scored candidate list — it wraps
  ANY existing retriever (sparse/dense/visual), so it's one function applied to
  scores, not a new retriever.
- **Effort:** LOW. ~30 lines (sklearn `GaussianMixture` / `KMeans` on the 1-D
  score array; fall back to fixed-k on degenerate distributions — GMM misbehaves
  on <~4 pages or near-uniform scores, per ViDoRAG limitation).
- **Test:** on the dev slice, assert adaptive-k returns a variable number of
  pages across queries and degrades to fixed-k on tiny docs. Recall@selected
  should be comparable to a tuned fixed-k.

### 2. Flat vs structure-aware chunking  [Stage 2 / represent]
- **Source:** RAG-DocVQA (reports layout-guided chunking did NOT help on
  MP-DocVQA/DUDE/InfoVQA) vs MultiDocFusion + DMAP (hierarchy DOES help). A
  genuine, unresolved disagreement in the literature.
- **What it changes:** the `chunking` axis already has `page|chunk|section`. Make
  the `section` (structure-aware) path real and comparable to flat `chunk`:
  group chunks under their parser-derived section/heading so a retrieved chunk
  carries its hierarchical context, vs flat fixed-length chunks that ignore it.
- **Why it matters:** this is arguably the single most PUBLISHABLE result here —
  you ADJUDICATE a live dispute rather than adding a bar to a chart. Pairs
  naturally with sub-study B (representation).
- **Where:** `mpvrdu/represent/` chunking module; reuse the existing
  `representation.chunking: section` value. Structure-aware chunking needs the
  parser to emit headings/hierarchy (PyMuPDF4LLM Markdown headers, MinerU
  structure). Degrade gracefully to flat when the parser gives no structure
  (already the documented behaviour).
- **Effort:** MEDIUM. The hook exists; the work is deriving section boundaries
  from parser output and a DFS-style grouping (MultiDocFusion's recipe: split a
  section chunk only when it exceeds max length, mark with Markdown headers).
- **Test:** assert section chunks never split mid-heading on a structured PDF;
  assert flat vs section produce different unit boundaries on the same doc.

### 3. LLM/VLM reranking toggle  [Stage 1 / retrieve — "retrieve → rerank → read"]
- **Source:** SimpleDoc (dual-cue: embedding shortlist → LLM re-rank over page
  summaries), CREAM (RankVicuna re-rank in groups), MHier-RAG (joint similarity +
  LLM re-rank), MM-R5 (the trained version — OUT OF SCOPE; the *prompted* version
  is addable).
- **What it changes:** adds an optional second stage that takes the retriever's
  top-N candidates and asks the FROZEN generator (or judge model) to re-score /
  filter them down to top-k, using page text/summaries rather than embedding
  similarity. A pure post-retrieval toggle.
- **Why it matters:** the coarse-to-fine pattern shared by most strong systems.
  SimpleDoc's own numbers show reranking swings page-level F1 dramatically
  (over-retrieval at high k tanks F1; reranking to ~3 pages recovers it). Gives a
  clean factor: retriever-alone vs retriever+rerank, training-free.
- **Where:** new `retrieval.rerank: {none|llm}` toggle + `rerank_candidates: N`
  (retrieve N, rerank to `top_k`). Implement in `retrieve/` as a wrapper that
  reuses the already-loaded VLM/judge — mind the two-phase execution: reranking
  uses a model, so either reuse the Phase-1 retriever's slot or run it as a small
  Phase-1.5 with explicit load/unload to respect the 12GB ceiling.
- **Effort:** MEDIUM. Prompt design + careful GPU-memory sequencing.
- **Test:** assert rerank reorders the candidate list and returns exactly `top_k`;
  assert recall@k(rerank) >= recall of the raw retriever at the same k on the slice.

### 4. Parent-page / parent-section expansion  [Stage 1 / retrieve — post-processing]
- **Source:** MHier-RAG (parent-page retrieval: hit a chunk, return the WHOLE
  parent page with its visual elements), MultiDocFusion (hierarchical parent
  context).
- **What it changes:** after a sub-page (chunk) retriever selects a chunk, expand
  the selection to its parent page (or section) so co-located figures/tables come
  along. Training-free post-processing on the selected units.
- **Why it matters:** directly tackks "the answer text and the figure are on the
  same page but chunking split them" — a multi-modality-disconnection failure the
  corpus repeatedly cites. Cheap, MP-distinctive.
- **Where:** `retrieval.expand: {none|parent_page|parent_section}` toggle. Each
  `Unit` already records its source page (per README §3), so parent-page
  expansion is a group-by-page on the selected units.
- **Effort:** LOW. ~20 lines on top of existing chunk retrieval.
- **Test:** assert expansion only adds pages (never drops), and that a chunk-level
  hit yields its full page in the selected set.

### 5. Adjacent-page expansion  [Stage 1 / retrieve — post-processing]
- **Source:** DFVC (the TRAINED adapter is OUT OF SCOPE; its INSIGHT — neighbour
  pages carry context for spanning tables/paragraphs — has a training-free form).
- **What it changes:** after selecting page i, also include i-1 and i+1 (window
  configurable). The training-free shadow of cross-page-context fusion.
- **Why it matters:** lets you MEASURE whether the cross-page-continuity problem
  matters at all on the benchmark, with zero training. If +/-1 expansion helps,
  that motivates the trained approaches (and your S10 method); if it doesn't,
  that's also a finding.
- **Where:** `retrieval.expand: adjacent` + `expand_window: 1`. Same expansion
  hook as #4.
- **Effort:** LOW. ~15 lines.
- **Test:** assert window=1 adds at most 2 pages per hit, clamped at doc bounds.

------------------------------------------------------------------------------
## TIER 2 — implement if time allows (clear value, more effort or narrower)

### 6. Hybrid fusion variants (text+visual, not just sparse+dense)  [Stage 1]
- **Source:** MDocAgent (parallel ColBERT text + ColPali visual), M2RAG (BM25 +
  VisRAG dual-tower), VisDoM (BGE text + ColPali visual, parallel).
- **What it changes:** the harness has `hybrid` via RRF over `[bm25, dense]`.
  Extend `hybrid_methods` to allow a TEXT+VISUAL fusion (e.g. `[bm25, colpali]`),
  fusing a text-retriever ranking with a visual-retriever ranking via RRF.
- **Why it matters:** the canonical multimodal-RAG recipe in the corpus; tests
  whether fusing modalities beats the best single modality.
- **Where:** already-present `retrieval.hybrid_methods` + `rrf_k`; just allow
  visual members and make the two-phase code build both indices in Phase 1.
- **Effort:** MEDIUM (two retriever models in one phase — watch 12GB; may need
  sequential index-build with unload between).
- **Test:** assert RRF over a text and a visual ranking yields a fused order
  distinct from either input.

### 7. Summary-augmented indexing  [Stage 1+2]
- **Source:** SimpleDoc (index each page twice: ColPali embedding + a 3-5
  sentence VLM-generated summary; retrieve over both cues).
- **What it changes:** an offline pass where the frozen VLM writes a short summary
  per page; a text retriever indexes the summaries; retrieval can use the summary
  channel alone or fused with the visual channel.
- **Why it matters:** tests whether cheap generated summaries are a better text
  proxy than raw OCR/parser text for retrieval.
- **Where:** new `representation.text_format: page_summary` (the "text" the text
  retriever indexes becomes VLM summaries rather than parser output) OR a new
  retriever variant. Offline summary cache like the render cache.
- **Effort:** MEDIUM-HIGH (offline generation pass + caching; uses the VLM).
- **Test:** assert summaries cached once per page and reused; assert summary index
  retrieves sanely on the slice.

### 8. Three-step Evidence-Curation → CoT → Answer prompting  [Stage 3 / generate]
- **Source:** VisDoM (Evidence Curation + CoT + Answer), MHier-RAG (CoT +
  structured output), SLEUTH (clue discovery then decision).
- **What it changes:** the generator prompt structure: first curate which
  retrieved units are relevant, then reason (CoT), then answer — vs the plain
  "answer directly" prompt. A generation-side toggle.
- **Why it matters:** isolates whether structured prompting over the SAME
  retrieved evidence improves answers — a clean Stage-3 factor independent of
  retrieval.
- **Where:** `generation.prompt_style: {plain|curate_cot}` toggle in `generate/`.
- **Effort:** LOW-MEDIUM (prompt templates + parsing the final answer out).
- **Test:** assert the structured prompt still yields a parseable final answer the
  judge can score; compare on the slice.

### 9. Sampling-adjudication / self-consistency  [Stage 3 / generate — cross-cutting]
- **Source:** DocLens (Answer Sampler + Adjudicator), ViDoRAG (Inspector
  reflection), self-consistency (majority vote).
- **What it changes:** sample multiple answers (temperature>0) and either
  majority-vote or have the frozen model adjudicate the best.
- **Why it matters:** a training-free robustness lever; trades inference cost for
  accuracy. Interacts with the cost metric already logged (n_llm_calls).
- **Where:** `generation.n_samples: 1` + `generation.aggregate: {none|vote|adjudicate}`.
  Note this changes `temperature` from 0 — record it; ties into the seed-variance
  analysis already built.
- **Effort:** MEDIUM. Watch the cost-axis interpretation (multiplies calls).
- **Test:** assert n_samples>1 produces multiple raw generations and one
  aggregated answer; assert cost logs reflect the extra calls.

### 10. Evidence-citing output schema  [Stage 3 / generate — faithfulness]
- **Source:** DocR1 / DocSeeker output FORMAT (<think>/<evidence_page>/<answer>) —
  the TRAINED reward is OUT OF SCOPE; the prompted output schema is addable.
- **What it changes:** prompt the frozen generator to name the evidence page(s)
  it used before answering. Yields a predicted-evidence signal at generation time.
- **Why it matters:** a training-free faithfulness probe — does forcing evidence
  citation change accuracy, and do cited pages match gold `evidence_pages`?
  Connects to the abstention/hallucination analysis already built.
- **Where:** `generation.cite_evidence: bool` toggle; parse the cited pages and
  log them next to the prediction so analysis can score citation precision.
- **Effort:** LOW-MEDIUM.
- **Test:** assert cited pages are parsed into a list and logged; assert the final
  answer is still scoreable.

------------------------------------------------------------------------------
## TIER 3 — only if a Tier-1/2 result motivates it (high effort / edges agentic)

### 11. Graph / tree-traversal retrieval (training-free)  [Stage 1]
- **Source:** KGP (KG over passages + structural nodes, LLM traversal), MoLoRAG
  (ColPali page graph + frozen-VLM logical-relevance scoring + hop budget),
  MHier-RAG (topological cross-page tree).
- **Addable form:** build a page/section graph OFFLINE from embedding similarity
  (threshold edges); at query time seed from top-w similar pages and let the
  frozen VLM score unvisited neighbours by prompted logical relevance, to a hop
  budget. MoLoRAG's base (not MoLoRAG+, which fine-tunes) is training-free.
- **Why Tier 3:** highest-effort retrieval addition, edges into iterative/agentic
  control (variable-length trajectories complicate the controlled comparison and
  the cost axis). Only worth it if relation-aware retrieval looks promising from
  the parent-page/adjacent results (#4/#5).
- **Where:** a new `retrieval.method: graph_traverse` with its own offline
  graph-build cache. Keep the hop budget small and FIXED to preserve comparability.
- **Effort:** HIGH. Offline graph build + traversal loop + multiple VLM calls/query.
- **Test:** assert the graph builds deterministically; assert traversal respects
  the hop budget; recall vs plain dense at matched k.

------------------------------------------------------------------------------
## OUT OF SCOPE for the analysis (training / proprietary required) — cite only

These are the frontier your training-free conditions are measured against, and
the candidate basis for the S10 stretch method. Do NOT attempt to reproduce them
in the analysis.

- **Trained MLLM-centric / RL systems:** DocR1 (EviGRPO), DocSeeker (ALR+EviGRPO),
  Doc-V* (SFT+GRPO agent), MM-Doc-R1 (multi-turn SPO), VRAG-RL (visual-perception
  GRPO), URaG (in-MLLM retrieval module), CoR (Chain-of-Reading + Mask-AR + DPO),
  DREAM (LightGBM ranker + MoE MLLM), CREAM (trained multi-page vision encoder).
- **Trained retrievers / rerankers:** MM-R5 (SFT+GRPO reranker), DFVC (trained
  cross-page adapter), VDocRAG (retrieval-pretrained LVLM), SV-RAG (dual-LoRA
  self-retriever), Self-Attention Scoring (trained scoring head).
- **Native trained multi-page MLLMs:** Docopilot (Doc-750K), Leopard, mPLUG-
  DocOwl2, Texthawk2, DocSLM, Hi-VT5, GRAM, RM-T5, Arctic-TILT, LayTokenLLM,
  InstructDoc/InstructDr — these are backbones/encoders, not addable conditions.
- **Proprietary-model-dependent systems:** DocAgent, DocLens, DMAP, MACT (lean on
  GPT-4o / Qwen-plus as core controllers). Their *component ideas* (outline/XML
  structure, parent expansion, sampling-adjudication, plan→execute) are extracted
  into Tier 1-2 above where training-free; the full systems are not reproduced.

------------------------------------------------------------------------------
## Suggested implementation order (maps to roadmap)

1. #1 adaptive top-k  +  #4 parent-page  +  #5 adjacent-page  (all LOW; do first —
   they wrap existing retrievers and need no model changes).
2. #2 flat-vs-structure chunking  (MEDIUM; the high-value publishable result).
3. #3 LLM rerank toggle  (MEDIUM; mind two-phase GPU sequencing).
4. #6 text+visual hybrid, then #8 prompt-style, #10 cite-evidence  (Stage-3 toggles).
5. #7 summary indexing, #9 sampling-adjudication  (cost-heavy; only if motivated).
6. #11 graph traversal  (only if relation-aware retrieval looks promising).

Each becomes a new sub-study or an axis on an existing one. Keep one variable per
comparison; add the confirmatory best-combination run noted in the roadmap's
design caveats so component INTERACTIONS (parser x retriever, modality x retriever,
rerank x k) are not missed by the greedy fix-best-then-vary protocol.