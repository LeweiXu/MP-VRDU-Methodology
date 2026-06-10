# Research Questions & Hypotheses Specification

> The TOP-LEVEL document for the analysis study. Sits ABOVE context.md.
> context.md (study design) and the grid suites are the IMPLEMENTATION of this
> spec. Every sub-study must trace back to a research question (RQ) here.
> Last updated: 2026-06-08.

## 0. Why this document exists (the framing principle)

This study analyses **mechanisms**, not **products**. The object of study is a
*property of the MP-VRDU pipeline* (e.g. "relation-aware retrieval", "parallel
reasoning", "structural granularity"), NOT a named tool (ColPali, MinerU, BM25).
Named tools are only **instantiations** that operationalise a mechanism.

- WRONG framing: "Which is better, ColPali or BM25?" (a procurement question;
  obsolete when the next model ships; no hypothesis; produces a leaderboard).
- RIGHT framing: "Does visual late-interaction retrieval beat lexical retrieval
  for sparse cross-page evidence, and where does the gap appear?" (a mechanism
  question; generalises; has a falsifiable directional hypothesis; produces
  analysis). ColPali and BM25 are just how we realise the two sides.

The mechanism families below are drawn directly from the survey's taxonomy. The
survey already posed these questions; this spec turns them into experiments.

Every RQ states: (a) the mechanism, (b) a DIRECTIONAL hypothesis, (c) WHERE the
effect should appear (the discriminating subset), (d) the tools that instantiate
each level, (e) the control it is measured against.

## 1. The reference pipeline (the control / fairness backbone)

Every condition is a ONE-MECHANISM perturbation of this fixed skeleton. All
non-tested components stay pinned at these defaults and are declared in methods.

```
REFERENCE PIPELINE (the control everything is compared against):
  Generator         : Qwen2.5-VL-7B-Instruct (frozen)        [pinned]
  Representation    : ONE default parser (text)              [pinned default]
  Retrieval         : DENSE TEXT retriever, fixed k          [pinned default]
  Reasoning / prompt: plain single-pass CoT                  [pinned default]
  Generation modality: fixed (declare image or text)         [pinned default]
  Judge             : fixed LLM judge, declared              [pinned]
```

**Why dense text as the default retriever (the control):**
- Modality-neutral: operates on parser text, so REPRESENTATION experiments have
  a clean background (a visual default would make parser experiments impossible).
- A reasonable baseline, not a strawman floor: mechanism gains are honest, not
  inflated by an artificially weak control (the risk with a BM25 default).
- The natural MIDPOINT of the retrieval axis (sparse BM25 <-> dense <-> visual
  ColPali), so the retrieval RQ reads as "move from centre in two directions".
- Cost: introduces one hidden variable (which embedder) -> neutralise by pinning
  ONE specific frozen embedder and declaring it, like the generator.

**Per-RQ control caveat:** when the mechanism under test IS retrieval, retrieval
can't also be the fixed control. For RQ1 the controls become the NO-RETRIEVAL
floor and ORACLE ceiling instead. Each RQ states its own control explicitly.

## 2. Scope management (READ THIS — all four families were chosen principal)

All four survey mechanism families are designated principal. That is ambitious
for ~5 months. To stay honest, the families are ordered by priority and the plan
commits depth-first: finish RQ1 fully before RQ2, etc. A partial study with two
families done rigorously beats four done shallowly.

- **SPINE (must complete):** RQ1 Retrieval, RQ2 Representation. These are the
  survey's load-bearing axes and the ones the harness already mostly supports.
- **PRINCIPAL (target):** RQ3 Reasoning. Currently ABSENT from the harness (no
  reasoning axis exists yet) — needs building. High value, clearly in the survey.
- **PRINCIPAL-IF-TIME (extension):** RQ4 Coarse-to-fine. Most complex; subsumes
  pieces of RQ1/RQ3; best run last once the others are instrumented.

The order is also a dependency order: RQ4 reuses RQ1's retrievers and RQ3's
prompts, so doing it last means its parts already exist.

------------------------------------------------------------------------------
## RQ1 — Retrieval: similarity vs relation-aware (+ modality)  [SPINE]

**Survey origin:** retrieval section, similarity-based vs relation-aware split.

**Mechanism:** how evidence pages are *scored and selected* — by independent
query-similarity, or by traversing explicit document structure (sections,
cross-references). Sub-mechanism: the SIGNAL similarity uses (lexical / dense
semantic / visual).

**Levels & instantiations:**
- Similarity / lexical        -> BM25 (and TF-IDF as a second lexical point)
- Similarity / dense semantic -> the pinned dense text embedder (= the control)
- Similarity / visual         -> ColPali, ColQwen
- Relation-aware / structural -> tree/section-traversal, parent-page expansion
- Relation-aware / graph      -> page-graph traversal w/ frozen-VLM logical score
- (fusion)                    -> RRF over lexical+dense, and text+visual

**Hypotheses (directional, falsifiable):**
- H1a: Visual late-interaction beats lexical on questions whose evidence is in
  charts/figures/tables; lexical is competitive or better on pure-text,
  keyword-heavy evidence. (Predicts an INTERACTION with evidence-source type,
  not a flat winner. Echoes UDA's BM25-wins-on-financial finding.)
- H1b: Relation-aware retrieval beats similarity-only SPECIFICALLY on multi-hop
  / cross-page questions whose individual evidence pages are low-similarity;
  the gap WIDENS with hop count, and is ~zero on single-page questions.
- H1c: Fusion >= best single signal, with diminishing returns; gains concentrate
  where the two signals disagree.

**Discriminating subsets (WHERE to look):** evidence-source type (text / table /
chart / figure / layout), and single-page vs cross-page question type. The
headline is NOT a single mean — it is the per-subset breakdown.

**Control:** no-retrieval floor + oracle ceiling (retrieval is the variable, so
it can't be the fixed control). Report each method's recovery of the floor->
oracle gap, per subset.

**Primary metric:** evidence-page recall@k (mechanism-level), then downstream
accuracy/F1 (does better recall convert to better answers?).

------------------------------------------------------------------------------
## RQ2 — Representation: extraction fidelity & structure  [SPINE]

**Survey origin:** representation section (text / visual / structure modalities).

**Mechanism:** how faithfully and with how much structure the document is turned
into something indexable/readable. NOT "which parser is best" — the question is
how much extraction FIDELITY and STRUCTURE-PRESERVATION matter to downstream QA.

**Levels & instantiations:**
- Fidelity axis (text quality): native-text parse (PyMuPDF4LLM) vs layout-aware
  ML parse (MinerU) vs OCR-from-image (Tesseract). The CONTRAST is fidelity, the
  tools are instances.
- Structure axis: flat chunking vs structure-aware (section/hierarchy) chunking.
- Granularity axis: chunk vs page vs section vs element.
- (format sub-study, text-only): raw vs Markdown vs HTML serialisation.

**Hypotheses:**
- H2a: Downstream accuracy is more sensitive to extraction fidelity on
  text-dense / table-heavy questions than on figure questions (where a visual
  reader bypasses text extraction entirely).
- H2b: Structure-aware chunking beats flat chunking on cross-page questions
  (preserves parent/section context) but not on single-page questions —
  ADJUDICATING the live dispute (RAG-DocVQA: no help; MultiDocFusion/DMAP: helps).
- H2c: There is a granularity sweet spot; finer-than-page improves recall
  precision but can fragment evidence and hurt the reader.

**Discriminating subsets:** evidence-source type; single- vs cross-page.

**Control:** the reference pipeline with the dense retriever fixed; vary ONLY the
representation. Text retrievers only (visual retrieval ignores the parser).

**Primary metric:** downstream accuracy/F1 (fidelity's effect is on the answer);
recall@k as the mediator (does worse text hurt retrieval or only reading?).

------------------------------------------------------------------------------
## RQ3 — Reasoning: sequential vs parallel  [PRINCIPAL — NOT YET BUILT]

**Survey origin:** prompted-reasoning section (sequential CoT/self-reflection vs
parallel self-consistency/ToT). NOTE: the harness currently has NO reasoning
axis — generation is plain "answer the question". This must be built.

**Mechanism:** the reasoning STRUCTURE applied over an assembled evidence buffer,
holding the evidence itself fixed. Isolates reasoning from retrieval.

**Levels & instantiations:**
- Direct answer (no explicit reasoning)        -> baseline prompt
- Sequential / linear  -> Chain-of-Thought
- Sequential / revised -> self-reflection (Reflexion-style single-thread)
- Parallel / vote      -> self-consistency (sample N, majority/adjudicate)
- Parallel / search    -> Tree-of-Thoughts (survey notes low MP-VRDU uptake —
  include as the test of WHETHER branching helps when evidence sparsity, not
  search, is the bottleneck)

**Hypotheses:**
- H3a: CoT helps on multi-hop questions (binds cross-page evidence step by step)
  but gives ~zero gain on single-hop lookup. (Locates CoT's value in the FIDELITY
  stage, not recall — it operates only over already-assembled evidence.)
- H3b: Self-consistency improves robustness proportional to sampling cost; the
  accuracy gain trades against the logged token/call cost (efficiency frontier).
- H3c: ToT gives little over CoT in MP-VRDU because the bottleneck is finding
  sparse evidence, not searching a reasoning tree — a predicted NULL result,
  which is itself a finding that corroborates the survey's claim.

**Discriminating subsets:** single-hop vs multi-hop; reasoning-heavy vs lookup.

**Control:** reference pipeline with retrieval AND representation fixed; the
evidence buffer is identical across reasoning conditions, so any delta is the
reasoning structure alone. (Critical: same retrieved pages every condition.)

**Primary metric:** accuracy/F1; cost (calls/tokens) as the co-axis for the
parallel methods.

------------------------------------------------------------------------------
## RQ4 — Coarse-to-fine / evidence-narrowing  [PRINCIPAL-IF-TIME — run last]

**Survey origin:** coarse-to-fine appendix (page->region, retrieve-rerank-read,
plan-then-execute).

**Mechanism:** progressive scope-narrowing — does a broad-then-narrow pass beat a
single-shot pass at matched final budget? Reuses RQ1 retrievers and RQ3 prompts,
so it is dependency-last.

**Levels & instantiations:**
- Single-pass (retrieve top-k -> read)              -> reference pipeline
- Retrieve -> rerank -> read (LLM/VLM rerank stage) -> coarse-to-fine retrieval
- Page -> region (retrieve page, then crop/zoom)    -> element-level narrowing
- Plan -> execute (decompose, then fetch per step)  -> the agentic end

**Hypotheses:**
- H4a: Reranking improves page-F1 most when the first-stage retriever
  over-retrieves (high recall, low precision at large k); at small k the rerank
  gain shrinks. (Echoes SimpleDoc's k-vs-F1 observation.)
- H4b: Coarse-to-fine's advantage over single-pass GROWS with document length —
  near zero on short docs, largest on the longest.
- H4c: Narrowing cannot recover evidence dropped by the coarse stage — accuracy
  is upper-bounded by first-stage recall, so gains come from PRECISION/denoising,
  not recall. (Testable against the oracle.)

**Discriminating subsets:** document length bins; over- vs under-retrieval regime.

**Control:** single-pass reference pipeline at matched FINAL evidence budget
(so the comparison isn't just "more compute"). Hold k-into-reader constant.

**Primary metric:** accuracy/F1 at matched budget; cost co-axis (extra stages
cost calls); recall vs precision decomposition.

------------------------------------------------------------------------------
## 3. Cross-cutting variables (swept within RQs, not RQs themselves)

- **top-k** — swept inside RQ1/RQ4; non-monotonic (attention dilution); produces
  the accuracy-vs-recall-vs-cost curves.
- **generation modality (image/text/both)** — a controlled secondary axis; pin it
  per-RQ and report the setting. Becomes its own mini-RQ only if time allows.
- **generator size (3B/7B/32B)** — robustness check: do the mechanism RANKINGS
  hold across backbone scale? Run on the SPINE RQs only.

## 4. What makes a result "analysis" not a "leaderboard" (the bar)

Each RQ result must report:
1. The per-subset breakdown (the discriminating subset), NOT just a grand mean —
   the hypothesis predicts WHERE the effect is, so show that location.
2. The directional hypothesis stated up front, and whether it held.
3. The effect measured against the stated control with a CI / paired test.
4. The causal story (from the survey) for WHY the mechanism behaves as observed.
5. Cost, where the mechanism trades accuracy for compute.

A null result that refutes a hypothesis (e.g. H3c: ToT doesn't help) is a
finding, not a failure — report it as one.

## 5. Mapping RQs -> sub-studies -> harness

| RQ | Mechanism | Harness sub-study (context.md) | Build status |
|----|-----------|-------------------------------|--------------|
| RQ1 | retrieval similarity/relation/modality | A_retrieval, A_topk (+ new relation-aware, fusion) | mostly built; relation-aware = new |
| RQ2 | representation fidelity/structure | B_representation (+ structure chunking) | built; structure chunking = extend |
| RQ3 | reasoning sequential/parallel | NEW reasoning sub-study | NOT built — add prompt axis |
| RQ4 | coarse-to-fine narrowing | NEW; reuses A + reasoning | NOT built — run last |

Named tools stay as config enum values; what changes is that each comparison is
now LABELLED and ANALYSED as a mechanism with a hypothesis, per-subset, against a
declared control — not reported as "tool X vs tool Y".