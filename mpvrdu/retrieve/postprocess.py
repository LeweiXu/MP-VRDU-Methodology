"""Training-free post-processing on a retriever's ranked page list (Tier 1).

These wrap ANY retriever (sparse / dense / visual) — they operate on the scored,
page-collapsed candidate list, never on a specific index. Two families:

- ``apply_k_strategy`` — adaptive top-k (#1). Replaces the constant ``top_k`` cut
  with a per-query cut read off the score distribution: keep the high-scoring
  cluster (ViDoRAG fits a 2-component GMM; AVIR uses k-means k=2 with a Top-8
  cap). Degenerate distributions (too few pages, near-uniform scores) fall back
  to the fixed cut, matching the documented ViDoRAG/AVIR limitations.
- ``expand_pages`` / ``build_page_sections`` — selection expansion (#4, #5).
  After the cut, bring co-located pages along: the parent section (MHier-RAG /
  MultiDocFusion) or the immediate neighbours (DFVC's training-free shadow).

All functions are pure (no models, no GPU) and deterministic — sklearn estimators
are seeded so a config hash maps to one selection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

# adaptive-k degrades to fixed below this many candidate pages (GMM/k-means
# misbehave on <~4 points — ViDoRAG limitation) ...
_MIN_PAGES_FOR_ADAPTIVE = 4
# ... or when the score spread is essentially flat (near-uniform distribution).
_FLAT_EPS = 1e-9
# AVIR caps adaptive k-means selection at Top-8.
_KMEANS_CAP = 8


# --------------------------------------------------------------------------- #
# #1 adaptive top-k
# --------------------------------------------------------------------------- #
def apply_k_strategy(scores: list[float], strategy: str, top_k: int) -> int:
    """Return how many of the (descending-sorted) candidates to keep.

    ``scores`` must be in retrieval rank order (best first); the adaptive cut
    keeps the leading high-score cluster, so the answer is a prefix length.
    """
    n = len(scores)
    if n == 0:
        return 0
    if strategy == "fixed":
        return min(top_k, n)

    import numpy as np

    arr = np.asarray(scores, dtype=float)
    # degenerate -> fixed fallback (too few points, or no spread to cluster on)
    if n < _MIN_PAGES_FOR_ADAPTIVE or float(arr.max() - arr.min()) < _FLAT_EPS:
        return min(top_k, n)

    try:
        if strategy == "gmm":
            keep = _gmm_keep(arr)
        elif strategy == "kmeans":
            keep = min(_kmeans_keep(arr), _KMEANS_CAP)
        else:
            raise ValueError(f"unknown k_strategy {strategy!r}")
    except Exception:  # noqa: BLE001 - any estimator hiccup -> safe fixed fallback
        return min(top_k, n)
    return max(1, min(keep, n))


def _gmm_keep(arr) -> int:
    """ViDoRAG: 2-component GMM over scores; keep the higher-mean component."""
    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = arr.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=1).fit(X)
    labels = gm.predict(X)
    high = int(np.argmax(gm.means_.ravel()))
    return int((labels == high).sum())


def _kmeans_keep(arr) -> int:
    """AVIR: k-means (k=2) over scores; keep the higher-centroid cluster."""
    import numpy as np
    from sklearn.cluster import KMeans

    X = arr.reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    high = int(np.argmax(km.cluster_centers_.ravel()))
    return int((km.labels_ == high).sum())


# --------------------------------------------------------------------------- #
# #4 / #5 selection expansion
# --------------------------------------------------------------------------- #
def expand_pages(pages: list[int], scores: list[float], *, mode: str,
                 n_pages: int, window: int = 1,
                 page_sections: Optional[dict[int, object]] = None
                 ) -> tuple[list[int], list[float]]:
    """Expand a selected page list. Only ever ADDS pages (never drops/reorders
    the originals); appended pages get score 0.0 and sit after the ranked hits.

    - ``parent_page``: no-op here — generation feeds whole pages, so a chunk hit
      already drags its parent page (and its figures) along. Kept as an explicit
      control for the methodology write-up.
    - ``parent_section``: add every page sharing a hit's section (``page_sections``
      maps page -> section id; degrades to parent_page when absent).
    - ``adjacent``: add p-w .. p+w around each hit, clamped to the doc bounds.
    """
    out = list(pages)
    out_scores = list(scores)
    if mode == "none" or mode == "parent_page" or not pages:
        return out, out_scores

    seen = set(pages)

    def add(p: int) -> None:
        if 0 <= p < n_pages and p not in seen:
            seen.add(p)
            out.append(p)
            out_scores.append(0.0)

    if mode == "adjacent":
        for p in pages:
            for d in range(1, window + 1):
                add(p - d)
                add(p + d)
        return out, out_scores

    if mode == "parent_section":
        if not page_sections:
            return out, out_scores      # no structure -> behaves like parent_page
        sec_to_pages: dict[object, list[int]] = defaultdict(list)
        for pg, sec in page_sections.items():
            sec_to_pages[sec].append(pg)
        for p in pages:
            sec = page_sections.get(p)
            if sec is None:
                continue
            for sp in sorted(sec_to_pages[sec]):
                add(sp)
        return out, out_scores

    raise ValueError(f"unknown expand mode {mode!r}")


def build_page_sections(parsed_pages) -> dict[int, object]:
    """Map each 0-based page to a section id from parser structure.

    A section runs from the page that introduces a heading until the next page
    with a different leading heading; intervening pages inherit it. Pages before
    any heading (and whole docs with no structure) get a UNIQUE id so
    parent_section degrades to a no-op there rather than swallowing the document.
    """
    page_sections: dict[int, object] = {}
    cur = -1
    last_heading = None
    for pp in parsed_pages:
        heading = None
        if pp.sections:
            for h, _ in pp.sections:
                if h:
                    heading = h
                    break
        if heading is not None and heading != last_heading:
            cur += 1
            last_heading = heading
        if cur < 0:
            page_sections[pp.page_index] = ("page", pp.page_index)  # unique
        else:
            page_sections[pp.page_index] = ("section", cur)
    return page_sections
