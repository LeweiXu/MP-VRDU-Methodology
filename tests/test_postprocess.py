"""Tier-1 retrieval post-processing: adaptive top-k (#1) + expansion (#4/#5)."""

import pytest

from mpvrdu.represent.base import ParsedPage
from mpvrdu.retrieve.postprocess import (apply_k_strategy, build_page_sections,
                                         expand_pages)


# --------------------------------------------------------------------------- #
# #1 adaptive top-k
# --------------------------------------------------------------------------- #
def test_fixed_strategy_is_constant_cut():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert apply_k_strategy(scores, "fixed", top_k=3) == 3
    assert apply_k_strategy(scores, "fixed", top_k=99) == 5      # clamped to n
    assert apply_k_strategy([], "fixed", top_k=3) == 0


@pytest.mark.parametrize("strategy", ["gmm", "kmeans"])
def test_adaptive_keeps_high_cluster(strategy):
    pytest.importorskip("sklearn")
    # clearly bimodal: three strong pages, then a low tail
    scores = [0.95, 0.92, 0.90, 0.10, 0.07, 0.05, 0.02]
    keep = apply_k_strategy(scores, strategy, top_k=5)
    assert keep == 3                                   # the high cluster only


@pytest.mark.parametrize("strategy", ["gmm", "kmeans"])
def test_adaptive_degrades_on_tiny_or_flat(strategy):
    pytest.importorskip("sklearn")
    # too few points -> fixed fallback
    assert apply_k_strategy([0.9, 0.1], strategy, top_k=4) == 2
    # near-uniform scores -> fixed fallback (cannot cluster)
    flat = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert apply_k_strategy(flat, strategy, top_k=2) == 2


def test_kmeans_cap_at_8():
    pytest.importorskip("sklearn")
    # high cluster of 12 strong pages, k-means should cap the keep at 8
    scores = [0.9] * 12 + [0.01] * 5
    assert apply_k_strategy(scores, "kmeans", top_k=20) == 8


def test_adaptive_returns_variable_across_queries():
    pytest.importorskip("sklearn")
    a = apply_k_strategy([0.9, 0.88, 0.1, 0.08, 0.05], "gmm", top_k=4)
    b = apply_k_strategy([0.9, 0.85, 0.8, 0.78, 0.05], "gmm", top_k=4)
    assert a != b                                      # cut adapts to the shape


# --------------------------------------------------------------------------- #
# #4 / #5 expansion
# --------------------------------------------------------------------------- #
def test_expand_none_and_parent_page_are_identity():
    pages, scores = [2, 0], [0.9, 0.5]
    for mode in ("none", "parent_page"):
        p, s = expand_pages(pages, scores, mode=mode, n_pages=5)
        assert p == pages and s == scores


def test_adjacent_adds_neighbours_clamped():
    p, s = expand_pages([2], [0.9], mode="adjacent", window=1, n_pages=5)
    assert p == [2, 1, 3]                              # original first, then ±1
    assert s == [0.9, 0.0, 0.0]
    # clamp at the bottom edge: page 0 only gains page 1
    p2, _ = expand_pages([0], [0.9], mode="adjacent", window=1, n_pages=5)
    assert p2 == [0, 1]


def test_adjacent_window_bound_and_never_drops():
    p, _ = expand_pages([5], [1.0], mode="adjacent", window=2, n_pages=20)
    assert set(p) == {3, 4, 5, 6, 7}                   # at most 2*window added
    assert p[0] == 5                                   # original kept, ranked first


def test_parent_section_adds_same_section_pages():
    # pages 0,1,2 share section S; pages 3,4 are section T
    page_sections = {0: ("section", 0), 1: ("section", 0), 2: ("section", 0),
                     3: ("section", 1), 4: ("section", 1)}
    p, _ = expand_pages([1], [0.9], mode="parent_section", n_pages=5,
                        page_sections=page_sections)
    assert p[0] == 1 and set(p) == {0, 1, 2}           # whole section, hit first


def test_parent_section_no_structure_is_noop():
    p, _ = expand_pages([1], [0.9], mode="parent_section", n_pages=5,
                        page_sections=None)
    assert p == [1]


# --------------------------------------------------------------------------- #
# section-id construction
# --------------------------------------------------------------------------- #
def test_build_page_sections_spans_until_new_heading():
    pages = [
        ParsedPage(0, "x", sections=[("Intro", "body")]),
        ParsedPage(1, "x", sections=None),                  # continues Intro
        ParsedPage(2, "x", sections=[("Methods", "body")]),
    ]
    ps = build_page_sections(pages)
    assert ps[0] == ps[1]                                   # 1 inherits 0's section
    assert ps[0] != ps[2]                                   # new heading -> new id


def test_build_page_sections_unstructured_pages_unique():
    pages = [ParsedPage(0, "x", sections=None),
             ParsedPage(1, "x", sections=None)]
    ps = build_page_sections(pages)
    assert ps[0] != ps[1]                                   # no merging w/o structure
