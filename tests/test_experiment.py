"""Experiment-suite expansion: axes -> validated RunConfigs."""

from mpvrdu.experiment import (expand_suite, load_suite, load_suite_metadata,
                               study_metadata)


def test_expand_cross_product():
    suite = {
        "defaults": {
            "data": {"name": "synthetic"},
            "generation": {"generator": "mock", "mock_mode": "gold"},
        },
        "substudies": {
            "A": {"axes": {
                "retrieval.method": ["bm25", "tfidf"],
                "generation.modality": ["image", "text"],
            }},
        },
    }
    runs = expand_suite(suite)
    assert len(runs) == 4                       # 2 x 2
    subs = {s for s, _ in runs}
    assert subs == {"A"}
    # defaults applied + axes overrides set
    by_name = {c.name: c for _, c in runs}
    assert "A__bm25__image" in by_name
    c = by_name["A__bm25__text"]
    assert c.retrieval.method == "bm25"
    assert c.generation.modality == "text"
    assert c.generation.generator == "mock"     # from defaults


def test_distinct_hashes():
    suite = {
        "defaults": {"data": {"name": "synthetic"}},
        "substudies": {"S": {"axes": {"retrieval.top_k": [1, 2, 4]}}},
    }
    runs = expand_suite(suite)
    hashes = {c.hash() for _, c in runs}
    assert len(hashes) == 3                      # each k -> distinct config/file


def test_rq_metadata_is_ignored_by_expander_and_exposed_separately():
    """RQ metadata lives alongside `axes`; the runner ignores it, the reporter
    consumes it (docs/pivot.md Step 2)."""
    suite = {
        "defaults": {"data": {"name": "synthetic"}},
        "substudies": {
            "RQ1_retrieval": {
                "rq": "RQ1",
                "subset": "evidence_source",
                "control": {"kind": "floor_ceiling"},
                "hypotheses": [{"id": "H1a", "text": "...", }],
                "axes": {"retrieval.method": ["bm25", "dense"]},
            },
            "plain": {"axes": {"retrieval.method": ["bm25"]}},
        },
    }
    # expansion still works and only sees axes
    runs = expand_suite(suite)
    assert len(runs) == 3                       # 2 + 1, metadata adds no runs
    # metadata is extracted keyed by sub-study; metadata-free studies omitted
    meta = study_metadata(suite)
    assert set(meta) == {"RQ1_retrieval"}
    assert meta["RQ1_retrieval"]["rq"] == "RQ1"
    assert meta["RQ1_retrieval"]["hypotheses"][0]["id"] == "H1a"


def test_shipped_suites_carry_rq_metadata():
    for path in ("experiments/grid_local_3b.yaml", "experiments/grid_kaya_7b.yaml"):
        meta = load_suite_metadata(path)
        rqs = {m.get("rq") for m in meta.values()}
        assert {"RQ1", "RQ2", "RQ3", "RQ4"} <= rqs
        # every study with hypotheses names a discriminating subset for at least one
        for name, m in meta.items():
            for h in m.get("hypotheses", []):
                assert "text" in h and "id" in h
                expect = h.get("expect") or {}
                # YAML 1.1 parses a bare `on:` key as boolean True — the subset
                # cell MUST be the string key `where` so it survives loading.
                assert True not in expect, f"{name}/{h['id']}: bare `on:` -> True key"
                if "where" in expect:
                    assert isinstance(expect["where"], dict)
                    assert all(isinstance(k, str) for k in expect["where"])


def test_kaya_switch_in_defaults():
    # changing defaults.generation flips every run's generator (the "1-line" switch)
    suite = {
        "defaults": {"data": {"name": "synthetic"},
                     "generation": {"generator": "kaya_vlm",
                                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"}},
        "substudies": {"A": {"axes": {"retrieval.method": ["bm25", "dense"]}}},
    }
    runs = expand_suite(suite)
    assert all(c.generation.generator == "kaya_vlm" for _, c in runs)
