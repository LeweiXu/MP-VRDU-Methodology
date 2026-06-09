"""Stage 0 — config schema + loader."""

import pytest

from mpvrdu.config import (ConfigError, RunConfig, dict_to_config, load_config)


def test_load_smoke_yaml_validates():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.data.name == "synthetic"
    assert cfg.retrieval.method == "oracle"
    assert cfg.generation.generator == "mock"


def test_roundtrip_dict():
    cfg = dict_to_config({"name": "x", "retrieval": {"method": "bm25", "top_k": 2}})
    again = dict_to_config(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()
    assert again.retrieval.top_k == 2


def test_hash_is_deterministic_and_config_sensitive():
    a = dict_to_config({"name": "a", "retrieval": {"top_k": 4}})
    b = dict_to_config({"name": "a", "retrieval": {"top_k": 4}})
    c = dict_to_config({"name": "a", "retrieval": {"top_k": 8}})
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_unknown_key_raises():
    with pytest.raises(ConfigError):
        dict_to_config({"retrieval": {"nope": 1}})


def test_bad_enum_raises():
    with pytest.raises(ConfigError):
        dict_to_config({"retrieval": {"method": "magic"}})
    with pytest.raises(ConfigError):
        dict_to_config({"generation": {"modality": "hologram"}})


def test_text_modality_needs_parser():
    with pytest.raises(ConfigError):
        dict_to_config({
            "representation": {"parser": "none"},
            "generation": {"modality": "text"},
        })


def test_defaults_are_valid():
    RunConfig().validate()


def test_tier1_enums_validate():
    # valid Tier-1 toggles round-trip
    cfg = dict_to_config({"retrieval": {
        "method": "bm25", "k_strategy": "gmm", "expand": "adjacent",
        "expand_window": 2, "rerank": "llm"}})
    assert cfg.retrieval.k_strategy == "gmm"
    # bad enum values raise
    for bad in ({"k_strategy": "psychic"}, {"rerank": "telepathy"},
                {"expand": "teleport"}):
        with pytest.raises(ConfigError):
            dict_to_config({"retrieval": {"method": "bm25", **bad}})


def test_tier1_numeric_guards():
    with pytest.raises(ConfigError):
        dict_to_config({"retrieval": {"method": "bm25", "expand": "adjacent",
                                      "expand_window": 0}})
    with pytest.raises(ConfigError):
        dict_to_config({"retrieval": {"method": "bm25", "candidate_k": -1}})
    # rerank is meaningless for the baseline selectors
    with pytest.raises(ConfigError):
        dict_to_config({"retrieval": {"method": "oracle", "rerank": "llm"}})
