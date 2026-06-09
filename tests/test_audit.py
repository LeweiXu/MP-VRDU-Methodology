"""Contamination-audit manifest + overlap check (real PDF hashing)."""

from mpvrdu.data.audit import build_doc_manifest, check_overlap


def test_manifest_fingerprints_docs(synthetic_ds):
    manifest = build_doc_manifest(synthetic_ds)
    assert len(manifest) == 3
    for m in manifest:
        assert m["present"]
        assert len(m["sha256"]) == 64           # sha256 hex
        assert m["num_pages"] >= 1
        assert m["n_questions"] >= 1


def test_manifest_hash_is_stable(synthetic_ds):
    a = build_doc_manifest(synthetic_ds)
    b = build_doc_manifest(synthetic_ds)
    assert {m["doc_id"]: m["sha256"] for m in a} == {m["doc_id"]: m["sha256"] for m in b}


def test_overlap_detects_known_doc(synthetic_ds):
    manifest = build_doc_manifest(synthetic_ds)
    hits = check_overlap(manifest, {"alpha.pdf"})
    assert len(hits) == 1 and hits[0]["doc_id"] == "alpha.pdf"
    assert check_overlap(manifest, {"nonexistent.pdf"}) == []


def test_overlap_by_sha256(synthetic_ds):
    manifest = build_doc_manifest(synthetic_ds)
    sha = manifest[0]["sha256"]
    hits = check_overlap(manifest, {sha})
    assert len(hits) == 1 and hits[0]["sha256"] == sha
