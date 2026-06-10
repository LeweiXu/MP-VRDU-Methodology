import json
from pathlib import Path

import pytest

from mpvrdu.represent import mineru


def test_mineru_requires_worker_environment(tmp_path):
    with pytest.raises(ImportError, match="MPVRDU_MINERU_PYTHON"):
        mineru.MinerUParser(python=tmp_path / "missing-python")


def test_mineru_runs_worker_and_reuses_cache(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")
    cache = tmp_path / "cache"
    monkeypatch.setattr(mineru, "CACHE_ROOT", cache)
    calls = []

    class Completed:
        stdout = "worker output"
        stderr = ""

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        output = Path(command[command.index("--output-json") + 1])
        output.write_text(
            json.dumps(["# First\npage one", "second page"]),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(mineru.subprocess, "run", fake_run)
    parser = mineru.MinerUParser(python=python)

    pages = parser.parse_document(pdf)
    assert [page.text for page in pages] == ["# First\npage one", "second page"]
    assert pages[0].sections == [("First", "page one")]
    assert parser.parse_page(pdf, 1) == "second page"
    assert len(calls) == 1
