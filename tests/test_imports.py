"""Guard: every module imports cleanly (catches lazy-import regressions).

Heavy deps (torch/transformers/colpali) are lazy-imported inside functions, so
importing these modules must never require the GPU stack.
"""

import importlib

import pytest

MODULES = [
    "mpvrdu", "mpvrdu.config", "mpvrdu.env", "mpvrdu.results",
    "mpvrdu.logging_utils", "mpvrdu.pipeline", "mpvrdu.experiment",
    "mpvrdu.data.dataset", "mpvrdu.data.render", "mpvrdu.data.slice",
    "mpvrdu.data.synthetic", "mpvrdu.data.load", "mpvrdu.data.audit",
    "mpvrdu.represent.base", "mpvrdu.represent.chunking",
    "mpvrdu.represent.mineru", "mpvrdu.represent.tesseract",
    "mpvrdu.retrieve.base", "mpvrdu.retrieve.retrievers", "mpvrdu.retrieve.hybrid",
    "mpvrdu.retrieve.factory", "mpvrdu.retrieve.evaluate",
    "mpvrdu.generate.base", "mpvrdu.generate.mock", "mpvrdu.generate.vlm",
    "mpvrdu.generate.factory",
    "mpvrdu.eval.extract", "mpvrdu.eval.metrics", "mpvrdu.eval.judge",
    "mpvrdu.analysis.aggregate", "mpvrdu.analysis.figures", "mpvrdu.analysis.report",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
