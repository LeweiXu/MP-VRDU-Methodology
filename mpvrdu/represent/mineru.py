"""MinerU parser adapter (layout/table-aware, ML-based; Apache-2.0).

MinerU 2+ is installed in a separate environment because its Transformers
constraint conflicts with the ColPali/Qwen stack used by this repository.
The worker normalizes MinerU's middle JSON into cached per-page Markdown.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from mpvrdu.env import CACHE_ROOT, REPO_ROOT
from mpvrdu.logging_utils import get_logger

from .base import Parser, ParsedPage, _markdown_sections

log = get_logger(__name__)
_CACHE_VERSION = "mineru-v3-pages-v1"


class MinerUParser(Parser):
    name = "mineru"
    structure_aware = True

    def __init__(self, python: str | Path | None = None):
        configured = python or os.environ.get("MPVRDU_MINERU_PYTHON")
        self.python = Path(configured or Path.home() / "venvs/mineru-venv/bin/python")
        if not self.python.is_file():
            raise ImportError(
                "MinerU environment not found. Install `mineru[pipeline]` in "
                "~/venvs/mineru-venv, or set MPVRDU_MINERU_PYTHON to its Python."
            )
        self.worker = Path(
            os.environ.get(
                "MPVRDU_MINERU_WORKER",
                REPO_ROOT / "scripts/mineru_worker.py",
            )
        )

    def _cache_key(self, pdf_path: Path) -> str:
        stat = pdf_path.stat()
        identity = (
            f"{_CACHE_VERSION}\0{pdf_path.resolve()}\0"
            f"{stat.st_size}\0{stat.st_mtime_ns}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:20]

    def _markdown_by_page(self, pdf_path: str | Path) -> list[str]:
        pdf_path = Path(pdf_path)
        key = self._cache_key(pdf_path)
        cache_dir = CACHE_ROOT / "mineru"
        pages_path = cache_dir / "pages" / f"{key}.json"
        if not pages_path.exists():
            pages_path.parent.mkdir(parents=True, exist_ok=True)
            output_dir = cache_dir / "runs" / key
            command = [
                str(self.python),
                str(self.worker),
                "--pdf",
                str(pdf_path.resolve()),
                "--output-dir",
                str(output_dir),
                "--output-json",
                str(pages_path),
            ]
            log.info("MinerU parsing %s (cache=%s)", pdf_path, pages_path)
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
                log.error("MinerU process output for %s:\n%s", pdf_path, output.rstrip())
                raise RuntimeError(
                    f"MinerU failed for {pdf_path} with exit code "
                    f"{exc.returncode}; inspect the current run log."
                ) from exc
            output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )
            if output:
                log.info("MinerU process output for %s:\n%s",
                         pdf_path, output.rstrip())

        try:
            pages = json.loads(pages_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid MinerU page cache: {pages_path}") from exc
        if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
            raise RuntimeError(f"invalid MinerU page cache schema: {pages_path}")
        return pages

    def parse_document(self, pdf_path: str | Path) -> list[ParsedPage]:
        return [
            ParsedPage(
                page_index=index,
                text=markdown,
                markdown=markdown,
                sections=_markdown_sections(markdown),
            )
            for index, markdown in enumerate(self._markdown_by_page(pdf_path))
        ]

    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        pages = self._markdown_by_page(pdf_path)
        return pages[page_index] if 0 <= page_index < len(pages) else ""
