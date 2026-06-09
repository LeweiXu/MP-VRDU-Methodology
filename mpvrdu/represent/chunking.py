"""Chunking (cross-cutting; Stage 4 uses it, Stage 5 sub-study varies it).

Turns parsed pages into retrievable chunks. Each chunk remembers its source
0-based page so retrieval-recall stays page-based regardless of granularity.

Strategies:
- "page":    one chunk per page (the default).
- "chunk":   fixed-size word windows within each page (with overlap).
- "section": one chunk per markdown section, carrying its heading as context;
             a section longer than `chunk_words` is SPLIT into sub-chunks (the
             MultiDocFusion recipe: split only when a section exceeds max length,
             re-prefixing the heading so a retrieved sub-chunk keeps its context)
             but a heading is never split off from its body. Gracefully DEGRADES
             to "page" when the parser gave no structure (context.md / Stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import ParsedPage


@dataclass
class Chunk:
    page_index: int          # 0-based source page (for recall)
    text: str
    section: Optional[str] = None


def _window(words: list[str], size: int, overlap: int):
    if size <= 0:
        yield words
        return
    step = max(1, size - overlap)
    for start in range(0, max(1, len(words)), step):
        piece = words[start:start + size]
        if piece:
            yield piece
        if start + size >= len(words):
            break


def _section_chunks(page_index: int, heading: str, body: str,
                    chunk_words: int, overlap: int) -> list[Chunk]:
    """One chunk per section, splitting only when the BODY exceeds chunk_words.

    The heading is re-prefixed onto every sub-chunk (so it never gets split off
    and each piece keeps its hierarchical context), and counts toward neither the
    window size nor the split decision — only the body length triggers a split.
    """
    sec = heading or None
    words = body.split()
    if not words and not heading:
        return []
    if len(words) <= chunk_words or chunk_words <= 0:
        text = (f"{heading}\n{body}" if heading else body).strip()
        return [Chunk(page_index=page_index, text=text, section=sec)] if text else []
    out: list[Chunk] = []
    for piece in _window(words, chunk_words, overlap):
        piece_body = " ".join(piece)
        text = (f"{heading}\n{piece_body}" if heading else piece_body).strip()
        if text:
            out.append(Chunk(page_index=page_index, text=text, section=sec))
    return out


def chunk_pages(pages: list[ParsedPage], strategy: str = "page",
                chunk_words: int = 180, overlap: int = 30) -> list[Chunk]:
    if strategy == "page":
        return [Chunk(page_index=p.page_index, text=p.text) for p in pages]

    if strategy == "chunk":
        chunks: list[Chunk] = []
        for p in pages:
            words = p.text.split()
            if not words:
                chunks.append(Chunk(page_index=p.page_index, text=p.text))
                continue
            for piece in _window(words, chunk_words, overlap):
                chunks.append(Chunk(page_index=p.page_index, text=" ".join(piece)))
        return chunks

    if strategy == "section":
        chunks = []
        any_structure = False
        for p in pages:
            if p.sections:
                any_structure = True
                for heading, body in p.sections:
                    chunks.extend(_section_chunks(p.page_index, heading, body,
                                                  chunk_words, overlap))
            else:
                chunks.append(Chunk(page_index=p.page_index, text=p.text))
        if not any_structure:
            # graceful degradation: no parser structure -> page-level
            return chunk_pages(pages, "page")
        return chunks

    raise ValueError(f"unknown chunking strategy {strategy!r}")
