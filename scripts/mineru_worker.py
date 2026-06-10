#!/usr/bin/env python3
"""Run current MinerU and emit normalized per-page Markdown JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    from mineru.cli.common import do_parse
    from mineru.utils.enum_class import MakeMode

    stem = args.pdf.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    do_parse(
        str(args.output_dir),
        [stem],
        [args.pdf.read_bytes()],
        ["en"],
        backend="pipeline",
        parse_method="auto",
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
        f_make_md_mode=MakeMode.MM_MD,
    )

    matches = list((args.output_dir / stem).glob("*/" + stem + "_middle.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one MinerU middle JSON for {args.pdf}, found {matches}"
        )
    middle = json.loads(matches[0].read_text(encoding="utf-8"))
    pages = [
        union_make([page], MakeMode.MM_MD, "images")
        for page in middle["pdf_info"]
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output_json)
    print(f"MinerU parsed {len(pages)} pages from {args.pdf}")


if __name__ == "__main__":
    main()
