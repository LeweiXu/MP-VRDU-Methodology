"""Second-dataset interface (S8): alias-tolerant loading of differently-keyed
sample schemas (e.g. LongDocURL) through the same Dataset path."""

import json

import fitz

from mpvrdu.data.dataset import load_samples_json, question_from_sample


def test_alias_fields_map_to_question():
    # LongDocURL-ish keys: doc_no / question_id / answer_type / evidence_page
    s = {"doc_no": "pdfs/report.pdf", "question_id": "q7", "query": "How many?",
         "answers": "5", "evidence_page": "[2, 3]", "answer_type": "Int",
         "task_tag": "Reasoning"}
    q = question_from_sample(s, 0)
    assert q.doc_id == "report.pdf"            # path basename, not "pdfs/report.pdf"
    assert q.qid == "q7"
    assert q.question == "How many?"
    assert q.answer == "5"
    assert q.evidence_pages == [2, 3]
    assert q.answer_format == "Int"
    assert q.doc_type == "Reasoning"


def test_load_dir_with_alias_schema(tmp_path):
    pdf_dir = tmp_path / "documents"
    pdf_dir.mkdir(parents=True)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "hello")
    doc.new_page().insert_text((72, 72), "world")
    doc.save(str(pdf_dir / "d1.pdf"))
    doc.close()

    samples = [
        {"doc_no": "d1.pdf", "question_id": "a", "query": "q1?", "answers": "x",
         "evidence_page": "[1]", "answer_type": "Str"},
        {"doc_no": "d1.pdf", "question_id": "b", "query": "q2?",
         "answers": "Not answerable", "evidence_page": "[]", "answer_type": "None"},
    ]
    (tmp_path / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    ds = load_samples_json(tmp_path / "samples.json", pdf_dir)
    assert len(ds) == 2 and len(ds.documents) == 1
    assert ds.questions[1].is_unanswerable
