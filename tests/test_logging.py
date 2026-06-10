import logging

from mpvrdu.logging_utils import file_logging, get_logger


def test_file_logging_writes_and_detaches(tmp_path):
    path = tmp_path / "logs" / "run.log"
    log = get_logger("test_file")

    with file_logging(path, mode="w"):
        log.info("persist this message")
    log.info("do not persist this message")

    text = path.read_text(encoding="utf-8")
    assert "persist this message" in text
    assert "do not persist this message" not in text


def test_file_logging_records_traceback(tmp_path):
    path = tmp_path / "error.log"
    log = get_logger("test_error")

    with file_logging(path, mode="w"):
        try:
            raise RuntimeError("example failure")
        except RuntimeError:
            log.exception("run failed")

    text = path.read_text(encoding="utf-8")
    assert "run failed" in text
    assert "RuntimeError: example failure" in text
