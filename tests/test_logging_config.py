import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import get_logger, setup_logging


def _reset_root_logger():
    root = logging.getLogger("adaptive_rag")
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)


def test_setup_logging_defaults_to_info(monkeypatch):
    _reset_root_logger()
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    setup_logging()
    assert logging.getLogger("adaptive_rag").level == logging.INFO


def test_setup_logging_respects_debug(monkeypatch):
    _reset_root_logger()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setup_logging()
    assert logging.getLogger("adaptive_rag").level == logging.DEBUG


def test_setup_logging_invalid_level_falls_back_to_info(monkeypatch):
    _reset_root_logger()
    monkeypatch.setenv("LOG_LEVEL", "banana")
    setup_logging()
    assert logging.getLogger("adaptive_rag").level == logging.INFO


def test_setup_logging_is_idempotent(monkeypatch):
    _reset_root_logger()
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging()
    n_after_first = len(logging.getLogger("adaptive_rag").handlers)
    setup_logging()
    setup_logging()
    n_after_repeat = len(logging.getLogger("adaptive_rag").handlers)
    assert n_after_first == n_after_repeat  # no duplicate handlers
    assert n_after_first >= 1


def test_get_logger_is_namespaced():
    log = get_logger("router")
    assert log.name == "adaptive_rag.router"


if __name__ == "__main__":
    print("run via pytest (uses monkeypatch)")
