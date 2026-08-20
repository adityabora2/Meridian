from __future__ import annotations

import logging
import os

# The single logger namespace for the whole application. Every module gets a
# child logger (e.g. "adaptive_rag.router") via get_logger, so one setup_logging
# call configures all of them.
_ROOT_NAME = "adaptive_rag"
_HANDLER_TAG = "adaptive_rag_console"


def setup_logging() -> None:
    """Configure structured console logging for the application.

    Reads the LOG_LEVEL environment variable (default INFO; invalid values fall
    back to INFO). Idempotent: safe to call repeatedly (Streamlit reruns the
    whole script on every interaction) -- it detects its own already-attached
    handler and never adds a duplicate.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)

    # Idempotency is keyed on the actual handler state (tagged with a sentinel
    # name), not a module flag -- so it stays correct even if handlers are
    # cleared and re-set (e.g. in tests) or the module is reimported.
    already = any(getattr(h, "_tag", None) == _HANDLER_TAG for h in logger.handlers)
    if not already:
        handler = logging.StreamHandler()  # stderr by default
        handler._tag = _HANDLER_TAG  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        # Don't propagate to the root logger -- keeps our lines from being
        # duplicated or entangled with third-party (e.g. huggingface) handlers.
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return the namespaced child logger for a module, e.g.
    get_logger("router") -> logging.getLogger("adaptive_rag.router")."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
