from __future__ import annotations

import time
from functools import lru_cache

try:
    from src import config
    from src.logging_config import get_logger
except ImportError:
    import config  # type: ignore
    from logging_config import get_logger  # type: ignore

log = get_logger("llm")


class LLMConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    import ollama

    try:
        ollama.list()
    except Exception as exc:
        raise LLMConfigError(
            "Cannot reach Ollama. Make sure it's running (`ollama serve`) and the "
            f"model is pulled (`ollama pull {config.OLLAMA_MODEL}`). "
            f"Original error: {exc}"
        ) from exc
    return ollama


def chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    label: str | None = None,
) -> str:
    client = _client()
    start = time.monotonic()
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "num_predict": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        },
    )
    elapsed = time.monotonic() - start
    tag = f"chat({label})" if label else "chat()"
    log.info("%s took %.1fs", tag, elapsed)
    return (resp["message"]["content"] or "").strip()
