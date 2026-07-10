from __future__ import annotations

from functools import lru_cache

try:
    from src import config
except ImportError:
    import config  # type: ignore


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
) -> str:
    client = _client()
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
    return (resp["message"]["content"] or "").strip()
