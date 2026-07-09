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
    from groq import Groq

    key = config.GROQ_API_KEY
    if not key or key == "your_groq_api_key_here":
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )
    return Groq(api_key=key, max_retries=5, timeout=30.0)


def chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
