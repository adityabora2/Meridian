"""Shared Groq LLM helper used by every node (router, decompose, generate, critique).

One client, one call signature, one place to handle missing keys and API errors. Nodes
differ by *prompt*, not by how they reach the model — keeping this in one spot is what
lets the demo be explained as "same model, four prompts".
"""

from __future__ import annotations

from functools import lru_cache

try:
    from src import config
except ImportError:  # running from inside src/
    import config  # type: ignore


class LLMConfigError(RuntimeError):
    """Raised when the Groq API key is missing so the UI can show a clear message."""


@lru_cache(maxsize=1)
def _client():
    from groq import Groq

    key = config.GROQ_API_KEY
    if not key or key == "your_groq_api_key_here":
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com/keys)."
        )
    return Groq(api_key=key)


def chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a system+user prompt to Groq and return the assistant's text.

    Thin wrapper: every node calls this so model name, temperature, and token limits
    come from config unless a node overrides them.
    """
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
