import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from app.config import settings

logger = logging.getLogger("shl_recommender.llm")

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key or "unset",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    return _client


class LLMError(Exception):
    pass


def _messages_payload(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call the LLM and force a JSON object response. Retries on transient
    errors. Raises LLMError if the response cannot be parsed as JSON after
    all retries -- callers must handle this with a safe fallback."""
    client = get_client()
    temp = temperature if temperature is not None else settings.llm_temperature_extract
    last_err: Optional[Exception] = None

    for attempt in range(settings.llm_max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=_messages_payload(system_prompt, user_prompt),
                temperature=temp,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except (json.JSONDecodeError, APIError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            logger.warning("LLM json call failed (attempt %s): %s", attempt, e)
            time.sleep(min(0.5 * (attempt + 1), 2.0))
        except Exception as e:  # noqa: BLE001 - defensive catch-all for provider quirks
            last_err = e
            logger.warning("LLM json call unexpected error (attempt %s): %s", attempt, e)
            time.sleep(min(0.5 * (attempt + 1), 2.0))

    raise LLMError(f"LLM JSON call failed after retries: {last_err}")


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
) -> str:
    client = get_client()
    temp = temperature if temperature is not None else settings.llm_temperature_generate
    last_err: Optional[Exception] = None

    for attempt in range(settings.llm_max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=_messages_payload(system_prompt, user_prompt),
                temperature=temp,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            logger.warning("LLM text call failed (attempt %s): %s", attempt, e)
            time.sleep(min(0.5 * (attempt + 1), 2.0))
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("LLM text call unexpected error (attempt %s): %s", attempt, e)
            time.sleep(min(0.5 * (attempt + 1), 2.0))

    raise LLMError(f"LLM text call failed after retries: {last_err}")
