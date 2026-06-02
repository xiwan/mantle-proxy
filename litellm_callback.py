"""LiteLLM custom callback that posts usage data to ACP Bridge.

The callback intentionally sends token counts and timing only. It does not send
prompts, completions, request headers, API keys, or AWS credentials.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("mantle_proxy.litellm_callback")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("invalid_float_env name=%s using_default=%s", name, default)
        return default


CALLBACK_URL = os.getenv("ACP_BRIDGE_CALLBACK_URL", "http://127.0.0.1:18010/internal/llm-callback")
CALLBACK_API_KEY = os.getenv("ACP_BRIDGE_CALLBACK_API_KEY", "")
CALLBACK_TIMEOUT = _env_float("ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS", 5.0)


class AcpBridgeLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = self._payload(kwargs, response_obj, start_time, end_time)
        if payload:
            await self._post_async(payload)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = self._payload(kwargs, response_obj, start_time, end_time)
        if payload:
            self._post_sync(payload)

    def _payload(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "")
        usage = getattr(response_obj, "usage", None)
        if not usage:
            return None

        prompt_tokens = _usage_get(usage, "prompt_tokens", 0) or 0
        completion_tokens = _usage_get(usage, "completion_tokens", 0) or 0
        total_tokens = _usage_get(usage, "total_tokens", 0) or 0

        # Bedrock returns cache_read_input_tokens / cache_creation_input_tokens
        # OpenAI returns prompt_tokens_details.cached_tokens
        cached_tokens = _usage_get(usage, "cache_read_input_tokens", 0) or 0
        if not cached_tokens:
            ptd = _usage_get(usage, "prompt_tokens_details", None)
            if ptd:
                cached_tokens = _usage_get(ptd, "cached_tokens", 0) or 0

        cache_creation = _usage_get(usage, "cache_creation_input_tokens", 0) or 0

        duration = (end_time - start_time).total_seconds() if start_time and end_time else 0.0

        return {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_creation_tokens": cache_creation,
            "response_time": duration,
        }

    async def _post_async(self, payload):
        headers = _headers()
        try:
            async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
                response = await client.post(CALLBACK_URL, json=payload, headers=headers)
                if response.status_code >= 400:
                    log.warning("acp_bridge_callback_failed status=%s", response.status_code)
        except Exception as exc:
            log.warning("acp_bridge_callback_failed: %s", exc.__class__.__name__)

    def _post_sync(self, payload):
        try:
            response = httpx.post(CALLBACK_URL, json=payload, headers=_headers(), timeout=CALLBACK_TIMEOUT)
            if response.status_code >= 400:
                log.warning("acp_bridge_callback_failed status=%s", response.status_code)
        except Exception as exc:
            log.warning("acp_bridge_callback_failed: %s", exc.__class__.__name__)


def _usage_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _headers() -> dict[str, str]:
    if CALLBACK_API_KEY:
        return {"Authorization": f"Bearer {CALLBACK_API_KEY}"}
    return {}


proxy_handler_instance = AcpBridgeLogger()
