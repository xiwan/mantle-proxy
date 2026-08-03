"""LiteLLM Custom Callback — posts usage (incl. cache tokens) to ACP Bridge."""

import logging
import os

import httpx
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("mantle_proxy.acp_bridge")

DEFAULT_CALLBACK_URL = "http://127.0.0.1:18010/internal/llm-callback"
DEFAULT_CALLBACK_TIMEOUT = 5.0

CALLBACK_URL = os.getenv("ACP_BRIDGE_CALLBACK_URL", DEFAULT_CALLBACK_URL)
CALLBACK_API_KEY = os.getenv("ACP_BRIDGE_CALLBACK_API_KEY", "")


def _timeout_from_env() -> float:
    """Read the callback timeout, falling back on an unusable value.

    A bad timeout must not disable usage reporting, and must not raise at import
    time inside LiteLLM's callback loader.
    """
    raw = os.getenv("ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_CALLBACK_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("invalid ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS=%r, using %s",
                    raw, DEFAULT_CALLBACK_TIMEOUT)
        return DEFAULT_CALLBACK_TIMEOUT
    if value <= 0:
        log.warning("non-positive ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS=%r, using %s",
                    raw, DEFAULT_CALLBACK_TIMEOUT)
        return DEFAULT_CALLBACK_TIMEOUT
    return value


CALLBACK_TIMEOUT = _timeout_from_env()

# Models that reject thinking.type="enabled"
_FABLE_MODELS = {"us.anthropic.claude-fable-5", "anthropic.claude-fable-5"}


def _patch_thinking(data: dict) -> dict:
    """Convert thinking.type=enabled to adaptive for Fable 5 models."""
    model = data.get("model", "")
    if not any(m in model for m in _FABLE_MODELS):
        return data
    t = data.get("thinking")
    if isinstance(t, dict) and t.get("type") == "enabled":
        t["type"] = "adaptive"
    op = data.get("optional_params")
    if isinstance(op, dict):
        t2 = op.get("thinking")
        if isinstance(t2, dict) and t2.get("type") == "enabled":
            t2["type"] = "adaptive"
    kw = data.get("kwargs")
    if isinstance(kw, dict):
        t3 = kw.get("thinking")
        if isinstance(t3, dict) and t3.get("type") == "enabled":
            t3["type"] = "adaptive"
    return data


def _extract_cache_tokens(usage) -> tuple[int, int]:
    """Pull (cached, cache_creation) out of a LiteLLM usage object.

    Two upstream shapes reach here through mantle-proxy:
      - Anthropic Messages  -> top-level cache_read_input_tokens /
                               cache_creation_input_tokens
      - Responses / OpenAI  -> prompt_tokens_details.cached_tokens
                               (mantle-proxy converts input_tokens_details into
                               this shape; Mantle itself does not emit it)
    """
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    if not cached:
        ptd = getattr(usage, "prompt_tokens_details", None)
        if isinstance(ptd, dict):
            cached = ptd.get("cached_tokens", 0) or 0
        elif ptd is not None:
            cached = getattr(ptd, "cached_tokens", 0) or 0

    creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if not creation:
        ptd = getattr(usage, "prompt_tokens_details", None)
        if isinstance(ptd, dict):
            creation = ptd.get("cache_write_tokens", 0) or 0
        elif ptd is not None:
            creation = getattr(ptd, "cache_write_tokens", 0) or 0

    return int(cached), int(creation)


class AcpBridgeLogger(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return _patch_thinking(data)

    async def async_log_pre_api_call(self, model, messages, kwargs):
        """Patch thinking right before the actual API call."""
        _patch_thinking(kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log(kwargs, response_obj, start_time, end_time)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log(kwargs, response_obj, start_time, end_time)

    def _post_sync(self, payload: dict) -> None:
        """POST the payload. Never raises; failures are logged, not swallowed."""
        headers = {}
        if CALLBACK_API_KEY:
            headers["Authorization"] = f"Bearer {CALLBACK_API_KEY}"
        try:
            resp = httpx.post(CALLBACK_URL, json=payload,
                              timeout=CALLBACK_TIMEOUT,
                              headers=headers or None)
        except Exception as exc:
            log.warning("acp_bridge callback failed: %s", type(exc).__name__)
            return
        status = getattr(resp, "status_code", 0)
        if not 200 <= status < 300:
            log.warning("acp_bridge callback rejected: status=%s", status)

    def _log(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "")
        usage = getattr(response_obj, "usage", None)
        if not usage:
            return

        cached_tokens, cache_creation = _extract_cache_tokens(usage)
        duration = (end_time - start_time).total_seconds() if start_time and end_time else 0.0

        self._post_sync({
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "cached_tokens": cached_tokens,
            "cache_creation_tokens": cache_creation,
            "response_time": duration,
        })


proxy_handler_instance = AcpBridgeLogger()
