import datetime
import importlib
import logging
import sys
import types

import pytest

LOGGER_NAME = "mantle_proxy.acp_bridge"
MODULE_NAME = "litellm_callback"


def load_callback(monkeypatch, timeout=None, url=None, api_key=None, status=200):
    """Import litellm_callback with litellm/httpx stubbed out.

    The module lives at the repo root (it is loaded by LiteLLM, not by the proxy
    package), and imports litellm at module scope, so the import must be
    performed under monkeypatched sys.modules.
    """
    for name, value in (("ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS", timeout),
                        ("ACP_BRIDGE_CALLBACK_URL", url),
                        ("ACP_BRIDGE_CALLBACK_API_KEY", api_key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    calls = []
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(calls, status))
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    monkeypatch.setitem(sys.modules, "litellm.integrations",
                        types.ModuleType("litellm.integrations"))
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger",
                        fake_custom_logger_module())
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)
    module._test_calls = calls
    return module


def fake_httpx(calls, status=200, raises=None):
    module = types.ModuleType("httpx")

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if raises:
            raise raises
        return types.SimpleNamespace(status_code=status)

    module.post = post
    return module


def fake_custom_logger_module():
    module = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    module.CustomLogger = CustomLogger
    return module


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_callback_defaults(monkeypatch):
    module = load_callback(monkeypatch)
    assert module.CALLBACK_TIMEOUT == 5.0
    assert module.CALLBACK_URL == "http://127.0.0.1:18010/internal/llm-callback"
    assert module.CALLBACK_API_KEY == ""


@pytest.mark.parametrize("bad", ["not-a-number", "", "0", "-1"])
def test_callback_timeout_falls_back_for_invalid_env(monkeypatch, bad):
    """A bad timeout must not disable usage reporting or raise at import."""
    module = load_callback(monkeypatch, timeout=bad)
    assert module.CALLBACK_TIMEOUT == 5.0


def test_callback_timeout_honours_valid_env(monkeypatch):
    module = load_callback(monkeypatch, timeout="12.5")
    assert module.CALLBACK_TIMEOUT == 12.5


def test_callback_url_override(monkeypatch):
    module = load_callback(monkeypatch, url="http://127.0.0.1:19999/hook")
    assert module.CALLBACK_URL == "http://127.0.0.1:19999/hook"


# --------------------------------------------------------------------------
# Delivery and error reporting
# --------------------------------------------------------------------------


def test_callback_logs_http_error_status(monkeypatch, caplog):
    module = load_callback(monkeypatch, status=500)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})

    assert "status=500" in caplog.text


def test_callback_does_not_log_on_success(monkeypatch, caplog):
    module = load_callback(monkeypatch, status=204)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})

    assert caplog.text == ""


def test_callback_logs_transport_exception_without_raising(monkeypatch, caplog):
    module = load_callback(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "httpx",
                        fake_httpx(calls, raises=OSError("connection refused")))
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})

    assert "callback failed" in caplog.text
    assert "OSError" in caplog.text


def test_callback_sends_bearer_when_api_key_set(monkeypatch):
    module = load_callback(monkeypatch, api_key="s3cr3t-for-test")
    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})

    sent = module._test_calls[-1]
    assert sent["headers"]["Authorization"] == "Bearer s3cr3t-for-test"
    assert sent["timeout"] == 5.0


def test_callback_omits_headers_when_no_api_key(monkeypatch):
    module = load_callback(monkeypatch)
    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})
    assert module._test_calls[-1]["headers"] is None


# --------------------------------------------------------------------------
# Cache token extraction — both real upstream shapes
# --------------------------------------------------------------------------


class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _emit(module, usage, model="m"):
    start = datetime.datetime(2026, 1, 1, 0, 0, 0)
    end = datetime.datetime(2026, 1, 1, 0, 0, 2)
    module.AcpBridgeLogger()._log({"model": model},
                                  types.SimpleNamespace(usage=usage), start, end)
    return module._test_calls[-1]["json"]


def test_extracts_cache_tokens_from_anthropic_shape(monkeypatch):
    """Anthropic Messages: top-level cache_read_input_tokens."""
    module = load_callback(monkeypatch)
    payload = _emit(module, _Usage(prompt_tokens=17, completion_tokens=4, total_tokens=21,
                                   cache_read_input_tokens=12400,
                                   cache_creation_input_tokens=0))
    assert payload["cached_tokens"] == 12400
    assert payload["cache_creation_tokens"] == 0
    assert payload["response_time"] == 2.0


def test_extracts_cache_tokens_from_prompt_tokens_details_dict(monkeypatch):
    """Responses via mantle-proxy: converted into prompt_tokens_details."""
    module = load_callback(monkeypatch)
    payload = _emit(module, _Usage(
        prompt_tokens=5741, completion_tokens=16, total_tokens=5757,
        prompt_tokens_details={"cached_tokens": 5739, "cache_write_tokens": 0}))
    assert payload["cached_tokens"] == 5739


def test_extracts_cache_write_tokens_on_cold_request(monkeypatch):
    module = load_callback(monkeypatch)
    payload = _emit(module, _Usage(
        prompt_tokens=5741, completion_tokens=16, total_tokens=5757,
        prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 5739}))
    assert payload["cached_tokens"] == 0
    assert payload["cache_creation_tokens"] == 5739


def test_extracts_cache_tokens_from_prompt_tokens_details_object(monkeypatch):
    """LiteLLM wraps details in PromptTokensDetailsWrapper, not a dict."""
    module = load_callback(monkeypatch)
    payload = _emit(module, _Usage(
        prompt_tokens=1000, completion_tokens=20, total_tokens=1020,
        prompt_tokens_details=_Usage(cached_tokens=800, cache_write_tokens=0)))
    assert payload["cached_tokens"] == 800


def test_no_usage_sends_nothing(monkeypatch):
    module = load_callback(monkeypatch)
    module.AcpBridgeLogger()._log({"model": "m"},
                                  types.SimpleNamespace(usage=None), None, None)
    assert module._test_calls == []


def test_usage_without_cache_fields_reports_zero(monkeypatch):
    module = load_callback(monkeypatch)
    payload = _emit(module, _Usage(prompt_tokens=8, completion_tokens=5, total_tokens=13))
    assert payload["cached_tokens"] == 0
    assert payload["cache_creation_tokens"] == 0


# --------------------------------------------------------------------------
# Fable 5 thinking patch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["anthropic.claude-fable-5",
                                   "us.anthropic.claude-fable-5",
                                   "mantle/anthropic.claude-fable-5"])
def test_thinking_patched_for_fable_models(monkeypatch, model):
    module = load_callback(monkeypatch)
    data = {"model": model, "thinking": {"type": "enabled"},
            "optional_params": {"thinking": {"type": "enabled"}},
            "kwargs": {"thinking": {"type": "enabled"}}}

    out = module._patch_thinking(data)

    assert out["thinking"]["type"] == "adaptive"
    assert out["optional_params"]["thinking"]["type"] == "adaptive"
    assert out["kwargs"]["thinking"]["type"] == "adaptive"


def test_thinking_untouched_for_other_models(monkeypatch):
    module = load_callback(monkeypatch)
    data = {"model": "anthropic.claude-haiku-4-5", "thinking": {"type": "enabled"}}
    assert module._patch_thinking(data)["thinking"]["type"] == "enabled"
