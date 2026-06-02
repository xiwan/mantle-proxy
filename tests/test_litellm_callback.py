import importlib
import logging
import sys
import types


def test_callback_timeout_falls_back_for_invalid_env(monkeypatch):
    module = load_callback(monkeypatch, timeout="not-a-number")

    assert module.CALLBACK_TIMEOUT == 5.0


def test_callback_logs_http_error_status(monkeypatch, caplog):
    module = load_callback(monkeypatch)
    caplog.set_level(logging.WARNING, logger="mantle_proxy.integrations.acp_bridge")

    module.AcpBridgeLogger()._post_sync({"model": "openai.test"})

    assert "status=500" in caplog.text


def load_callback(monkeypatch, timeout="5"):
    monkeypatch.setenv("ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS", timeout)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx())
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    monkeypatch.setitem(sys.modules, "litellm.integrations", types.ModuleType("litellm.integrations"))
    monkeypatch.setitem(
        sys.modules,
        "litellm.integrations.custom_logger",
        fake_custom_logger_module(),
    )
    sys.modules.pop("mantle_proxy.integrations.acp_bridge.litellm_callback", None)
    return importlib.import_module("mantle_proxy.integrations.acp_bridge.litellm_callback")


def fake_httpx():
    module = types.ModuleType("httpx")

    def post(*args, **kwargs):
        return types.SimpleNamespace(status_code=500)

    module.post = post
    return module


def fake_custom_logger_module():
    module = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    module.CustomLogger = CustomLogger
    return module
