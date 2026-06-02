import pytest

from mantle_proxy.server import (
    Config,
    ConfigError,
    chat_to_responses,
    maybe_parse_json_body,
    normalize_path,
    responses_to_chat,
)


def test_normalize_path_strips_v1_prefix():
    assert normalize_path("/v1/responses") == "responses"
    assert normalize_path("v1/models") == "models"
    assert normalize_path("/responses") == "responses"


def test_chat_to_responses_converts_function_tools():
    request = {
        "model": "openai.test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    converted = chat_to_responses(request, "fallback")

    assert converted["model"] == "openai.test"
    assert converted["max_output_tokens"] == 64
    assert converted["input"] == [{"role": "user", "content": "hello"}]
    assert converted["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup a value",
            "parameters": {"type": "object"},
        }
    ]


def test_responses_to_chat_maps_tool_calls_and_usage():
    response = {
        "id": "resp_123",
        "model": "openai.test",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "lookup",
                "arguments": {"query": "abc"},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    converted = responses_to_chat(response, "fallback")

    choice = converted["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert converted["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"query":"abc"}',
    }


def test_remote_bind_requires_proxy_api_key(monkeypatch):
    monkeypatch.setenv("MANTLE_PROXY_HOST", "0.0.0.0")
    monkeypatch.delenv("MANTLE_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MANTLE_ALLOW_INSECURE_REMOTE", raising=False)

    with pytest.raises(ConfigError):
        Config.from_env()


def test_maybe_parse_json_body_handles_missing_content_type():
    assert maybe_parse_json_body(b'{"stream": true}', "") == {"stream": True}
    assert maybe_parse_json_body(b"not-json", "text/plain") is None
    with pytest.raises(ValueError):
        maybe_parse_json_body(b"not-json", "application/json")
