import asyncio

import pytest
from aiohttp import web

from mantle_proxy.server import (
    Config,
    ConfigError,
    auth_middleware,
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


def test_chat_to_responses_converts_tool_choice_and_response_format():
    request = {
        "model": "openai.test",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lookup_response",
                "description": "A lookup result",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }

    converted = chat_to_responses(request, "fallback")

    assert converted["tool_choice"] == {"type": "function", "name": "lookup"}
    assert converted["text"] == {
        "format": {
            "type": "json_schema",
            "name": "lookup_response",
            "description": "A lookup result",
            "schema": {"type": "object"},
            "strict": True,
        }
    }


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


def test_auth_middleware_requires_matching_bearer_token():
    request = FakeRequest({"Authorization": "Bearer wrong"})

    response = asyncio.run(auth_middleware(request, ok_handler))

    assert response.status == 401


def test_auth_middleware_accepts_matching_bearer_token():
    request = FakeRequest({"Authorization": "Bearer secret"})

    response = asyncio.run(auth_middleware(request, ok_handler))

    assert response.status == 204


class FakeRequest:
    def __init__(self, headers):
        self.app = {"config": _config(proxy_api_key="secret")}
        self.headers = headers


async def ok_handler(request):
    return web.Response(status=204)


def _config(proxy_api_key=""):
    return Config(
        region="us-east-2",
        aws_service="bedrock",
        base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        host="127.0.0.1",
        port=4010,
        default_model="openai.gpt-5.5",
        proxy_api_key=proxy_api_key,
        allow_insecure_remote=False,
        request_timeout_s=120.0,
        max_body_bytes=20 * 1024 * 1024,
        log_level="INFO",
    )
