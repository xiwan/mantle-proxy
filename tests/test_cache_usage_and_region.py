"""Tests for prompt cache usage passthrough and region override validation."""

import pytest
from aiohttp import web

from mantle_proxy.server import (
    REGION_OVERRIDE_HEADER,
    Config,
    SigV4Client,
    chat_to_responses,
    _extract_forward_headers,
    build_usage,
    is_valid_region,
    responses_to_chat,
    uses_responses_api,
)


def _config(region="us-east-2"):
    return Config(
        region=region,
        aws_service="bedrock",
        base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
        host="127.0.0.1",
        port=4010,
        default_model="openai.gpt-5.5",
        proxy_api_key="",
        allow_insecure_remote=False,
        request_timeout_s=120.0,
        max_body_bytes=20 * 1024 * 1024,
        log_level="INFO",
    )


class FakeRequest:
    def __init__(self, headers):
        self.headers = headers


# --------------------------------------------------------------------------
# Patch 1: prompt cache usage passthrough
# --------------------------------------------------------------------------


def _response_with_usage(usage):
    return {
        "id": "resp_1",
        "model": "openai.gpt-5.6-sol",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": usage,
    }


def test_responses_to_chat_preserves_input_tokens_details():
    """Mantle Responses API shape: usage.input_tokens_details."""
    converted = responses_to_chat(
        _response_with_usage({
            "input_tokens": 3671,
            "output_tokens": 12,
            "total_tokens": 3683,
            "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0},
        }),
        "fallback",
    )

    assert converted["usage"]["prompt_tokens"] == 3671
    assert converted["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 3626,
        "cache_write_tokens": 0,
    }


def test_responses_to_chat_preserves_prompt_tokens_details():
    """OpenAI Chat Completions shape: usage.prompt_tokens_details."""
    converted = responses_to_chat(
        _response_with_usage({
            "input_tokens": 1000,
            "output_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 0},
        }),
        "fallback",
    )

    assert converted["usage"]["prompt_tokens_details"]["cached_tokens"] == 800


def test_responses_to_chat_preserves_cold_cache_write():
    """A cold request writes the cache: cached=0 but cache_write>0 must survive."""
    converted = responses_to_chat(
        _response_with_usage({
            "input_tokens": 3682,
            "output_tokens": 9,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3626},
        }),
        "fallback",
    )

    assert converted["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 0,
        "cache_write_tokens": 3626,
    }


def test_responses_to_chat_omits_cache_details_when_no_caching():
    """No cache activity must not add noise, preserving the pre-patch contract."""
    converted = responses_to_chat(
        _response_with_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        "fallback",
    )

    assert converted["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_responses_to_chat_maps_anthropic_top_level_cache_fields():
    converted = responses_to_chat(
        _response_with_usage({
            "input_tokens": 500,
            "output_tokens": 5,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 100,
            "input_tokens_details": {},
        }),
        "fallback",
    )

    assert converted["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 400,
        "cache_write_tokens": 100,
    }


def test_build_usage_tolerates_non_dict_details():
    assert build_usage({"input_tokens_details": "garbage"}, 1, 2, 3) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_cache_hit_rate_computable_from_converted_usage():
    """End-to-end: the three requests from the AWS GPT-5.6 caching blog.

    Verifies a downstream consumer can now derive a token-weighted cache rate
    from the proxy's Chat Completions output.
    """
    upstream = [
        {"input_tokens": 3682, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3626}},
        {"input_tokens": 3671, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0}},
        {"input_tokens": 3662, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0}},
    ]
    converted = [responses_to_chat(_response_with_usage(u), "fallback") for u in upstream]

    cached = sum(c["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                 for c in converted)
    total_input = sum(c["usage"]["prompt_tokens"] for c in converted)
    hits = sum(1 for c in converted
               if c["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0)

    assert cached == 7252
    assert total_input == 11015
    assert cached / total_input == pytest.approx(0.6584, abs=1e-4)
    assert hits / len(converted) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Patch 2: region override validation (SSRF)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("region", [
    "us-east-1", "us-west-2", "ap-southeast-1", "ap-northeast-3",
    "eu-central-1", "ca-central-1", "us-gov-west-1", "cn-north-1",
    "me-central-1", "il-central-1", "ap-southeast-4",
])
def test_is_valid_region_accepts_canonical_regions(region):
    assert is_valid_region(region) is True


@pytest.mark.parametrize("region", [
    "evil.com/x?",                      # SSRF: host takeover via path/query
    "us-east-1.evil.com",               # SSRF: subdomain suffix
    "us-east-1/../../foo",              # path traversal
    "us-east-1?x=",                     # query injection
    "us-east-1#frag",
    "us-east-1:8080",                   # port override
    "US-EAST-1",                        # wrong case
    "us-east-1 ",                       # trailing space
    "us-east-1\nX-Injected: 1",         # header/CRLF injection
    "", "-", "useast1", "us-east-", "1-east-1",
])
def test_is_valid_region_rejects_malicious_or_malformed(region):
    assert is_valid_region(region) is False


def test_is_valid_region_rejects_non_string():
    assert is_valid_region(None) is False


def test_extract_forward_headers_rejects_malicious_region():
    request = FakeRequest({REGION_OVERRIDE_HEADER: "evil.com/x?"})

    with pytest.raises(web.HTTPBadRequest):
        _extract_forward_headers(request)


def test_extract_forward_headers_accepts_valid_region_and_forwards_allowlist():
    request = FakeRequest({
        REGION_OVERRIDE_HEADER: "us-west-2",
        "OpenAI-Project": "proj_123",
        "Authorization": "Bearer should-not-be-forwarded",
        "X-Random": "nope",
    })

    fwd, region = _extract_forward_headers(request)

    assert region == "us-west-2"
    assert fwd == {"OpenAI-Project": "proj_123"}


def test_extract_forward_headers_without_region_override():
    fwd, region = _extract_forward_headers(FakeRequest({}))
    assert (fwd, region) == (None, None)


def test_target_url_rejects_invalid_region_override():
    client = SigV4Client(_config(), http=None)

    with pytest.raises(ValueError):
        client._target_url("responses", region_override="evil.com/x?")


def test_target_url_builds_expected_hosts():
    client = SigV4Client(_config(region="us-east-2"), http=None)

    assert client._target_url("responses") == (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    )
    assert client._target_url("responses", region_override="us-west-2") == (
        "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"
    )
    assert client._target_url("anthropic/v1/messages") == (
        "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages"
    )


# --------------------------------------------------------------------------
# Route selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", [
    "openai.gpt-5.5", "openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna",
])
def test_responses_api_used_for_responses_capable_openai_models(model):
    assert uses_responses_api(model) is True


@pytest.mark.parametrize("model", ["openai.gpt-oss-120b", "openai.gpt-oss-20b"])
def test_chat_completions_used_for_gpt_oss(model):
    """Observed upstream: gpt-oss rejects /openai/v1/responses with validation_error.

    Routing on the bare "openai." prefix sent these to /responses and made them
    unreachable, since no other path can target chat/completions for an
    openai.* model.
    """
    assert uses_responses_api(model) is False


@pytest.mark.parametrize("model", [
    "anthropic.claude-haiku-4-5", "moonshotai.kimi-k2.5", "qwen.qwen3-coder-next",
    "", "gpt-5.5", "not-openai.gpt-5.5", None, 123,
])
def test_responses_api_not_used_for_non_openai_models(model):
    assert uses_responses_api(model) is False


# --------------------------------------------------------------------------
# Explicit prompt cache control
# --------------------------------------------------------------------------


def test_chat_to_responses_forwards_prompt_cache_key():
    """Verified upstream: a different prompt_cache_key on an identical prefix is
    a cache miss, so dropping the key collapses all callers into one shared
    implicit partition and removes explicit cache control entirely.
    """
    converted = chat_to_responses({
        "model": "openai.gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt_cache_key": "tenant-42",
        "prompt_cache_retention": "in_memory",
    }, "fallback")

    assert converted["prompt_cache_key"] == "tenant-42"
    assert converted["prompt_cache_retention"] == "in_memory"


def test_chat_to_responses_omits_cache_control_when_absent():
    converted = chat_to_responses({
        "model": "openai.gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hi"}],
    }, "fallback")

    assert "prompt_cache_key" not in converted
    assert "prompt_cache_retention" not in converted
