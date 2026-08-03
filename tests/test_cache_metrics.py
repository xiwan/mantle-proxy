"""Tests for read-only streaming usage sniffing and cache metric emission.

The central guarantee under test: sniffing must not alter the bytes forwarded
to the client, and must never raise.
"""

import json
import logging

import pytest

from mantle_proxy.server import (
    CACHE_METRICS_MODES,
    Config,
    ConfigError,
    _project_from_headers,
    _reorder_sse_block,
    _sniff_usage,
    emit_cache_metrics,
    extract_usage_from_sse_block,
    normalize_cache_usage,
)

# --------------------------------------------------------------------------
# Realistic SSE fixtures
# --------------------------------------------------------------------------

RESPONSES_STREAM = [
    b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\n',
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"He"}\n\n',
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"llo"}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed","response":'
    b'{"id":"resp_1","model":"openai.gpt-5.6-sol","usage":{"input_tokens":3671,'
    b'"output_tokens":12,"total_tokens":3683,'
    b'"input_tokens_details":{"cached_tokens":3626,"cache_write_tokens":0}}}}\n\n',
]

CHAT_COMPLETIONS_STREAM = [
    b'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n',
    b'data: {"object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":1000,'
    b'"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":800,'
    b'"cache_write_tokens":0}}}\n\n',
    b'data: [DONE]\n\n',
]

ANTHROPIC_STREAM = [
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
    b'"usage":{"input_tokens":45,"cache_read_input_tokens":3626,'
    b'"cache_creation_input_tokens":0,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta",'
    b'"delta":{"text":"hi"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":12}}\n\n',
]


def _run_loop_with_sniff(chunks):
    """Mirror of the sniff + write loop inside _handle_stream."""
    written = []
    usage_seen = None
    buf = b""
    for chunk in chunks:
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            usage_seen = _sniff_usage(block, usage_seen)
            written.append(_reorder_sse_block(block) + b"\n\n")
    if buf:
        usage_seen = _sniff_usage(buf, usage_seen)
        written.append(_reorder_sse_block(buf) + b"\n\n")
    return b"".join(written), usage_seen


def _run_loop_without_sniff(chunks):
    """Pre-patch behaviour, for byte-identity comparison."""
    written = []
    buf = b""
    for chunk in chunks:
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            written.append(_reorder_sse_block(block) + b"\n\n")
    if buf:
        written.append(_reorder_sse_block(buf) + b"\n\n")
    return b"".join(written)


# --------------------------------------------------------------------------
# The core guarantee: forwarded bytes are untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [
    RESPONSES_STREAM, CHAT_COMPLETIONS_STREAM, ANTHROPIC_STREAM,
])
def test_sniffing_does_not_change_forwarded_bytes(stream):
    with_sniff, _ = _run_loop_with_sniff(stream)
    assert with_sniff == _run_loop_without_sniff(stream)


def test_sniffing_survives_split_across_chunk_boundaries():
    """SSE blocks arriving split mid-JSON must still be forwarded verbatim."""
    joined = b"".join(RESPONSES_STREAM)
    fragments = [joined[i:i + 7] for i in range(0, len(joined), 7)]

    with_sniff, usage = _run_loop_with_sniff(fragments)

    assert with_sniff == _run_loop_without_sniff(fragments)
    assert usage["input_tokens_details"]["cached_tokens"] == 3626


def test_sniffing_does_not_break_on_garbage_stream():
    garbage = [
        b'data: {not json at all\n\n',
        b': keepalive comment\n\n',
        b'event: ping\n\n',
        b'data: null\n\n',
        b'data: [1,2,3]\n\n',
        b'data: \xff\xfe invalid utf8\n\n',
    ]
    with_sniff, usage = _run_loop_with_sniff(garbage)

    assert with_sniff == _run_loop_without_sniff(garbage)
    assert usage is None


# --------------------------------------------------------------------------
# extract_usage_from_sse_block
# --------------------------------------------------------------------------


def test_extract_usage_from_responses_completed():
    _, usage = _run_loop_with_sniff(RESPONSES_STREAM)
    assert usage["input_tokens"] == 3671
    assert usage["input_tokens_details"]["cached_tokens"] == 3626


def test_extract_usage_from_chat_completions_final_chunk():
    _, usage = _run_loop_with_sniff(CHAT_COMPLETIONS_STREAM)
    assert usage["prompt_tokens"] == 1000
    assert usage["prompt_tokens_details"]["cached_tokens"] == 800


def test_extract_usage_from_anthropic_message_start():
    _, usage = _run_loop_with_sniff(ANTHROPIC_STREAM)
    assert usage["cache_read_input_tokens"] == 3626


def test_extract_usage_ignores_output_only_usage():
    """Anthropic message_delta carries output_tokens only; useless for cache rate."""
    block = b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":12}}'
    assert extract_usage_from_sse_block(block) is None


def test_extract_usage_handles_multiline_data_field():
    block = (b'event: response.completed\n'
             b'data: {"response":{"usage":\n'
             b'data: {"input_tokens":10,"output_tokens":2}}}')
    assert extract_usage_from_sse_block(block) == {"input_tokens": 10, "output_tokens": 2}


@pytest.mark.parametrize("block", [
    b'data: [DONE]',
    b'data:',
    b': just a comment',
    b'event: ping',
    b'',
    b'data: not-json',
    b'data: "a string"',
    b'data: 42',
])
def test_extract_usage_returns_none_without_raising(block):
    assert extract_usage_from_sse_block(block) is None


# --------------------------------------------------------------------------
# normalize_cache_usage — the denominator is the easy thing to get wrong
# --------------------------------------------------------------------------


def test_normalize_inclusive_semantics_responses():
    """Responses API: input_tokens ALREADY includes cached + write."""
    norm = normalize_cache_usage({
        "input_tokens": 3671, "output_tokens": 12,
        "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0},
    })
    assert norm["total_input_tokens"] == 3671
    assert norm["fresh_input_tokens"] == 45
    assert norm["cache_hit"] == 1


def test_normalize_inclusive_semantics_chat_completions():
    norm = normalize_cache_usage({
        "prompt_tokens": 1000, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 800},
    })
    assert norm["total_input_tokens"] == 1000
    assert norm["fresh_input_tokens"] == 200
    assert norm["output_tokens"] == 20


def test_normalize_exclusive_semantics_anthropic():
    """Anthropic: input_tokens EXCLUDES cache, so the denominator must add them."""
    norm = normalize_cache_usage({
        "input_tokens": 45, "output_tokens": 12,
        "cache_read_input_tokens": 3626, "cache_creation_input_tokens": 100,
    })
    assert norm["total_input_tokens"] == 45 + 3626 + 100
    assert norm["fresh_input_tokens"] == 45
    assert norm["cached_tokens"] == 3626
    assert norm["cache_write_tokens"] == 100


def test_normalize_cold_request_is_a_miss_not_a_hit():
    norm = normalize_cache_usage({
        "input_tokens": 3682, "output_tokens": 9,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3626},
    })
    assert norm["cache_hit"] == 0
    assert norm["cache_write_tokens"] == 3626
    assert norm["fresh_input_tokens"] == 56


def test_normalize_without_any_caching():
    norm = normalize_cache_usage({"input_tokens": 100, "output_tokens": 5})
    assert norm == {
        "total_input_tokens": 100,
        "output_tokens": 5,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "fresh_input_tokens": 100,
        "cache_hit": 0,
    }


@pytest.mark.parametrize("usage", [None, "garbage", 42, {}, {"output_tokens": 5}])
def test_normalize_returns_none_for_unusable_usage(usage):
    assert normalize_cache_usage(usage) is None


def test_normalize_clamps_impossible_split_instead_of_going_negative(caplog):
    """Guards against a future upstream semantics change producing a negative counter."""
    with caplog.at_level(logging.WARNING):
        norm = normalize_cache_usage({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 0},
        })
    assert norm["fresh_input_tokens"] == 0
    assert "cache_usage_semantics_mismatch" in caplog.text


def test_normalize_tolerates_non_dict_details():
    norm = normalize_cache_usage({"input_tokens": 10, "input_tokens_details": "garbage"})
    assert norm["total_input_tokens"] == 10
    assert norm["cached_tokens"] == 0


# --------------------------------------------------------------------------
# emit_cache_metrics
# --------------------------------------------------------------------------


def _norm():
    return normalize_cache_usage({
        "input_tokens": 3671, "output_tokens": 12,
        "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0},
    })


def test_emit_off_produces_nothing(capsys, caplog):
    with caplog.at_level(logging.DEBUG):
        emit_cache_metrics("openai.gpt-5.6-sol", _norm(), "off")
    assert capsys.readouterr().out == ""
    assert caplog.text == ""


def test_emit_emf_writes_valid_embedded_metric_format(capsys):
    emit_cache_metrics("openai.gpt-5.6-sol", _norm(), "emf", project="proj_123")

    record = json.loads(capsys.readouterr().out.strip())
    meta = record["_aws"]["CloudWatchMetrics"][0]
    assert meta["Namespace"] == "Custom/MantleProxy"
    assert ["Model", "Project"] in meta["Dimensions"]
    assert {m["Name"] for m in meta["Metrics"]} == {
        "CacheReadTokens", "CacheWriteTokens", "FreshInputTokens",
        "CacheHitRequests", "Requests",
    }
    assert record["Model"] == "openai.gpt-5.6-sol"
    assert record["Project"] == "proj_123"
    # Counters, not a ratio: averaging per-request ratios weights requests wrongly.
    assert record["CacheReadTokens"] == 3626
    assert record["FreshInputTokens"] == 45
    assert record["CacheHitRequests"] == 1
    assert record["Requests"] == 1
    assert "cache_rate" not in record


def test_emit_log_mode_writes_structured_line_without_bodies(caplog):
    with caplog.at_level(logging.INFO):
        emit_cache_metrics("openai.gpt-5.6-sol", _norm(), "log")
    assert "cache_usage" in caplog.text
    assert "cached=3626" in caplog.text
    assert "fresh=45" in caplog.text


def test_emit_ignores_missing_norm(capsys):
    emit_cache_metrics("m", None, "emf")
    assert capsys.readouterr().out == ""


def test_emit_never_raises_on_malformed_norm(capsys):
    emit_cache_metrics("m", {"unexpected": "shape"}, "emf")
    emit_cache_metrics("m", {"unexpected": "shape"}, "log")
    assert capsys.readouterr().out == ""


def test_emf_counters_aggregate_to_expected_cache_rate(capsys):
    """Three requests from the AWS GPT-5.6 caching blog, summed the CloudWatch way."""
    upstreams = [
        {"input_tokens": 3682, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3626}},
        {"input_tokens": 3671, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0}},
        {"input_tokens": 3662, "output_tokens": 9,
         "input_tokens_details": {"cached_tokens": 3626, "cache_write_tokens": 0}},
    ]
    for u in upstreams:
        emit_cache_metrics("openai.gpt-5.6-sol", normalize_cache_usage(u), "emf")

    records = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    read = sum(r["CacheReadTokens"] for r in records)
    write = sum(r["CacheWriteTokens"] for r in records)
    fresh = sum(r["FreshInputTokens"] for r in records)
    hits = sum(r["CacheHitRequests"] for r in records)
    requests = sum(r["Requests"] for r in records)

    assert read + write + fresh == 11015          # matches summed input_tokens
    assert read / (read + write + fresh) == pytest.approx(0.6584, abs=1e-4)
    assert hits / requests == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Project dimension + config
# --------------------------------------------------------------------------


@pytest.mark.parametrize("headers,expected", [
    ({"OpenAI-Project": "proj_1"}, "proj_1"),
    ({"openai-project": "proj_2"}, "proj_2"),
    ({"anthropic-workspace": "ws_1"}, "ws_1"),
    ({"OpenAI-Project": ""}, "default"),
    ({"X-Other": "v"}, "default"),
    ({}, "default"),
    (None, "default"),
])
def test_project_from_headers(headers, expected):
    assert _project_from_headers(headers) == expected


def _clean_env(monkeypatch):
    for name in ("MANTLE_PROXY_HOST", "MANTLE_PROXY_API_KEY",
                 "MANTLE_ALLOW_INSECURE_REMOTE", "MANTLE_CACHE_METRICS"):
        monkeypatch.delenv(name, raising=False)


def test_cache_metrics_mode_defaults_to_log(monkeypatch):
    _clean_env(monkeypatch)
    assert Config.from_env().cache_metrics_mode == "log"


@pytest.mark.parametrize("mode", sorted(CACHE_METRICS_MODES))
def test_cache_metrics_mode_accepts_valid_values(monkeypatch, mode):
    _clean_env(monkeypatch)
    monkeypatch.setenv("MANTLE_CACHE_METRICS", mode.upper())
    assert Config.from_env().cache_metrics_mode == mode


def test_cache_metrics_mode_rejects_invalid_value(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("MANTLE_CACHE_METRICS", "prometheus")
    with pytest.raises(ConfigError):
        Config.from_env()
