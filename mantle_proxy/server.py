"""Production-oriented SigV4 proxy for Bedrock Mantle OpenAI endpoints."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError
from botocore.session import Session

SUPPORTED_TOOL_TYPES = {"function", "mcp", "custom", "namespace", "tool_search"}
DEFAULT_REGION = "us-east-1"
DEFAULT_PORT = 4010
DEFAULT_MODEL = "openai.gpt-5.5"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
FORWARD_HEADERS = {"openai-project", "openai-organization", "anthropic-workspace", "anthropic-version"}
REGION_OVERRIDE_HEADER = "x-mantle-region"

log = logging.getLogger("mantle_proxy")


@dataclass(frozen=True)
class Config:
    region: str
    aws_service: str
    base_url: str
    host: str
    port: int
    default_model: str
    proxy_api_key: str
    allow_insecure_remote: bool
    request_timeout_s: float
    max_body_bytes: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        region = os.getenv("MANTLE_AWS_REGION", DEFAULT_REGION)
        base_url = os.getenv(
            "MANTLE_BASE_URL",
            f"https://bedrock-mantle.{region}.api.aws/openai/v1",
        ).rstrip("/")
        host = os.getenv("MANTLE_PROXY_HOST", "127.0.0.1")
        allow_insecure_remote = _env_bool("MANTLE_ALLOW_INSECURE_REMOTE", False)
        proxy_api_key = os.getenv("MANTLE_PROXY_API_KEY", "")

        if host not in LOCAL_HOSTS and not proxy_api_key and not allow_insecure_remote:
            raise ConfigError(
                "Refusing to bind to a non-localhost interface without "
                "MANTLE_PROXY_API_KEY. Set MANTLE_PROXY_API_KEY or explicitly "
                "set MANTLE_ALLOW_INSECURE_REMOTE=true."
            )

        return cls(
            region=region,
            aws_service=os.getenv("MANTLE_AWS_SERVICE", "bedrock"),
            base_url=base_url,
            host=host,
            port=_env_int("MANTLE_PROXY_PORT", DEFAULT_PORT),
            default_model=os.getenv("MANTLE_DEFAULT_MODEL", DEFAULT_MODEL),
            proxy_api_key=proxy_api_key,
            allow_insecure_remote=allow_insecure_remote,
            request_timeout_s=_env_float("MANTLE_REQUEST_TIMEOUT_SECONDS", 120.0),
            max_body_bytes=_env_int("MANTLE_MAX_BODY_BYTES", 20 * 1024 * 1024),
            log_level=os.getenv("MANTLE_LOG_LEVEL", "INFO").upper(),
        )


class ConfigError(RuntimeError):
    """Invalid startup configuration."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


class SigV4Client:
    def __init__(self, config: Config, http: ClientSession):
        self.config = config
        self.http = http
        self.aws_session = Session()

    async def request(self, method: str, path: str, body: bytes | None, query: str = "",
                      extra_headers: dict[str, str] | None = None,
                      region_override: str | None = None):
        url = self._target_url(path, query, region_override)
        region = region_override or self.config.region
        aws_headers = {"Accept": "application/json"}
        if body is not None:
            aws_headers["Content-Type"] = "application/json"
        if extra_headers:
            aws_headers.update(extra_headers)

        credentials = self.aws_session.get_credentials()
        if credentials is None:
            raise UpstreamConfigError("AWS credentials were not found")

        aws_req = AWSRequest(
            method=method.upper(),
            url=url,
            data=body or b"",
            headers=aws_headers,
        )
        SigV4Auth(
            credentials.get_frozen_credentials(),
            self.config.aws_service,
            region,
        ).add_auth(aws_req)

        started = time.monotonic()
        async with self.http.request(
            method.upper(),
            url,
            data=body,
            headers=dict(aws_req.headers),
        ) as resp:
            raw = await resp.read()
            elapsed_ms = (time.monotonic() - started) * 1000
            log.info("mantle_request method=%s path=%s status=%s duration_ms=%.1f",
                     method.upper(), _safe_path(path), resp.status, elapsed_ms)
            return resp.status, raw, resp.headers.get("Content-Type", "application/json")

    async def request_stream(self, method: str, path: str, body: bytes | None, query: str = "",
                             extra_headers: dict[str, str] | None = None,
                             region_override: str | None = None):
        """Streaming variant: yields chunks from upstream SSE response."""
        url = self._target_url(path, query, region_override)
        region = region_override or self.config.region
        aws_headers = {"Accept": "text/event-stream"}
        if body is not None:
            aws_headers["Content-Type"] = "application/json"
        if extra_headers:
            aws_headers.update(extra_headers)

        credentials = self.aws_session.get_credentials()
        if credentials is None:
            raise UpstreamConfigError("AWS credentials were not found")

        aws_req = AWSRequest(
            method=method.upper(),
            url=url,
            data=body or b"",
            headers=aws_headers,
        )
        SigV4Auth(
            credentials.get_frozen_credentials(),
            self.config.aws_service,
            region,
        ).add_auth(aws_req)

        async with self.http.request(
            method.upper(),
            url,
            data=body,
            headers=dict(aws_req.headers),
        ) as resp:
            log.info("mantle_stream method=%s path=%s status=%s",
                     method.upper(), _safe_path(path), resp.status)
            yield resp.status, resp.headers.get("Content-Type", "text/event-stream")
            async for chunk in resp.content.iter_any():
                yield chunk

    def _target_url(self, path: str, query: str = "", region_override: str | None = None) -> str:
        region = region_override or self.config.region
        normalized = normalize_path(path)
        # Anthropic Messages API uses a different base path
        if normalized.startswith("anthropic/"):
            base = f"https://bedrock-mantle.{region}.api.aws"
        elif region_override:
            base = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
        else:
            base = self.config.base_url
        url = f"{base}/{normalized}" if normalized else base
        if query:
            url = f"{url}?{query}"
        return url


class UpstreamConfigError(RuntimeError):
    """Configuration problem before contacting Mantle."""


def normalize_path(path: str) -> str:
    stripped = path.strip("/")
    if stripped == "v1":
        return ""
    if stripped.startswith("v1/"):
        return stripped[3:]
    return stripped


def _safe_path(path: str) -> str:
    return "/" + normalize_path(path)


@web.middleware
async def error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except UpstreamConfigError as exc:
        log.warning("upstream_config_error: %s", exc)
        return json_error(503, "upstream_configuration_error", str(exc))
    except (BotoCoreError, ClientError, TimeoutError) as exc:
        log.warning("upstream_request_failed: %s", exc.__class__.__name__)
        return json_error(502, "upstream_request_failed", "Mantle request failed")
    except json.JSONDecodeError:
        return json_error(400, "invalid_json", "Request body must be valid JSON")
    except Exception:
        log.exception("unhandled_request_error")
        return json_error(500, "internal_error", "Internal proxy error")


@web.middleware
async def auth_middleware(request: web.Request, handler):
    config: Config = request.app["config"]
    if config.proxy_api_key:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, config.proxy_api_key):
            return json_error(401, "unauthorized", "Missing or invalid proxy API key")
    return await handler(request)


def json_error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": code, "code": code}},
        status=status,
    )


def _extract_forward_headers(request: web.Request) -> tuple[dict[str, str] | None, str | None]:
    """Extract client headers to forward and optional region override."""
    fwd = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
    region = request.headers.get(REGION_OVERRIDE_HEADER)
    return fwd or None, region


async def health(request: web.Request) -> web.Response:
    config: Config = request.app["config"]
    return web.json_response({
        "status": "ok",
        "region": config.region,
        "base_url": config.base_url,
        "auth_required": bool(config.proxy_api_key),
    })


async def handle_chat_completions(request: web.Request) -> web.Response:
    data = await read_json(request)

    config: Config = request.app["config"]
    client: SigV4Client = request.app["sigv4"]
    fwd, region = _extract_forward_headers(request)
    model = data.get("model") or config.default_model

    # OpenAI models only support Responses API; others support Chat Completions natively
    if model.startswith("openai."):
        payload = chat_to_responses(data, config.default_model)
        if data.get("stream"):
            payload["stream"] = True
            body = encode_json(payload)
            return await _handle_stream(request, client, "responses", body,
                                        extra_headers=fwd, region_override=region)
        body = encode_json(payload)
        status, raw, content_type = await client.request("POST", "responses", body,
                                                         extra_headers=fwd, region_override=region)
        if status < 200 or status >= 300:
            return web.Response(body=raw, status=status, content_type=_content_type(content_type))
        upstream = json.loads(raw)
        return web.json_response(responses_to_chat(upstream, config.default_model))

    # All other models: pass through to Chat Completions endpoint
    data = filter_tools(data)
    body = encode_json(data)
    if data.get("stream"):
        return await _handle_stream(request, client, "chat/completions", body,
                                    extra_headers=fwd, region_override=region)
    status, raw, content_type = await client.request("POST", "chat/completions", body,
                                                     extra_headers=fwd, region_override=region)
    return web.Response(body=raw, status=status, content_type=_content_type(content_type))


async def handle_passthrough(request: web.Request) -> web.Response:
    client: SigV4Client = request.app["sigv4"]
    body = await request.read()
    outbound_body: bytes | None = body or None
    is_stream = False

    if body:
        data = maybe_parse_json_body(body, request.content_type)
        if isinstance(data, dict):
            is_stream = bool(data.get("stream"))
            data = filter_tools(data)
            outbound_body = encode_json(data)

    path = request.match_info.get("path", "")
    fwd, region = _extract_forward_headers(request)

    if is_stream:
        return await _handle_stream(request, client, path, outbound_body,
                                    extra_headers=fwd, region_override=region)

    status, raw, content_type = await client.request(
        request.method,
        path,
        outbound_body,
        request.query_string,
        extra_headers=fwd,
        region_override=region,
    )
    return web.Response(body=raw, status=status, content_type=_content_type(content_type))


async def _handle_stream(request: web.Request, client: SigV4Client,
                         path: str, body: bytes | None,
                         extra_headers: dict[str, str] | None = None,
                         region_override: str | None = None) -> web.StreamResponse:
    stream_gen = client.request_stream(request.method, path, body, request.query_string,
                                       extra_headers=extra_headers,
                                       region_override=region_override)
    header = await stream_gen.__anext__()
    status, content_type = header

    resp = web.StreamResponse(status=status, headers={"Content-Type": _content_type(content_type)})
    resp.enable_chunked_encoding()
    await resp.prepare(request)

    buf = b""
    async for chunk in stream_gen:
        buf += chunk
        # Process complete SSE blocks (separated by \n\n)
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            await resp.write(_reorder_sse_block(block) + b"\n\n")
    if buf:
        await resp.write(_reorder_sse_block(buf) + b"\n\n")

    await resp.write_eof()
    return resp


def _reorder_sse_block(block: bytes) -> bytes:
    """Reorder SSE fields so event: comes before data: (OpenAI standard order)."""
    lines = block.split(b"\n")
    event_lines = []
    data_lines = []
    other_lines = []
    for line in lines:
        if line.startswith(b"event:"):
            event_lines.append(line)
        elif line.startswith(b"data:"):
            data_lines.append(line)
        else:
            other_lines.append(line)
    return b"\n".join(event_lines + data_lines + other_lines)


def maybe_parse_json_body(body: bytes, content_type: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        if "json" in content_type.lower():
            raise
        return None


async def read_json(request: web.Request) -> dict[str, Any]:
    body = await request.read()
    if not body:
        return {}
    data = json.loads(body)
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="JSON request body must be an object")
    return data


def encode_json(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_type(raw: str) -> str:
    if not raw:
        return "application/octet-stream"
    return raw.split(";", 1)[0]


def chat_to_responses(data: dict[str, Any], default_model: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": data.get("model") or default_model,
        "input": chat_messages_to_response_input(data.get("messages") or []),
    }

    field_map = {
        "max_tokens": "max_output_tokens",
        "max_completion_tokens": "max_output_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "metadata": "metadata",
        "store": "store",
        "parallel_tool_calls": "parallel_tool_calls",
        "user": "user",
    }
    for source, target in field_map.items():
        if source in data and data[source] is not None:
            out[target] = data[source]

    if "tool_choice" in data and data["tool_choice"] is not None:
        out["tool_choice"] = convert_tool_choice(data["tool_choice"])

    if "response_format" in data and isinstance(data["response_format"], dict):
        text_format = convert_response_format(data["response_format"])
        if text_format:
            out["text"] = {"format": text_format}

    if "reasoning" in data and isinstance(data["reasoning"], dict):
        out["reasoning"] = data["reasoning"]
    elif "reasoning_effort" in data:
        out["reasoning"] = {"effort": data["reasoning_effort"]}

    if "tools" in data:
        tools = convert_tools(data["tools"])
        if tools:
            out["tools"] = tools

    return out


def chat_messages_to_response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_input: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "tool":
            response_input.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": content_to_text(content),
            })
            continue

        if role == "assistant" and message.get("tool_calls"):
            if content:
                response_input.append({"role": role, "content": convert_content(role, content)})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                response_input.append({
                    "type": "function_call",
                    "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": function.get("name", ""),
                    "arguments": normalize_arguments(function.get("arguments", "")),
                })
            continue

        response_input.append({"role": role, "content": convert_content(role, content)})
    return response_input


def convert_content(role: str, content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    converted = []
    for part in content:
        if not isinstance(part, dict):
            converted.append(part)
            continue
        part_type = part.get("type")
        if part_type == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url") or {}
            converted.append({"type": "input_image", "image_url": image_url.get("url", "")})
        else:
            converted.append(part)
    return converted


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    converted = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") not in SUPPORTED_TOOL_TYPES:
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            response_tool = {
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
            if "strict" in function:
                response_tool["strict"] = function["strict"]
            converted.append(response_tool)
        else:
            converted.append(tool)
    return converted


def convert_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
        return {"type": "function", "name": tool_choice["function"].get("name", "")}
    return tool_choice


def convert_response_format(response_format: dict[str, Any]) -> dict[str, Any] | None:
    format_type = response_format.get("type")
    if format_type in {"text", "json_object"}:
        return {"type": format_type}
    if format_type != "json_schema":
        return None

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return response_format

    converted: dict[str, Any] = {"type": "json_schema"}
    for key in ("name", "description", "schema", "strict"):
        if key in json_schema:
            converted[key] = json_schema[key]
    return converted


def filter_tools(data: dict[str, Any]) -> dict[str, Any]:
    data = normalize_responses_input(data)
    if "tools" not in data:
        return data
    copied = dict(data)
    tools = convert_tools(copied["tools"])
    if tools:
        copied["tools"] = tools
    else:
        copied.pop("tools", None)
    return copied


def normalize_responses_input(data: dict[str, Any]) -> dict[str, Any]:
    """Hoist Codex 'additional_tools' input items into the top-level tools array.

    Codex >= 0.144 injects {"type": "additional_tools", "role": "developer",
    "tools": [...]} into `input` (collaboration mode). Upstream OpenAI accepts
    this, but Bedrock Mantle's Responses API rejects it with
    "Invalid 'input': value did not match any expected variant".
    Mantle does accept the same tools in the top-level `tools` array.
    """
    items = data.get("input")
    if not isinstance(items, list):
        return data
    extra_tools: list[Any] = []
    new_input: list[Any] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            tools = item.get("tools")
            if isinstance(tools, list):
                extra_tools.extend(tools)
        else:
            new_input.append(item)
    if not extra_tools:
        return data
    copied = dict(data)
    copied["input"] = new_input
    existing = copied.get("tools")
    copied["tools"] = (existing if isinstance(existing, list) else []) + extra_tools
    return copied


def responses_to_chat(data: dict[str, Any], default_model: str) -> dict[str, Any]:
    content = ""
    reasoning = ""
    tool_calls: list[dict[str, Any]] = []

    for item in data.get("output", []) or []:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []) or []:
                if part.get("type") in {"output_text", "text"}:
                    content += part.get("text", "")
        elif item_type == "reasoning":
            reasoning += extract_reasoning(item)
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": normalize_arguments(item.get("arguments", "")),
                },
            })

    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    finish_reason = infer_finish_reason(data, bool(tool_calls))

    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    return {
        "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model") or default_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def extract_reasoning(item: dict[str, Any]) -> str:
    chunks = []
    for key in ("summary", "content"):
        for part in item.get(key, []) or []:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(part))
    return "".join(chunks)


def normalize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)


def infer_finish_reason(data: dict[str, Any], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason")
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
    return "stop"


async def on_startup(app: web.Application) -> None:
    config: Config = app["config"]
    timeout = ClientTimeout(total=config.request_timeout_s)
    app["http"] = ClientSession(timeout=timeout)
    app["sigv4"] = SigV4Client(config, app["http"])


async def on_cleanup(app: web.Application) -> None:
    http: ClientSession | None = app.get("http")
    if http is not None:
        await http.close()


def create_app(config: Config | None = None) -> web.Application:
    config = config or Config.from_env()
    app = web.Application(
        middlewares=[error_middleware, auth_middleware],
        client_max_size=config.max_body_bytes,
    )
    app["config"] = config
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", health)
    app.router.add_post("/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_route("*", "/{path:.*}", handle_passthrough)
    return app


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> int:
    try:
        config = Config.from_env()
        setup_logging(config.log_level)
        log.info("starting host=%s port=%s region=%s base_url=%s auth_required=%s",
                 config.host, config.port, config.region, config.base_url, bool(config.proxy_api_key))
        web.run_app(create_app(config), host=config.host, port=config.port)
        return 0
    except ConfigError as exc:
        setup_logging("ERROR")
        log.error("configuration_error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
