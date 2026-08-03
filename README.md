# Mantle SigV4 Proxy

Local SigV4 signing proxy for Bedrock Mantle's OpenAI-compatible endpoint.

The service accepts OpenAI-style requests on a local port, signs outgoing requests with AWS SigV4, and forwards them to:

```text
https://bedrock-mantle.<region>.api.aws/openai/v1
```

## Features

- SigV4 signing through the standard AWS credential provider chain.
- `/chat/completions` → OpenAI models route to `/responses`, others pass through to Mantle Chat Completions.
- `/anthropic/` path routes to Bedrock Mantle Anthropic Messages API (Claude + workspace cost tracking).
- Streaming (`stream: true`) is supported on all routes as an SSE pass-through. See [Streaming](#streaming) for the protocol caveat.
- Prompt cache observability: read-only usage sniffing on both streaming and non-streaming paths, optionally emitted as CloudWatch metrics. See [Prompt Cache Observability](#prompt-cache-observability).
- Per-request region override via `X-Mantle-Region` header, validated against canonical AWS region codes.
- Forwards `OpenAI-Project`, `anthropic-workspace`, `anthropic-version` headers for Project/Workspace cost tracking.
- Passthrough for other Mantle OpenAI-compatible paths, including `/v1/responses`.
- Optional local proxy API key for non-local deployments.
- Safe defaults: listens on `127.0.0.1`, does not forward client `Authorization` headers, and does not log request or response bodies.

## Requirements

- Python 3.9+
- AWS credentials that can call Bedrock Mantle.
- Network access to the Mantle endpoint.

Install the core proxy dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package install with the `mantle-sigv4-proxy` console command:

```bash
pip install -e .
```

Install LiteLLM callback dependencies only if you use an optional integration:

```bash
pip install -r requirements-litellm.txt
```

## LiteLLM Integration

This repo also hosts the LiteLLM proxy configuration used by the [acp-bridge](https://github.com/xiwan/acp-bridge) stack:

| File | Purpose |
|------|---------|
| `litellm-config.yaml` | Model registry — maps model names to Bedrock/Mantle backends. No secrets (master key read from `LITELLM_API_KEY` env var). |
| `litellm_callback.py` | Custom callback that posts token usage to acp-bridge and patches Fable 5 thinking params. |

The systemd `litellm.service` points `--config` and `WorkingDirectory` to this repo so LiteLLM can load both files at startup.

### Environment variables read by `litellm-config.yaml`

These are consumed by LiteLLM, not by the proxy:

| Variable | Notes |
|----------|-------|
| `LITELLM_API_KEY` | LiteLLM master key. |
| `MANTLE_PROJECT_ID` | Injected as the `OpenAI-Project` / `anthropic-workspace` header on Mantle-backed models for Project and Workspace cost tracking. **Must be a full project ARN**, not a bare name — Mantle rejects anything else with `validation_error: '<value>' is not a valid project ARN`. Format: `arn:aws:bedrock-mantle:<region>:<account-id>:project/<project-name>` (the default project is named `default`). |

### Adding a model

Append to `litellm-config.yaml` under `model_list`:

```yaml
  - model_name: "bedrock/<model-id>"
    litellm_params:
      model: "bedrock/<model-id>"
      aws_region_name: "us-east-1"
```

Then `sudo systemctl restart litellm`.

## Configuration

The proxy is configured by environment variables. Start from the example file:

```bash
cp .env.example .env
```

Load it in your shell if you use `.env` manually:

```bash
set -a
source .env
set +a
```

Important variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `MANTLE_AWS_REGION` | `us-east-1` | AWS region used for endpoint and SigV4 scope. `.env.example` ships `us-east-2`; the built-in code default is `us-east-1`. |
| `MANTLE_AWS_SERVICE` | `bedrock` | SigV4 service name. |
| `MANTLE_BASE_URL` | region-derived Mantle URL | Override for custom endpoints. |
| `MANTLE_PROXY_HOST` | `127.0.0.1` | Keep localhost unless you have external auth and network controls. |
| `MANTLE_PROXY_PORT` | `4010` | Listening port. |
| `MANTLE_DEFAULT_MODEL` | `openai.gpt-5.5` | Used when the client omits `model`. |
| `MANTLE_PROXY_API_KEY` | empty | Optional bearer token required by this proxy. |
| `MANTLE_REQUEST_TIMEOUT_SECONDS` | `120` | Total upstream request timeout. |
| `MANTLE_MAX_BODY_BYTES` | `20971520` | Max accepted request body size. |
| `MANTLE_CACHE_METRICS` | `log` | Prompt cache observability mode: `off`, `log`, or `emf`. Invalid values fail at startup. |
| `MANTLE_ALLOW_INSECURE_REMOTE` | `false` | Escape hatch for binding a remote interface without an API key. |
| `MANTLE_LOG_LEVEL` | `INFO` | Logs method, path, status, timing, and cache token counts only. |

## Run

```bash
python3 mantle-sigv4-proxy.py
```

Health check:

```bash
curl http://127.0.0.1:4010/health
```

Chat Completions compatibility call:

```bash
curl http://127.0.0.1:4010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai.gpt-5.5",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Direct Responses passthrough:

```bash
curl http://127.0.0.1:4010/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai.gpt-5.5",
    "input": "Say hello in one sentence."
  }'
```

## Prompt Cache Observability

Bedrock Mantle publishes **no cache metric to CloudWatch**. The `AWS/BedrockMantle`
namespace has `Inferences`, `InferenceClientErrors`, `TotalInputTokens`,
`TotalOutputTokens`, `InputTokens`, and `OutputTokens` — nothing cache-specific.
(`bedrock-runtime`'s `AWS/Bedrock` namespace does have `CacheReadInputTokenCount`
and `CacheWriteInputTokenCount`, but this proxy does not use that endpoint.)

The response `usage` object is therefore the only authoritative source. This proxy
sniffs it read-only and emits counters. Sniffing never modifies the bytes forwarded
to the client and never raises.

Set the mode with `MANTLE_CACHE_METRICS`:

| Mode | Behaviour |
| --- | --- |
| `off` | No emission. |
| `log` | Default. One structured log line per request. No behaviour change. |
| `emf` | Embedded Metric Format on stdout, for pickup by the CloudWatch agent. |

`log` output:

```text
cache_usage model=openai.gpt-5.6-sol total_input=3671 cached=3626 cache_write=0 fresh=45 output=12 hit=1
```

`emf` mode emits five **counters** under namespace `Custom/MantleProxy`, with
`Model` and `Project` dimensions (`Project` is taken from the forwarded
`OpenAI-Project` or `anthropic-workspace` header, else `default`):

```text
CacheReadTokens  CacheWriteTokens  FreshInputTokens  CacheHitRequests  Requests
```

Counters, not a ratio: averaging per-request ratios in CloudWatch weights a long
cached prompt the same as a short uncached one. Divide the Sums instead.

```json
{
  "metrics": [
    [ "Custom/MantleProxy", "CacheReadTokens",  "Project", "default", { "id": "r", "visible": false, "stat": "Sum" } ],
    [ ".", "CacheWriteTokens", ".", ".", { "id": "w", "visible": false, "stat": "Sum" } ],
    [ ".", "FreshInputTokens", ".", ".", { "id": "f", "visible": false, "stat": "Sum" } ],
    [ ".", "CacheHitRequests", ".", ".", { "id": "h", "visible": false, "stat": "Sum" } ],
    [ ".", "Requests",         ".", ".", { "id": "q", "visible": false, "stat": "Sum" } ],
    [ { "expression": "100*r/(r+w+f)", "label": "Cached token %" } ],
    [ { "expression": "100*h/q",       "label": "Request hit rate" } ],
    [ { "expression": "100*w/(r+w+f)", "label": "Cache write % (fragmentation)" } ]
  ],
  "stat": "Sum", "period": 300, "view": "timeSeries"
}
```

Read all three together. A high cached percentage with cache write near zero means
caching is working. A persistently high cache write percentage means the cache is
being rebuilt repeatedly — usually an unstable prompt prefix (timestamps,
whitespace drift, reordered JSON keys).

### Token accounting

Two incompatible conventions exist upstream, and mixing them up produces a rate
that is wrong by roughly 2x. The proxy discriminates on which fields the upstream
actually sent:

| Convention | Upstream | Denominator |
| --- | --- | --- |
| inclusive | Responses API, OpenAI Chat Completions (`input_tokens_details` / `prompt_tokens_details`) | `input_tokens` already includes cached and cache-write tokens |
| exclusive | Anthropic Messages (`cache_read_input_tokens` / `cache_creation_input_tokens`) | `input_tokens` excludes them, so add both |

Non-streaming `/chat/completions` responses for OpenAI models additionally carry
the cache detail through to the client as `usage.prompt_tokens_details`, so
downstream consumers such as LiteLLM can read it.

On Anthropic streaming, `message_start` and `message_delta` both carry the full
usage object — `input_tokens`, `cache_read_input_tokens` and
`cache_creation_input_tokens` are repeated on `message_delta` alongside the final
`output_tokens`. The sniffer keeps the last usage it sees, so both cache
accounting and `output_tokens` are accurate on that path.

## Client Authentication

By default the proxy is unauthenticated because it binds to `127.0.0.1`.

If you bind to a non-localhost interface, set `MANTLE_PROXY_API_KEY`. The service refuses to start on a remote interface without this key unless `MANTLE_ALLOW_INSECURE_REMOTE=true` is explicitly set.

```bash
export MANTLE_PROXY_HOST=0.0.0.0
export MANTLE_PROXY_API_KEY="$(openssl rand -hex 32)"
python3 mantle-sigv4-proxy.py
```

Clients then send:

```text
Authorization: Bearer <MANTLE_PROXY_API_KEY>
```

Do not reuse an OpenAI API key, AWS access key, GitHub token, or any other production credential as `MANTLE_PROXY_API_KEY`.

## AWS Credential Safety

AWS credentials are never configured in this repository. `botocore` reads credentials from the standard AWS chain:

- Environment variables such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`.
- `~/.aws/config` and `~/.aws/credentials`.
- AWS SSO.
- Web identity.
- EC2/ECS instance role credentials.

For production, prefer short-lived credentials from SSO, STS, or instance roles. Avoid long-lived IAM user keys. Never commit `.env`, AWS credentials, LiteLLM keys, or proxy API keys.

## Optional Integrations

The proxy does not require `acp-bridge`. The core service runs with only the configuration above.

An optional LiteLLM callback for ACP Bridge lives at:

```text
litellm_callback.py
```

It provides a LiteLLM `CustomLogger` instance named `proxy_handler_instance`, and
is loaded by LiteLLM (not by this proxy). When configured, it posts usage counters to:

```text
http://127.0.0.1:18010/internal/llm-callback
```

Override with:

```bash
export ACP_BRIDGE_CALLBACK_URL=http://127.0.0.1:18010/internal/llm-callback
export ACP_BRIDGE_CALLBACK_TIMEOUT_SECONDS=5      # invalid or non-positive falls back to 5
export ACP_BRIDGE_CALLBACK_API_KEY=               # optional; sent as Authorization: Bearer
```

Delivery failures are logged as warnings on the `mantle_proxy.acp_bridge` logger
(both transport errors and non-2xx responses) rather than silently discarded.

The callback sends only:

- model name
- prompt tokens
- completion tokens
- total tokens
- cached input tokens
- cache creation tokens
- response time

It does not send prompt text, completion text, request headers, API keys, or AWS credentials. Leave this integration unconfigured if you do not use ACP Bridge.

The callback also rewrites `thinking.type` from `enabled` to `adaptive` for
Claude Fable 5 models, which reject `enabled`.

## Compatibility Notes

### Streaming

Streaming is implemented as an SSE pass-through on all routes. The proxy reorders
`event:` before `data:` within each SSE block (OpenAI field order) and sniffs usage
read-only; it does not otherwise alter the byte stream.

⚠️ **Protocol caveat.** For `openai.*` models the two paths speak different
protocols:

| Request | Upstream path | Response protocol |
| --- | --- | --- |
| `/v1/chat/completions`, no `stream` | `/responses` | converted back to `chat.completion` |
| `/v1/chat/completions`, `stream: true` | `/responses` | **raw Responses SSE** (`response.output_text.delta`, ...) — *not* `chat.completion.chunk` |

A client that posts to `/v1/chat/completions` with `stream: true` and expects
`chat.completion.chunk` events will not parse the result. This suits clients that
already speak the Responses protocol (for example Codex); it does not suit a
stock OpenAI SDK or LiteLLM. Non-`openai.*` models stream Chat Completions SSE
unchanged and are unaffected.

### Verified upstream behaviour

Observed against `bedrock-mantle.us-east-1.api.aws` with `openai.gpt-5.6-sol`
(2026-08-03):

- Mantle's Responses API reports cache activity under
  **`usage.input_tokens_details`** (`cached_tokens`, `cache_write_tokens`), not the
  OpenAI Chat Completions `usage.prompt_tokens_details`. `input_tokens` **includes**
  cached and cache-write tokens.
- Early events (`response.created`, `response.in_progress`) carry `usage: null`.
  A stream truncated by `max_output_tokens` terminates with
  `response.incomplete` and **never emits `response.completed`** — the usage
  sniffer therefore keys on where usage appears, not on the event name.
- `GET /v1/models` returns HTTP `404`; Mantle does not expose a model list here.
- `max_output_tokens` has a minimum of `16` for `openai.gpt-5.6-sol`; lower values
  are rejected with `integer_below_min_value`.
- Model route support is **not** derivable from the `openai.` prefix alone.
  `openai.gpt-oss-*` rejects `/openai/v1/responses` with `The model '<id>' does
  not support the '/openai/v1/responses' API`, so it is routed to
  `chat/completions` (see `uses_responses_api()`). On the account tested it was
  rejected there too (`isn't supported on this route`), so gpt-oss appears
  unavailable rather than merely mis-routed — the route correction is unverified
  end to end.
- Converse-only models are not reachable through this proxy at all.
  `moonshotai.kimi-k2.5` and `qwen.qwen3-coder-next` exist in Mantle's registry
  but return `isn't supported on this route` on both `chat/completions` and
  `responses`; `litellm-config.yaml` correctly routes them via `bedrock/`
  (Converse) instead. `amazon.nova-*` and `deepseek.*` return `does not exist`.
- SigV4 with service name `bedrock` (the `MANTLE_AWS_SERVICE` default) is accepted.
- Anthropic Messages (`anthropic.claude-haiku-4-5`) reports cache activity as
  top-level `cache_read_input_tokens` / `cache_creation_input_tokens`, and
  `input_tokens` **excludes** both — the opposite of the Responses API. Observed
  `input_tokens: 17` with `cache_read_input_tokens: 12400`, so the denominator is
  `17 + 12400`. The payload also carries `cache_creation.ephemeral_5m_input_tokens`
  / `ephemeral_1h_input_tokens` (TTL split) and `service_tier`.
- On Anthropic streaming both `message_start` and `message_delta` carry the full
  usage object; `message_delta` repeats `input_tokens` and the cache fields
  alongside the final `output_tokens`.
- Anthropic models may be unavailable from some locations, returning
  `invalid_request_error: Access to Anthropic models is not allowed from
  unsupported countries, regions, or territories`. This is independent of the AWS
  region and of this proxy.

### Region override

`X-Mantle-Region` is validated against canonical AWS region codes
(`^[a-z]{2}(?:-[a-z]+)+-\d+$`). A malformed value is rejected with HTTP `400` and
`type: invalid_region_override`. The header is interpolated into the upstream
hostname, so an unvalidated value would let a client redirect signed requests and
the prompt body to a host it controls.

### Tools

Function tools are converted from Chat Completions shape:

```json
{"type": "function", "function": {"name": "tool_name", "parameters": {}}}
```

to Responses tool shape:

```json
{"type": "function", "name": "tool_name", "parameters": {}}
```

Other supported tool types are passed through when their `type` is one of `function`, `mcp`, `custom`, `namespace`, or `tool_search`.

## Security Checklist Before Publishing

- Confirm `.env`, `*.log`, and local virtualenv files are ignored by git.
- Review history before pushing; do not publish previous commits containing secrets.
- Keep `MANTLE_PROXY_HOST=127.0.0.1` unless a reverse proxy, firewall, and `MANTLE_PROXY_API_KEY` are in place.
- Do not log request bodies, response bodies, `Authorization`, `X-Api-Key`, AWS keys, or `X-Amz-Security-Token`.
- Use least-privilege IAM permissions for Mantle access.
- Rotate any token that was ever pasted into a terminal, shell history, log, issue, or README draft.

## Development Checks

Syntax check:

```bash
python3 -m compileall mantle_proxy mantle-sigv4-proxy.py
```

Unit tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Run the server locally:

```bash
python3 mantle-sigv4-proxy.py
```

In another shell:

```bash
curl http://127.0.0.1:4010/health
```
