# Mantle SigV4 Proxy

Local SigV4 signing proxy for Bedrock Mantle's OpenAI-compatible endpoint.

The service accepts OpenAI-style requests on a local port, signs outgoing requests with AWS SigV4, and forwards them to:

```text
https://bedrock-mantle.<region>.api.aws/openai/v1
```

It also includes a LiteLLM callback that records token usage in `acp-bridge` without sending prompts, completions, API keys, or AWS credentials.

## Features

- SigV4 signing through the standard AWS credential provider chain.
- `/chat/completions` and `/v1/chat/completions` translation to `/responses`.
- Passthrough for other Mantle OpenAI-compatible paths, including `/v1/responses`.
- Optional local proxy API key for non-local deployments.
- Safe defaults: listens on `127.0.0.1`, does not forward client `Authorization` headers, and does not log request or response bodies.
- Explicitly rejects `stream: true` until streaming translation is implemented.

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

Install LiteLLM callback dependencies only if you use the callback:

```bash
pip install -r requirements-litellm.txt
```

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
| `MANTLE_AWS_REGION` | `us-east-2` | AWS region used for endpoint and SigV4 scope. |
| `MANTLE_AWS_SERVICE` | `bedrock` | SigV4 service name. |
| `MANTLE_BASE_URL` | region-derived Mantle URL | Override for custom endpoints. |
| `MANTLE_PROXY_HOST` | `127.0.0.1` | Keep localhost unless you have external auth and network controls. |
| `MANTLE_PROXY_PORT` | `4010` | Listening port. |
| `MANTLE_DEFAULT_MODEL` | `openai.gpt-5.5` | Used when the client omits `model`. |
| `MANTLE_PROXY_API_KEY` | empty | Optional bearer token required by this proxy. |
| `MANTLE_LOG_LEVEL` | `INFO` | Logs method, path, status, and timing only. |

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

## LiteLLM Usage Callback

`litellm_callback.py` provides a LiteLLM `CustomLogger` instance named `proxy_handler_instance`.

It posts usage counters to:

```text
http://127.0.0.1:18010/internal/llm-callback
```

Override with:

```bash
export ACP_BRIDGE_CALLBACK_URL=http://127.0.0.1:18010/internal/llm-callback
export ACP_BRIDGE_CALLBACK_API_KEY=
```

The callback sends only:

- model name
- prompt tokens
- completion tokens
- total tokens
- cached input tokens
- cache creation tokens
- response time

It does not send prompt text, completion text, request headers, API keys, or AWS credentials.

## Compatibility Notes

Streaming is not implemented. Requests with `stream: true` return HTTP `501` instead of silently downgrading to non-streaming behavior.

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
python3 -m compileall mantle_proxy mantle-sigv4-proxy.py litellm_callback.py
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
