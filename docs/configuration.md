# Configuration

Every environment variable translator reads, with its default. Copy
`.env.example` to `.env` and fill in your values — that file is the source of
truth for what compose passes into the containers.

Only `OPENAI_API_BASE` and `TEXT_MODEL` are required; everything else has a
working default. Without the edge gateway, translating also needs
`TRANSLATOR_DEFAULT_IDENTITY`, which the dev overlay sets — see
[Default identity](#default-identity-dev-only).

## Inference endpoint

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible endpoint — must include `/v1` (e.g. `http://vllm-router:4000/v1` or `http://ollama:11434/v1`) |
| `OPENAI_API_KEY` | No | `dummy` | API key; any value works for local servers that do not enforce auth |
| `OPENAI_TIMEOUT` | No | `60` | Per-request timeout in seconds |
| `TEXT_MODEL` | Yes | — (compose fallback: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) | Model identifier passed in every chat-completions request. It serves both translation and source-language detection, so it must be instruction-tuned. Never hardcoded in Python — the fallback lives only in `docker/compose.yaml` |

translator ships no model weights and runs no local inference. Swapping
`OPENAI_API_BASE` is the only change needed to move between providers.

## Language

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DEFAULT_TARGET_LANGUAGE` | No (build-time) | `English` | Default target language, passed as the `VITE_DEFAULT_TARGET_LANGUAGE` build arg and baked into the SPA at image build. Changing it requires `make build` |
| `RESPONSE_LANGUAGE` | No | `en` | UI language of the SPA — `en` or `de`. Drives the interface chrome only; the translation target language is chosen per request in the UI and is unaffected |

## Networking and runtime

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `INFERENCE_NETWORK` | No | `inference-net` | Name of the external Docker network the backend joins |
| `EDGE_NET` | No | `edge-net` | Name of the external Docker network the frontend joins, where the edge-plane gateway reaches it as `translator-frontend` |
| `TRANSLATOR_FRONTEND_HOST_PORT` | No | `8501` | Dev-only host port for the SPA; mapped to the frontend container's nginx on `:8080` |
| `TRANSLATOR_BACKEND_HOST_PORT` | No | `8000` | Dev-only host port for the FastAPI backend |
| `LOG_LEVEL` | No | `INFO` | Minimum log level emitted on stderr |
| `EXTRA_NO_PROXY` | No | — | Comma-separated hostnames appended to `NO_PROXY` / `no_proxy`. Must start with a leading comma, e.g. `,ollama,vllm-router` |

Host ports come from `docker/compose.override.yaml`, the dev overlay that
`make up-dev` layers on. The base `docker/compose.yaml` is the production
shape and publishes nothing.

The backend writes its operational logs to stderr; its stdout carries only
the data-protection audit events (see [Audit log](#audit-log)). No log line
carries input text or model output. The backend and nginx access logs record
the request path without its query string, and nginx's also omits client IP,
referer and user agent. nginx's error log is the exception: on a
request-level error, such as an unreachable backend, it still quotes the full
request line.

## Audit log

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AUDIT_OUTPUT` | No | `stdout` (compose sets `container`) | Where the backend's audit events go: `stdout` for the process's stdout, `container` for the container's main stdout under a container-wide lock. Any other value stops startup |
| `AUDIT_REQUIRE_PURPOSE` | No | — (empty; compose default empty too) | Comma-separated audit actions that need a stated purpose (`X-Audit-Purpose`). Only `ai_call`, which every translation records, has an effect here: `AUDIT_REQUIRE_PURPOSE=ai_call` refuses a translation without a purpose with 428, and the SPA's purpose bar then asks for one. Empty turns it off. An unknown action name stops startup |

Every translation writes one `ai_call` event; [audit.md](audit.md) covers
what it records, the endpoint list, the SPA's purpose bar, and how to require
a purpose for translations or turn that off again.

## Inference provider setup

Any OpenAI-compatible endpoint works. In the federation, translator reaches
the shared LiteLLM router at `http://vllm-router:4000/v1` on `inference-net`
— see [vllm-service](https://github.com/nos-tromo/vllm-service) for bringing
that up.

For a standalone dev host with no router, Ollama is the smallest option:

```bash
docker network create inference-net
docker volume create ollama-cache

docker run -d \
  --network inference-net \
  --name ollama \
  --gpus all \
  -v ollama-cache:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:0.20.2

docker exec ollama ollama pull <model>
```

Then set `OPENAI_API_BASE=http://ollama:11434/v1` (or
`http://localhost:11434/v1` when running the backend outside Docker) and
`TEXT_MODEL=<model>`.

## Production entry

In production the SPA is served under the canonical `/translator/` sub-path
behind the `edge-plane` gateway, not at its own vhost root. The frontend joins
the external `edge-net` network as alias `translator-frontend`; the gateway is
the sole production entry point and supplies `X-Auth-User` / `X-Auth-Name`.
`POST /api/v1/translate` acts as the `X-Auth-User` account and answers 401
without one, because every translation is recorded with the account that
asked for it ([audit.md](audit.md)). This changed in 1.7.0: before, the
endpoint served anonymous requests too. `GET /api/v1/whoami` echoes both
headers back so the SPA header can show who is signed in. With the edge
secret below set, the headers count only on requests that carry that
secret. See `edge-plane` for the gateway side.

### Edge secret

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TRANSLATOR_EDGE_SECRET` | No | — | Secret shared with the edge gateway, which sends it in `X-Edge-Secret`. When set, the identity headers count only on requests that carry it. At least 32 characters; surrounding whitespace is ignored |

The backend is also on the shared `inference-net`, so any container there can
reach it and send identity headers of its own. The edge secret tells the
gateway's requests apart: the gateway adds `X-Edge-Secret` to every request it
proxies to the frontend, whose nginx passes it on to the backend. With
`TRANSLATOR_EDGE_SECRET` set:

- A request whose `X-Edge-Secret` matches the secret (compared in constant
  time) is edge-verified, and its identity headers are used.
- On any other request the identity headers are ignored, as if absent:
  `/translate` answers 401 and `/whoami` returns nulls. The backend logs a warning naming the method and path,
  never the secret or a header value.
- A value shorter than 32 characters stops the backend at startup; the error
  does not contain the value.

Unset or blank keeps the previous behaviour — identity headers are trusted
from any caller — and the backend logs one warning at startup.

Generate the secret with `make secret` in edge-plane; the same value goes into
edge-plane's `.env` (see its `docs/identity-contract.md`). Roll it out gateway
first: once the gateway sends `X-Edge-Secret`, set `TRANSLATOR_EDGE_SECRET` in
translator's `.env` and run `make up`, which recreates the backend with it. In
the other order every request is unverified until the gateway catches up, so
`/translate` answers 401, `/whoami` returns nulls and the SPA shows no
signed-in user.

### Default identity (dev only)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TRANSLATOR_DEFAULT_IDENTITY` | No | — (dev overlay: `dev`) | Account that a request without a trusted `X-Auth-User` acts as. Dev only, for a host without the gateway; ignored while `TRANSLATOR_EDGE_SECRET` is set |

Without the gateway no request carries `X-Auth-User`, so `/translate` answers
401 unless this is in effect. The dev overlay (`docker/compose.override.yaml`,
used by `make dev` and `make up-dev`) sets it to `dev`; set it in `.env` to
pick another invented account. A local run without Docker needs it in the
environment or `.env`. The production compose file does not set it. While it
is in effect the backend logs one warning at startup.

With `TRANSLATOR_EDGE_SECRET` also set it is ignored, and one startup warning
naming both variables says so: a backend that enforces the edge secret never
lets an unverified caller act as anyone, so a request without a trusted
identity gets 401 whatever the default identity says. Leave it unset in
production.
