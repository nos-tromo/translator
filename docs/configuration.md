# Configuration

Every environment variable translator reads, with its default. Copy
`.env.example` to `.env` and fill in your values — that file is the source of
truth for what compose passes into the containers.

Only `OPENAI_API_BASE` and `TEXT_MODEL` are required; everything else has a
working default.

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
| `INFERENCE_NETWORK` | No | `inference-net` | Name of the external Docker network to join |
| `TRANSLATOR_FRONTEND_HOST_PORT` | No | `8501` | Dev-only host port for the SPA; mapped to the frontend container's nginx on `:8080` |
| `TRANSLATOR_BACKEND_HOST_PORT` | No | `8000` | Dev-only host port for the FastAPI backend |
| `LOG_LEVEL` | No | `INFO` | Minimum log level emitted on stderr |
| `EXTRA_NO_PROXY` | No | — | Comma-separated hostnames appended to `NO_PROXY` / `no_proxy`. Must start with a leading comma, e.g. `,ollama,vllm-router` |

Host ports come from `docker/compose.override.yaml`, the dev overlay that
`make up-dev` layers on. The base `docker/compose.yaml` is the production
shape and publishes nothing.

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
the sole production entry point and supplies an `X-Auth-User` header, which
this app ignores. See `edge-plane` for the gateway side.
