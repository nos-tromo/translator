# Audit log

translator writes a data-protection audit log, with §76 BDSG as the reference
standard: one hash-chained JSON event per audited operation on the backend's
stdout, which obs-plane routes to its audit store. This page covers what
translator records and the settings that shape it. The store, its retention
and who may read it belong to obs-plane: see its
[`docs/audit.md`](https://github.com/nos-tromo/obs-plane/blob/main/docs/audit.md).

## What is recorded

translator stores nothing. The one act it records is handing a caller's text
to the inference service: every translation request writes exactly one
`ai_call` event when it completes. That event is the transfer record. A
request refused before it can become an operation writes none: without an
authenticated caller (401), with an invalid body (422), or with a malformed
purpose header (400).

| Field | Value |
|---|---|
| `action` / `operation` | `ai_call` / `translate` |
| `outcome` | `ok`; `error` when the translation failed; `denied` when a required purpose was missing (see [Purpose](#why-the-stated-purpose)) |
| `actor` | the signed-in account (`via: edge`), or the dev default identity (`via: system`) — see [Actor](#who-the-actor) |
| `purpose` | the stated purpose, when the request states one |
| `request_id` | the gateway's `X-Request-ID`, when it is a plausible id |
| `recipient` | `inference:translate` |
| `criteria` | `source_lang` (as requested; `null` means detect it), `detected_lang` (only when the model detected the source language; `null` if it could not), `target_lang`, `input_chars` |
| `ai` | `model_alias` (`TEXT_MODEL`), `model_id` (the model the endpoint reports having served), `input_tokens` / `output_tokens` (summed over the request's completed model calls, given only when the endpoint reported usage for each), `output_sha256` (SHA-256 of the translation returned; absent on `error` and `denied`) |

Never recorded: the text, the translation, the prompt, or the gateway's
display name. When the source language is detected, the model also receives
the first 500 characters of the text in a separate call to the same recipient;
a `detected_lang` field marks such a request.

The output hash lets a copy of a translation found later be matched to its
event without the log holding the translation itself.

## Endpoints

| Endpoint | Audited as | Reason |
|---|---|---|
| `POST /api/v1/translate` | `ai_call` / `translate` | hands the caller's text to the inference service |
| `GET /api/v1/languages` | — | static list, no personal data |
| `GET /api/v1/version` | — | the release number |
| `GET /api/v1/config` | — | the SPA's bootstrap config: UI language and the actions that need a purpose |
| `GET /api/v1/whoami` | — | echoes the caller's own identity headers, for display |
| `GET /api/v1/health` | — | liveness probe |
| `GET /metrics` | — | aggregate request metrics, no request content |
| `GET /docs`, `/redoc`, `/openapi.json` | — | API description |

translator has no CLI and runs no background jobs, so nothing outside a
request is recorded except the chain's own events (below).

## Who: the actor

`POST /api/v1/translate` needs an authenticated caller
(`identity.py::resolve_principal`):

1. A trusted `X-Auth-User` makes the actor that account, `via: edge`. When
   the header is trusted is the edge secret's rule, see
   [configuration](configuration.md#edge-secret).
2. Without one, `TRANSLATOR_DEFAULT_IDENTITY`, the dev-only fallback for a
   host without the gateway (the dev overlay sets `dev`), makes the actor that
   account, `via: system`. It counts only while `TRANSLATOR_EDGE_SECRET` is
   unset; with both set it is ignored.
3. Otherwise the request gets 401 and nothing is recorded. An unauthenticated
   request has no actor, and recording such requests would let any container
   on the shared networks flood the log.

A request refused for a missing purpose (428) is authenticated and is
recorded as `denied`.

## Why: the stated purpose

A request states its purpose in the `X-Audit-Purpose` header: the
form-encoded serialization of up to two fields, `case_ref` and `reason`, for
example `case_ref=case-0001&reason=Quality+review`. The SPA sends it from its
[purpose bar](#the-purpose-bar). The purpose goes into the event and nowhere
else: no ops log line carries it (of the request headers, the frontend's
nginx access log records only the request id), and the inference service
never receives it.

- A malformed, oversized or ambiguous value, or more than one
  `X-Audit-Purpose` header, gets 400 and nothing is recorded: there is no
  valid operation yet.
- `AUDIT_REQUIRE_PURPOSE` lists the actions that need a stated purpose. Of
  these, only `ai_call` occurs in translator, so **`ai_call` is the value that
  requires a purpose for translations**. With it listed, a translation without
  a purpose is recorded as `denied` (with its criteria) and answered with 428;
  it never reaches the model. Any other action in the list refuses nothing
  here: translator stores nothing, so it never queries or discloses stored
  data, and a `query,disclose` list shared with the apps that do changes
  nothing for it.

### The default, and how to change it

The default is empty, in the code and in `docker/compose.yaml`
(`AUDIT_REQUIRE_PURPOSE: ${AUDIT_REQUIRE_PURPOSE:-}`): translations need no
purpose, though the bar still lets the user state one and every request then
carries it. Requiring one stays an operator's choice for translator.

- To require a purpose for every translation, set
  `AUDIT_REQUIRE_PURPOSE=ai_call` in `.env` and run `make up`, which recreates
  the backend with it. From then on a translation without a purpose is
  refused with 428, and the SPA's bar asks for one.
- To turn it off again, set it empty (`AUDIT_REQUIRE_PURPOSE=`) or remove the
  line, and run `make up`.

The backend reads the value once at startup; an unknown action name stops it.

## The purpose bar

The SPA's header carries `@infra/ui`'s `PurposeBar`, where the user states a
case reference, a reason, or both, once. The SPA's fetch wrapper
(`frontend/src/api/client.ts`) sends it as `X-Audit-Purpose` with every API
request, under the scope `translator`; while none is stated, no header is
sent.

- The bar is always mounted, so a purpose can be stated whether or not one is
  required. `GET /api/v1/config` reports `require_purpose`: the actions listed
  in `AUDIT_REQUIRE_PURPOSE` that translator's endpoints record, sorted —
  today `["ai_call"]` or `[]`. While it is non-empty, the empty bar is a
  warning ("Purpose required") and a blank purpose cannot be saved.
- A request answered with 428 opens the bar's dialog; when the user saves a
  purpose, the request is sent once more with it. The SPA's 120-second limit
  for a translation applies to each attempt, so the time spent in the dialog
  does not shorten the resend. Cancelling leaves the 428,
  and the SPA says so in its own words instead of its generic error: the
  action needs a stated purpose, to be stated in the header.
- The purpose stays in the browser tab's `sessionStorage` — across reloads,
  never into another tab — kept apart from the other apps' behind the same
  gateway, until the user changes or clears it, closes the tab, or signs out
  through the header's account menu, which clears every app's purpose.
- The bar's labels follow `RESPONSE_LANGUAGE` (English or German).

### Requests the SPA makes

Every request the SPA sends to the API is a `fetch` through that wrapper. It
makes none that a header cannot ride on: no download link, media element,
`EventSource`, `window.open` or form post against the API.

| Request | Audited as | How it carries the purpose |
|---|---|---|
| `POST /api/v1/translate` | `ai_call` / `translate` | `X-Audit-Purpose` from the fetch wrapper; a 428 asks through the bar and sends once more |
| `GET /api/v1/config`, `/languages`, `/version`, `/whoami` | — | the same wrapper sends it; these endpoints do not use it |
| `.txt` upload | — | read in the browser; nothing is sent until the user translates |
| Copy translation | — | the clipboard; no request |
| The page and its static assets (`/translator/`, `/translator/assets/…`) | — | served by nginx; no API request |

## Correlation

The event carries the gateway's `X-Request-ID` as `request_id` when it is 1
to 128 letters, digits and `._:-`, starting with a letter or digit (anything
else is dropped, not repaired). The same id goes to the inference service as
`X-Request-ID` on each model call of the request — one, or two with
source-language detection — so the service's own records can be matched with
the event. Nothing else about the caller goes upstream: never the identity,
the purpose or the edge secret.

## Fail closed

- If a translation's event cannot be written, the request fails with a
  generic 500 and the translation is not returned. If a refusal's event
  cannot be written, the request gets the same generic 500.
- If the chain cannot be opened at startup (an unknown `AUDIT_OUTPUT`, a
  stream that cannot be opened), the backend does not start; nor with an
  unknown action in `AUDIT_REQUIRE_PURPOSE`.

## The chain

Each backend process writes its own chain, created at startup (the FastAPI
lifespan), never at import. Its events carry `service: translator`, the
compose project name the audit store binds the chain to, and the release as
`service_version`:

- `lifecycle` / `start` opens it;
- `admin` / `config.audit` (actor `via: system`) records the operating
  mode: `edge_secret_enforced`, `default_identity_enabled` and
  `require_purpose` (every action `AUDIT_REQUIRE_PURPOSE` lists, including
  any translator does not record). Never a secret or an account;
- `lifecycle` / `heartbeat` follows every 60 seconds;
- `lifecycle` / `shutdown` closes it on a clean stop.

The event format and the hash chain come from
`translator/utils/audit_event.py`; the stream, the purpose header and the
request-id rule from `translator/utils/audit_runtime.py`. Both are
byte-identical copies of the canonical files in `nos-tromo/.github`
(`configs/audit/`) and are never edited here. `translator/audit.py` is
translator's own glue around them.

## Where the lines go

stdout carries audit events only; every operational log line goes to stderr,
including uvicorn's access log, which uvicorn's default config would write
to stdout (`log_cfg.py`). With `AUDIT_OUTPUT=container`, which
`docker/compose.yaml` sets, events go to the container's main stdout
(`/proc/1/fd/1`) under a lock held by every audit writer in the container.
The lock file is `/tmp/audit-stream.lock` on the backend's `/tmp` tmpfs.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `AUDIT_OUTPUT` | `stdout` (compose sets `container`) | Where the chain goes: `stdout` for the process's stdout, `container` for the container's main stdout under the container-wide lock. Any other value stops startup |
| `AUDIT_REQUIRE_PURPOSE` | empty (compose too) | Comma-separated actions that need a stated purpose; only `ai_call` occurs in translator, so `ai_call` makes every translation without a purpose a 428, and empty turns that off ([above](#the-default-and-how-to-change-it)). Any audit action except `lifecycle` is accepted, so a list shared with other services works; any other name stops startup |
| `TRANSLATOR_DEFAULT_IDENTITY` | — (dev overlay: `dev`) | Dev-only fallback account for requests without a trusted identity; ignored while `TRANSLATOR_EDGE_SECRET` is set. Leave it unset in production |

`AUDIT_CASE_REF`, `AUDIT_REASON` and `AUDIT_CLI_USER` state the purpose and
operator of command-line runs in the federation's other apps. translator has
no CLI and reads none of them.
