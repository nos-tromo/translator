### translator — build-host helpers.
###
### translator is a thin app: FastAPI backend + React SPA (nginx) frontend that
### call inference over the shared `inference-net`. It has no local GPU
### code and no persistent state of its own, so there are no
### `cpu`/`cuda` profiles and `docker compose down -v` is always safe.
###
### The lifecycle targets (network/build/bundle/up/up-dev/stop/down/logs/
### pre-commit/test) + the versioned image tag come from make/common.mk,
### vendored from nos-tromo/.github. Only translator-specific config and the
### help text live here.

.DEFAULT_GOAL := help

REPO     := translator
NETWORKS := inference-net edge-net
include make/common.mk

.PHONY: help
help:
	@echo "translator — FastAPI backend + React SPA (nginx)."
	@echo
	@echo "  make network    create the shared inference-net"
	@echo "  make build      build backend + frontend images"
	@echo "  make bundle     ship images as a versioned .tar.gz (latest annotated release tag)"
	@echo "  make bundle-dev like 'bundle', but from the current working tree (dev/soak)"
	@echo "  make up         start backend + frontend, detached; no build (production shape, no host ports)"
	@echo "  make up-dev     like 'up' + host ports; detached, no build (run 'make build' first)"
	@echo "  make dev        build, then up-dev"
	@echo "  make stop       stop containers (keep them)"
	@echo "  make down       stop + remove containers"
	@echo "  make logs       tail combined logs"
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make verify     pre-push gate: pre-commit + frontend lint/build; mirrors CI's lint gate"
	@echo "  make test       run pytest"
