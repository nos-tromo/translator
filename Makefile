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
NETWORKS := inference-net
UP_ENV   := DOCKER_BUILDKIT=1
UP_FLAGS := --no-build
include make/common.mk

.PHONY: help
help:
	@echo "translator — FastAPI backend + React SPA (nginx)."
	@echo
	@echo "  make network    create the shared inference-net"
	@echo "  make build      build backend + frontend images"
	@echo "  make bundle     ship images as a versioned .tar.gz (built locally)"
	@echo "  make up         start backend + frontend (production shape, no host ports)"
	@echo "  make up-dev     like 'up', but publishes backend + frontend ports on the host"
	@echo "  make stop       stop containers (keep them)"
	@echo "  make down       stop + remove containers"
	@echo "  make logs       tail combined logs"
	@echo "  make pre-commit run pre-commit hooks (ruff + pyrefly)"
	@echo "  make test       run pytest"
