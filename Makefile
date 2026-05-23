### translator — build-host helpers.
###
### translator is a thin app: FastAPI backend + Streamlit frontend that
### call inference over the shared `inference-net`. It has no local GPU
### code and no persistent state of its own, so there are no
### `cpu`/`cuda` profiles and `docker compose down -v` is always safe.

.DEFAULT_GOAL := help

.PHONY: help network build bundle up stop down logs pre-commit test

# Versioned image tag.
# On production: read from .translator-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting TRANSLATOR_VERSION before invoking make.
TRANSLATOR_VERSION ?= $(shell \
    cat .translator-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export TRANSLATOR_VERSION

COMPOSE := docker compose --env-file .env -f docker/compose.yaml -f docker/compose.override.yaml

help:
	@echo "translator — FastAPI backend + Streamlit frontend."
	@echo
	@echo "  make network    create the shared inference-net"
	@echo "  make build      build backend + frontend images"
	@echo "  make bundle     ship images as a versioned .tar.gz (built locally)"
	@echo "  make up         start backend + frontend"
	@echo "  make stop       stop containers (keep them)"
	@echo "  make down       stop + remove containers"
	@echo "  make logs       tail combined logs"
	@echo "  make pre-commit run pre-commit hooks (ruff + mypy)"
	@echo "  make test       run pytest"

# Create the shared external network (one-time per host; idempotent).
network:
	docker network create inference-net >/dev/null 2>&1 || true

# Build backend + frontend images.
build:
	DOCKER_BUILDKIT=1 $(COMPOSE) build

# Build images and ship as a versioned .tar.gz (built + pulled).
bundle:
	./scripts/bundle_images.sh

# Start backend + frontend without rebuilding.
up:
	DOCKER_BUILDKIT=1 $(COMPOSE) up --no-build

# Stop containers without removing them.
stop:
	$(COMPOSE) stop

# Stop + remove containers. No application-state volumes exist, so this
# is always safe.
down:
	$(COMPOSE) down

# Tail combined logs.
logs:
	$(COMPOSE) logs -f --tail=100

# Run pre-commit hooks (ruff + mypy).
pre-commit:
	uv run pre-commit run --all-files

# Run the test suite.
test:
	uv run pytest -q
