# Build-host helpers for translator.

.PHONY: network volumes build bundle no-build up stop

# Versioned image tag.
# On production: read from .translator-version written by bundle_images.sh.
# On dev: compute YYYY-MM-DD[-<short-sha>] on the fly.
# Override entirely by exporting TRANSLATOR_VERSION before invoking make.
TRANSLATOR_VERSION ?= $(shell \
    cat .translator-version 2>/dev/null || \
    { _s=$$(git rev-parse --short HEAD 2>/dev/null); \
      echo "$$(date +%Y-%m-%d)$${_s:+-$$_s}"; } )
export TRANSLATOR_VERSION

# Create the external Docker volumes (one-time per host; idempotent)
network:
	DOCKER_BUILDKIT=1 docker network create inference-net

# Create the external Ollama cache
volumes:
	DOCKER_BUILDKIT=1 docker volume create ollama-cache

# Build the stack
build:
	DOCKER_BUILDKIT=1 docker compose build

# Build stack and ship as versioned .tar.gz pair (built + pulled)
bundle:
	./scripts/bundle_images.sh

# Run the stack without building
no-build:
	DOCKER_BUILDKIT=1 docker compose up --no-build

# Build and run the stack
up:
	DOCKER_BUILDKIT=1 docker compose up

# Stop the stack
stop:
	docker compose stop