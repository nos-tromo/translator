# Build-host helpers for translator.

.PHONY: network volumes build bundle no-build up

# Versioned image tag: YYYY-MM-DD-<short-sha>. Override by exporting
# TRANSLATOR_VERSION before invoking make. Mirrors scripts/bundle_images.sh.
TRANSLATOR_VERSION ?= $(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD)
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
	DOCKER_BUILDKIT=1 docker compose up -d --no-build

# Build and run the stack
up:
	DOCKER_BUILDKIT=1 docker compose up -d
