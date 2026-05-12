#!/usr/bin/env bash
set -euo pipefail

# Always compute a fresh version from git so repeated bundle runs produce
# distinct tags. Uses the commit date (not the build date) for reproducibility.
# Falls back to today's date when not in a git repo.
# .translator-version (if present) is never used as input here — it is only
# written as output for production hosts.
# To pin a specific tag, set TRANSLATOR_VERSION_OVERRIDE in your shell before
# invoking make.
if [[ -n "${TRANSLATOR_VERSION_OVERRIDE:-}" ]]; then
  export TRANSLATOR_VERSION="$TRANSLATOR_VERSION_OVERRIDE"
else
  _git_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
  _git_date=$(git log -1 --format=%cs 2>/dev/null || true)
  _date="${_git_date:-$(date +%Y-%m-%d)}"
  export TRANSLATOR_VERSION="${_date}${_git_sha:+-${_git_sha}}"
fi
echo "TRANSLATOR_VERSION=$TRANSLATOR_VERSION"

# Persist the version so production hosts can run 'make no-build-*' without
# git or the original build date. Copy this file alongside docker-compose.yml.
echo "$TRANSLATOR_VERSION" > .translator-version

# Build locally-defined services
docker compose build

# Partition compose's image list and ensure local tag bindings exist:
#   built  = local-only names like "translator-backend" (already tagged by build)
#
# Docker Desktop sometimes drops the name:tag binding when you pull
# `name:tag@digest`, leaving only the digest. We re-tag explicitly so
# `docker save` produces a tarball that loads back with both tag and digest
# bindings — which compose needs for its `image: name:tag@digest` references.
built=()
pulled=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  if [[ "$img" == */* ]]; then
    if [[ "$img" =~ ^(.+):([^@]+)@(sha256:[a-f0-9]+)$ ]]; then
      name="${BASH_REMATCH[1]}"
      tag="${BASH_REMATCH[2]}"
      digest="${BASH_REMATCH[3]}"
      docker tag "${name}@${digest}" "${name}:${tag}"
      pulled+=("${name}:${tag}")
    else
      pulled+=("$img")
    fi
  else
    built+=("$img")
  fi
done < <(docker compose config --images)

echo "Built images:  ${built[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "translator-built-${TRANSLATOR_VERSION}.tar.gz"
fi

echo "Wrote: translator-built-${TRANSLATOR_VERSION}.tar.gz"
