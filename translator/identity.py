"""Caller identity from the edge gateway's headers, trusted only with the edge secret.

The edge gateway (edge-plane) authenticates every request and sets
``X-Auth-User`` / ``X-Auth-Name`` on it. The backend is also reachable on the
shared ``inference-net``, where any container could set those headers itself.
``TRANSLATOR_EDGE_SECRET`` closes that gap: the gateway sends the same value in
``X-Edge-Secret``, and while the secret is configured the identity headers
count only on a request that carries it.

:func:`resolve_identity` is the one place that decision is made; any code that
needs the caller's identity calls it instead of reading the headers.
"""

import hmac
import os
from dataclasses import dataclass
from functools import cache

from fastapi import Request
from loguru import logger

EDGE_SECRET_ENV = "TRANSLATOR_EDGE_SECRET"
EDGE_SECRET_HEADER = "X-Edge-Secret"
MIN_EDGE_SECRET_LENGTH = 32

USER_HEADER = "X-Auth-User"
NAME_HEADER = "X-Auth-Name"
IDENTITY_HEADERS = (USER_HEADER, NAME_HEADER)


@dataclass(frozen=True)
class Identity:
    """The caller's identity, as far as the backend trusts it.

    Attributes:
        username: The ``X-Auth-User`` value, or ``None`` when absent or not trusted.
        display_name: The ``X-Auth-Name`` value (Authelia displayname), or
            ``None`` when absent or not trusted.
    """

    username: str | None = None
    display_name: str | None = None


@cache
def get_edge_secret() -> str | None:
    """Return the edge secret, read from ``TRANSLATOR_EDGE_SECRET`` once per process.

    ``translator.main`` calls this at import, so a bad value stops the backend
    before it serves a request, as a missing ``TEXT_MODEL`` does.

    Returns:
        The secret with surrounding whitespace removed, or ``None`` when the
        variable is unset or blank. ``None`` keeps the legacy behaviour of
        trusting identity headers from any caller, and logs one warning.

    Raises:
        ValueError: If the secret is shorter than ``MIN_EDGE_SECRET_LENGTH``
            characters. The message does not contain the value.
    """
    secret = os.getenv(EDGE_SECRET_ENV, "").strip()
    if not secret:
        logger.warning(
            f"{EDGE_SECRET_ENV} is not set: identity headers are trusted from any caller that can reach "
            f"the backend. Set {EDGE_SECRET_ENV} (see edge-plane docs/identity-contract.md)."
        )
        return None
    if len(secret) < MIN_EDGE_SECRET_LENGTH:
        raise ValueError(f"{EDGE_SECRET_ENV} must be at least {MIN_EDGE_SECRET_LENGTH} characters long.")
    return secret


def _is_edge_verified(request: Request, secret: str) -> bool:
    """Tell whether a request carries the edge secret.

    Args:
        request: The incoming request.
        secret: The configured edge secret.

    Returns:
        ``True`` when ``X-Edge-Secret`` equals the secret, compared in constant
        time; ``False`` when it differs or is missing.
    """
    received = request.headers.get(EDGE_SECRET_HEADER)
    if received is None:
        return False
    # Starlette decodes header values as latin-1; encoding back yields the bytes
    # as sent, so any value compares without error.
    return hmac.compare_digest(received.encode("latin-1"), secret.encode())


def resolve_identity(request: Request) -> Identity:
    """Resolve the caller's identity from the gateway's identity headers.

    With no edge secret configured the headers are taken as sent. With one,
    they count only on an edge-verified request; on any other request they are
    ignored, as if absent, and a warning names the method and path. Neither
    the secret nor any header value is logged.

    Args:
        request: The incoming request.

    Returns:
        The trusted :class:`Identity`. Its fields are ``None`` when no trusted
        identity header arrived.
    """
    secret = get_edge_secret()
    if secret is None or _is_edge_verified(request, secret):
        return Identity(
            username=request.headers.get(USER_HEADER),
            display_name=request.headers.get(NAME_HEADER),
        )
    if any(name in request.headers for name in IDENTITY_HEADERS):
        logger.warning(
            f"Ignored identity headers without a valid {EDGE_SECRET_HEADER} on {request.method} {request.url.path}"
        )
    return Identity()
