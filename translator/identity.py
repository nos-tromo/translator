"""Caller identity from the edge gateway's headers, trusted only with the edge secret.

The edge gateway (edge-plane) authenticates every request and sets
``X-Auth-User`` / ``X-Auth-Name`` on it. The backend is also reachable on the
shared ``inference-net``, where any container could set those headers itself.
``TRANSLATOR_EDGE_SECRET`` closes that gap: the gateway sends the same value in
``X-Edge-Secret``, and while the secret is configured the identity headers
count only on a request that carries it.

:func:`resolve_identity` is the one place that decision is made; any code that
needs the caller's identity calls it instead of reading the headers.

An endpoint that must know who acts — ``POST /api/v1/translate``, whose every
call is recorded in the data-protection audit log — depends on
:func:`resolve_principal` instead, which fails closed with 401 when there is no
trusted identity. ``TRANSLATOR_DEFAULT_IDENTITY`` is its dev-only fallback for
a host without the gateway. It counts only while ``TRANSLATOR_EDGE_SECRET`` is
unset: a backend that enforces the edge secret never lets an unverified caller
act as anyone.
"""

import hmac
import os
from dataclasses import dataclass
from functools import cache

from fastapi import HTTPException, Request, status
from loguru import logger

EDGE_SECRET_ENV = "TRANSLATOR_EDGE_SECRET"
EDGE_SECRET_HEADER = "X-Edge-Secret"
MIN_EDGE_SECRET_LENGTH = 32
DEFAULT_IDENTITY_ENV = "TRANSLATOR_DEFAULT_IDENTITY"

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


@dataclass(frozen=True)
class Principal:
    """The authenticated caller of an endpoint that needs one.

    Attributes:
        name: The account the request acts as.
        from_gateway: ``True`` when ``name`` is the trusted ``X-Auth-User``
            value, ``False`` when it is the dev fallback
            ``TRANSLATOR_DEFAULT_IDENTITY``.
    """

    name: str
    from_gateway: bool


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


@cache
def get_default_identity() -> str | None:
    """Return the dev fallback identity in effect, read from ``TRANSLATOR_DEFAULT_IDENTITY`` once per process.

    The fallback counts only while no edge secret is set. With both set it
    is ignored, so a request without a trusted identity still fails closed.
    ``translator.main`` calls this at import, so either case is announced by
    one warning at startup. Neither warning names the account.

    Returns:
        The identity with surrounding whitespace removed, or ``None`` when the
        variable is unset or blank, or ignored because
        ``TRANSLATOR_EDGE_SECRET`` is set.
    """
    identity = os.getenv(DEFAULT_IDENTITY_ENV, "").strip()
    if not identity:
        return None
    if get_edge_secret() is not None:
        logger.warning(
            f"{DEFAULT_IDENTITY_ENV} is ignored because {EDGE_SECRET_ENV} is set: "
            "requests without a trusted identity get 401."
        )
        return None
    logger.warning(
        f"{DEFAULT_IDENTITY_ENV} is set: requests without a trusted identity act as that account. "
        "Dev only; leave it unset in production."
    )
    return identity


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


def resolve_principal(request: Request) -> Principal:
    """Return the authenticated caller, or refuse the request.

    The trusted ``X-Auth-User`` (as :func:`resolve_identity` decides it)
    comes first; without one, the default identity in effect (see
    :func:`get_default_identity`). Use it as a FastAPI dependency.

    Args:
        request: The incoming request.

    Returns:
        The :class:`Principal` the request acts as.

    Raises:
        HTTPException: 401 when the request carries no trusted identity and
            no default identity is in effect.
    """
    username = resolve_identity(request).username
    if username:
        return Principal(name=username, from_gateway=True)
    default_identity = get_default_identity()
    if default_identity:
        return Principal(name=default_identity, from_gateway=False)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authenticated principal.",
        headers={"WWW-Authenticate": "Bearer"},
    )
