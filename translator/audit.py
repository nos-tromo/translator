"""Data-protection audit events (§76 BDSG) for translator.

translator stores nothing, so the one act it records is handing a caller's
text to the inference service. Each translation request that gets past
authentication and validation writes exactly one ``ai_call`` event when it
completes — the transfer record: who acted, the
stated purpose, the correlation id, the recipient (``inference:translate``),
the model and its reported token usage, a SHA-256 of the translation, and the
request's parameters (language pair, input length). Never the text or the
translation. ``docs/audit.md`` lists every endpoint and what it records.

This module is translator's glue around the vendored
:mod:`translator.utils.audit_event` (event format, hash chain) and
:mod:`translator.utils.audit_runtime` (output stream, purpose header, request
ids). Like :mod:`translator.identity`, it reads its own environment variables:

* ``AUDIT_OUTPUT`` — where the chain goes: unset or ``stdout`` for the
  process's stdout, ``container`` for the container's main stdout under a
  container-wide lock (what ``docker/compose.yaml`` sets).
* ``AUDIT_REQUIRE_PURPOSE`` — comma-separated actions that need a stated
  purpose; of these, only ``ai_call`` occurs in translator
  (:data:`ENDPOINT_ACTIONS`). Default empty. ``GET /api/v1/config`` reports
  the ones in effect (:func:`required_endpoint_actions`), so the SPA's purpose
  bar shows a purpose as required exactly when a translation needs one.

One emitter, and so one hash chain, per process: created in the FastAPI
lifespan, never at import. Audit writes fail closed: the caller turns a failed
write into a generic 500 and returns no translation.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Final

from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger

from translator.identity import Principal, get_default_identity, get_edge_secret, resolve_principal
from translator.utils.audit_event import ACTIONS, Actor, AiCall, AuditEmitter, AuditWriteError, Purpose
from translator.utils.audit_runtime import (
    PURPOSE_HEADER,
    PURPOSE_REQUIRED_STATUS,
    REQUEST_ID_HEADER,
    PurposeError,
    clean_request_id,
    open_audit_stream,
    parse_purpose_header,
    parse_required_actions,
)

#: The compose project name; the audit store binds each chain to it.
SERVICE: Final = "translator"
OUTPUT_ENV: Final = "AUDIT_OUTPUT"
REQUIRE_PURPOSE_ENV: Final = "AUDIT_REQUIRE_PURPOSE"
#: The app-state attribute holding the process's emitter.
EMITTER_STATE_KEY: Final = "audit_emitter"
#: The actor of the process's own events.
SYSTEM_ACTOR: Final = Actor(user=None, via="system")

_REQUIRABLE_ACTIONS: Final = ACTIONS - {"lifecycle"}

#: The audit actions translator's endpoints record. :func:`audited` accepts no
#: other, so this stays the complete set :func:`required_endpoint_actions` reads.
ENDPOINT_ACTIONS: Final = frozenset({"ai_call"})


@cache
def get_required_actions() -> frozenset[str]:
    """Return the actions that need a stated purpose, read from ``AUDIT_REQUIRE_PURPOSE`` once per process.

    ``translator.main`` calls this at import, so a misspelt action stops the
    backend instead of silently switching a requirement off. Any audit action
    except ``lifecycle`` is accepted, so a value shared with other services
    works here too.

    Returns:
        The listed actions; empty when the variable is unset or blank.

    Raises:
        ValueError: If an entry is not an audit action.
    """
    return parse_required_actions(os.getenv(REQUIRE_PURPOSE_ENV), known=_REQUIRABLE_ACTIONS)


def required_endpoint_actions() -> list[str]:
    """Return the actions of translator's endpoints that need a stated purpose.

    What ``GET /api/v1/config`` reports as ``require_purpose``: the SPA's
    purpose bar shows a purpose as required while this is non-empty. An
    action in ``AUDIT_REQUIRE_PURPOSE`` that no endpoint records, as in a
    list shared with other services, refuses nothing here and is left out.

    Returns:
        The actions, sorted; empty when no request needs a stated purpose.
    """
    return sorted(get_required_actions() & ENDPOINT_ACTIONS)


def operating_mode() -> dict[str, object]:
    """Describe how far this process trusts its callers, for the start-up record.

    Returns:
        The criteria of the ``admin`` / ``config.audit`` event: whether the
        edge secret is enforced, whether the dev default identity is in effect
        (only ever without the edge secret), and which actions need a stated
        purpose. Never a secret or an account.
    """
    return {
        "edge_secret_enforced": get_edge_secret() is not None,
        "default_identity_enabled": get_default_identity() is not None,
        "require_purpose": sorted(get_required_actions()),
    }


def open_emitter(service_version: str) -> AuditEmitter:
    """Create this process's emitter, writing where ``AUDIT_OUTPUT`` selects.

    Call it once per process, from the FastAPI lifespan. The stream stays
    open for the life of the process.

    Args:
        service_version: The running release, for every event.

    Returns:
        The emitter, not yet started.

    Raises:
        ValueError: If ``AUDIT_OUTPUT`` names an unknown mode.
        OSError: If the container stream cannot be opened.
    """
    return AuditEmitter(SERVICE, service_version, stream=open_audit_stream(os.getenv(OUTPUT_ENV)))


def start_chain(app: FastAPI, emitter: AuditEmitter) -> None:
    """Open the emitter's chain, record the operating mode and make it the app's emitter.

    Args:
        app: The application whose requests the emitter records.
        emitter: The process's emitter.

    Raises:
        AuditWriteError: If an event cannot be written; startup must fail then.
    """
    emitter.start()
    emitter.emit("admin", "config.audit", SYSTEM_ACTOR, criteria=operating_mode())
    setattr(app.state, EMITTER_STATE_KEY, emitter)


def get_emitter(request: Request) -> AuditEmitter:
    """Return the process's emitter, or refuse audited work without one.

    Args:
        request: The incoming request.

    Returns:
        The emitter the lifespan installed.

    Raises:
        HTTPException: 500 when no emitter is installed.
    """
    emitter = getattr(request.app.state, EMITTER_STATE_KEY, None)
    if not isinstance(emitter, AuditEmitter):
        logger.error("No audit emitter in this process; refusing audited work.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Request failed.")
    return emitter


def actor_for(principal: Principal) -> Actor:
    """Map the authenticated caller to the actor of its events.

    Args:
        principal: The caller, as :func:`~translator.identity.resolve_principal` resolved it.

    Returns:
        ``via="edge"`` for an account the gateway asserted, ``via="system"``
        for the dev default identity.
    """
    return Actor(user=principal.name, via="edge" if principal.from_gateway else "system")


def request_purpose(request: Request) -> Purpose | None:
    """Return the purpose the request states in ``X-Audit-Purpose``.

    The purpose is personal data in its own right: it goes into audit events
    only, and is neither logged nor echoed.

    Args:
        request: The incoming request.

    Returns:
        The stated purpose, or ``None`` when the request states none.

    Raises:
        HTTPException: 400 for a malformed, oversized or ambiguous value, or
            for more than one ``X-Audit-Purpose`` header.
    """
    values = request.headers.getlist(PURPOSE_HEADER)
    if len(values) > 1:
        raise _invalid_purpose()
    try:
        case_ref, reason = parse_purpose_header(values[0] if values else None)
    except PurposeError as exc:
        raise _invalid_purpose() from exc
    if case_ref is None and reason is None:
        return None
    return Purpose(case_ref=case_ref, reason=reason)


def _invalid_purpose() -> HTTPException:
    """Build the generic refusal of an invalid ``X-Audit-Purpose``.

    Returns:
        A 400 whose detail names neither the value nor what is wrong with it.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid X-Audit-Purpose header.")


@dataclass(frozen=True)
class AuditedCall:
    """One audited operation of one request.

    Carries who acts, the stated purpose and the correlation id, so each
    event of the request records them the same way.

    Attributes:
        emitter: The process's emitter.
        action: The operation's audit action, e.g. ``ai_call``.
        operation: The operation's stable name, e.g. ``translate``.
        actor: Who acts.
        purpose: The stated purpose, or ``None``.
        request_id: The gateway's correlation id, or ``None``.
    """

    emitter: AuditEmitter
    action: str
    operation: str
    actor: Actor
    purpose: Purpose | None
    request_id: str | None

    def require_purpose(self, criteria: Mapping[str, object]) -> None:
        """Refuse the operation if its action needs a stated purpose and none was given.

        Every audited endpoint calls this before it does any work.

        Args:
            criteria: The request's parameters, recorded on the refusal.

        Raises:
            HTTPException: 428 after recording an event with
                ``outcome="denied"``; 500 if that event cannot be written.
        """
        if self.purpose is not None or self.action not in get_required_actions():
            return
        try:
            self.record("denied", criteria=criteria)
        except (AuditWriteError, RuntimeError) as exc:
            logger.error(f"Audit event of a refused {self.operation} could not be written: {type(exc).__name__}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Request failed.") from exc
        raise HTTPException(status_code=PURPOSE_REQUIRED_STATUS, detail="This action requires a stated purpose.")

    def record(
        self,
        outcome: str,
        *,
        criteria: Mapping[str, object] | None = None,
        recipient: str | None = None,
        ai: AiCall | None = None,
    ) -> dict[str, object]:
        """Write the operation's event.

        Args:
            outcome: ``ok``, ``denied`` or ``error``.
            criteria: The request's parameters — never content.
            recipient: Where the data went, e.g. ``inference:translate``.
            ai: Model provenance of a model call.

        Returns:
            The event as written.

        Raises:
            AuditWriteError: If the event cannot be written.
            RuntimeError: If the emitter is closed.
        """
        return self.emitter.emit(
            self.action,
            self.operation,
            self.actor,
            outcome=outcome,
            request_id=self.request_id,
            purpose=self.purpose,
            criteria=criteria,
            recipient=recipient,
            ai=ai,
        )


def audited(action: str, operation: str) -> Callable[[Request], AuditedCall]:
    """Build the FastAPI dependency of an audited endpoint.

    The dependency resolves, in this order, the caller (401 without one,
    nothing recorded), the stated purpose (400 for a malformed header,
    nothing recorded: there is no valid operation yet), the correlation id
    and the process's emitter.

    Args:
        action: The endpoint's audit action.
        operation: The endpoint's stable operation name.

    Returns:
        The dependency, which returns the request's :class:`AuditedCall`.

    Raises:
        ValueError: If ``action`` is not in :data:`ENDPOINT_ACTIONS`; an
            endpoint with a new action adds it there first.
    """
    if action not in ENDPOINT_ACTIONS:
        raise ValueError(f"not an endpoint audit action: {action!r} (add it to ENDPOINT_ACTIONS)")

    def dependency(request: Request) -> AuditedCall:
        """Resolve the audit context of one request.

        Args:
            request: The incoming request.

        Returns:
            The request's :class:`AuditedCall`.
        """
        actor = actor_for(resolve_principal(request))
        purpose = request_purpose(request)
        return AuditedCall(
            emitter=get_emitter(request),
            action=action,
            operation=operation,
            actor=actor,
            purpose=purpose,
            request_id=clean_request_id(request.headers.get(REQUEST_ID_HEADER)),
        )

    return dependency
