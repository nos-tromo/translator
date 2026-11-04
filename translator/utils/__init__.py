"""Modules shared verbatim across the federation.

``audit_event.py`` and ``audit_runtime.py`` are copies of the canonical files
in ``nos-tromo/.github`` ``configs/audit/``, kept byte-identical and checked by
that repo's ``scripts/validate_audit_lib.py``. Never edit them here: change the
canonical file and copy it again. :mod:`translator.audit` is translator's own
glue around them.
"""
