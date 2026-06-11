"""
Metadata-aware filter construction for vector store retrieval.

This module is intentionally self-contained: it does not import from
``tools.agentTools`` or ``services.silo_service`` to avoid circular import
cycles.  All logic operates on plain Python values and Pydantic models.

Supported backends:
  - PGVector  ($eq, $ne, $gt, $gte, $lt, $lte, $in)
  - Qdrant    ($eq, $ne, $gt, $gte, $lt, $lte, $in)

Neither store's existing translation layer handles ``$and`` as a top-level
key, so ``merge_filters_and`` combines filters field-by-field.  When the same
(field, op) pair appears in two dicts with different values the *first* dict
wins — that is, pinned / agent-configured filters take priority over
LLM-inferred ones.  Conflicts are logged as WARNING without logging the actual
values (injection guard).

Recommended entry point
-----------------------
Use ``build_filter_dict`` as the single public entrypoint for converting a list
of ``MetadataFilterClause`` objects into a backend filter dict.  It encapsulates
the full pipeline: ``ops_for_backend`` → ``validate_clauses`` →
``convert_clause_types`` → ``to_backend_filter``.  The lower-level functions
are exposed for testing and for callers that need intermediate results.

Security note on ``sanitize_metadata_value``
--------------------------------------------
``sanitize_metadata_value`` is intentionally **NOT** applied to filter values.
Filter values must match stored metadata exactly (exact-match semantics), and
they are passed as SQL bind parameters so they carry no injection risk.
``sanitize_metadata_value`` is reserved for descriptions and enum labels that
are embedded verbatim in LLM tool definitions (step_005).  Applying it to
filter values would silently corrupt matches.
"""

import logging
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Operators supported by PGVector's ``_build_filter_sql`` / ``_operator_condition``.
PGVECTOR_OPS: frozenset[str] = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"})

#: Operators supported by Qdrant's ``_translate_pgvector_filter_to_qdrant``.
#: ``$in`` translates to ``FieldCondition(match=MatchAny(any=[...]))`` natively.
QDRANT_OPS: frozenset[str] = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"})

#: Maximum allowed length for a sanitized string metadata value.
#: Consumed by retriever_tool_builder (step_005).
MAX_ENUM_VALUES: int = 25

#: Maximum number of example values surfaced in error messages.
#: Consumed by retriever_tool_builder (step_005).
MAX_EXAMPLE_VALUES: int = 10

#: System-managed metadata fields always allowed in filter clauses regardless of
#: whether a ``metadata_definition`` is configured on the silo.  These are the
#: fields written during indexation:
#:   - ``page``      : PDF page number (int), written by LangChain loaders
#:   - ``name``      : file URI / resource name (str)
#:   - ``file_type`` : file extension string (str), e.g. ".pdf"
#:   - ``url``       : crawled URL (str), written by the domain crawl executor
SYSTEM_METADATA_FIELDS: frozenset[str] = frozenset({"page", "name", "url", "file_type"})

#: Canonical type annotation for the per-field type declared in OutputParser.fields
_SYSTEM_FIELD_TYPES: dict[str, str] = {
    "page": "int",
    "name": "str",
    "url": "str",
    "file_type": "str",
}

# Prompt-injection patterns — case-insensitive, matched against the raw value
# string and replaced with an empty string.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+.*?instructions?", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:previous\s+)?instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+(?:now\s+)?", re.IGNORECASE),
    re.compile(r"act\s+as\s+", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"prompt\s+injection", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class MetadataFilterClause(BaseModel):
    """A single metadata filter predicate.

    A list of clauses is implicitly combined with AND when passed to
    ``to_backend_filter``.  For ``$in`` the value must be a list.

    Validators enforce that ``field`` is a non-empty stripped string and that
    ``op`` belongs to the union of all known backend operators.  Callers that
    build clauses programmatically (e.g. tests or the agent tool builder) will
    receive a ``ValidationError`` for invalid inputs rather than a silent
    runtime failure.  ``validate_clauses`` remains the authoritative gate for
    per-backend operator whitelisting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False)

    field: str
    op: str
    value: Any

    @field_validator("field")
    @classmethod
    def field_must_be_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("MetadataFilterClause.field must be a non-empty string")
        return stripped

    @field_validator("op")
    @classmethod
    def op_must_be_known(cls, v: str) -> str:
        all_ops = PGVECTOR_OPS | QDRANT_OPS
        if v not in all_ops:
            raise ValueError(
                f"MetadataFilterClause.op '{v}' is not in the known operator set "
                f"{sorted(all_ops)}"
            )
        return v


# ---------------------------------------------------------------------------
# Backend selector helper
# ---------------------------------------------------------------------------


def ops_for_backend(vector_db_type: str) -> frozenset[str]:
    """Return the whitelist of allowed operators for the given backend type.

    Args:
        vector_db_type: The string stored in ``Silo.vector_db_type``
            (e.g. ``"PGVECTOR"`` or ``"QDRANT"``).

    Returns:
        The corresponding operator whitelist.  Defaults to the more
        restrictive Qdrant set for unknown values.
    """
    if (vector_db_type or "").upper() == "PGVECTOR":
        return PGVECTOR_OPS
    return QDRANT_OPS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_clauses(
    clauses: list[MetadataFilterClause],
    metadata_definition: Any | None,
    backend_ops: frozenset[str],
) -> list[MetadataFilterClause]:
    """Discard clauses that reference unknown fields or unsupported operators.

    Rules applied in order:
    1. If ``metadata_definition`` is not None, the clause's ``field`` must
       appear in the definition's ``fields`` list (``[{name, ...}]``) **or**
       in ``SYSTEM_METADATA_FIELDS``.
    2. If ``metadata_definition`` is None, only ``SYSTEM_METADATA_FIELDS``
       are allowed.
    3. The clause's ``op`` must be in ``backend_ops``.

    Rejected clauses are dropped silently (logged as WARNING without values).
    Never raises.

    Args:
        clauses: Input list of filter clauses.
        metadata_definition: ``OutputParser`` ORM instance (with a ``fields``
            JSON attribute) **or** None.
        backend_ops: Operator whitelist for the target backend.

    Returns:
        List containing only the valid clauses, in original order.
    """
    if metadata_definition is not None:
        raw_fields: list[dict[str, Any]] = metadata_definition.fields or []
        allowed_fields: frozenset[str] = frozenset(
            f["name"] for f in raw_fields if isinstance(f, dict) and f.get("name")
        ) | SYSTEM_METADATA_FIELDS
    else:
        allowed_fields = SYSTEM_METADATA_FIELDS

    valid: list[MetadataFilterClause] = []
    for clause in clauses:
        if clause.field not in allowed_fields:
            logger.warning(
                "metadata_filters.validate_clauses: field '%s' not in allowed fields; clause discarded",
                clause.field,
            )
            continue
        if clause.op not in backend_ops:
            logger.warning(
                "metadata_filters.validate_clauses: op '%s' on field '%s' not in backend whitelist; clause discarded",
                clause.op,
                clause.field,
            )
            continue
        valid.append(clause)
    return valid


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def _convert_single_value(value: Any, field_type: str, field: str) -> tuple[Any, bool]:
    """Attempt to coerce *value* to *field_type*.

    Returns:
        ``(converted_value, ok)`` — if conversion fails, ``ok`` is False.
    """
    if value is None:
        logger.warning(
            "metadata_filters.convert_clause_types: None value for field '%s'; clause discarded",
            field,
        )
        return None, False

    try:
        if field_type == "int":
            return int(value), True
        if field_type == "float":
            return float(value), True
        if field_type == "bool":
            if isinstance(value, bool):
                return value, True
            as_str = str(value).strip().lower()
            if as_str in {"true", "1", "yes"}:
                return True, True
            if as_str in {"false", "0", "no"}:
                return False, True
            raise ValueError(f"Cannot convert {value!r} to bool")
        # str / unknown: keep as-is
        return str(value) if not isinstance(value, str) else value, True
    except (ValueError, TypeError):
        logger.warning(
            "metadata_filters.convert_clause_types: cannot convert value for field '%s' to type '%s'; clause discarded",
            field,
            field_type,
        )
        return None, False


def convert_clause_types(
    clauses: list[MetadataFilterClause],
    metadata_definition: Any | None,
) -> list[MetadataFilterClause]:
    """Coerce clause values to the declared Python type for each field.

    Uses the ``type`` attribute from the ``metadata_definition.fields`` list.
    System metadata fields use ``_SYSTEM_FIELD_TYPES``.  Unknown field types
    are treated as ``str``.

    For ``$in`` clauses, every element of the list is converted individually;
    if any element cannot be converted the entire clause is discarded.

    Clauses for which conversion fails are dropped with a WARNING (no values
    logged).  Never raises.

    Args:
        clauses: Input clauses (already validated against the whitelist).
        metadata_definition: ``OutputParser`` ORM instance or None.

    Returns:
        List of clauses with correctly typed values.
    """
    if metadata_definition is not None:
        raw_fields: list[dict[str, Any]] = metadata_definition.fields or []
        field_type_map: dict[str, str] = {
            f["name"]: f.get("type", "str")
            for f in raw_fields
            if isinstance(f, dict) and f.get("name")
        }
    else:
        field_type_map = {}

    # System fields override user-defined types
    effective_type_map: dict[str, str] = {**field_type_map, **_SYSTEM_FIELD_TYPES}

    result: list[MetadataFilterClause] = []
    for clause in clauses:
        field_type = effective_type_map.get(clause.field, "str")

        if clause.op == "$in":
            if not isinstance(clause.value, list):
                logger.warning(
                    "metadata_filters.convert_clause_types: $in value for field '%s' is not a list; clause discarded",
                    clause.field,
                )
                continue

            converted_list: list[Any] = []
            ok = True
            for item in clause.value:
                converted, item_ok = _convert_single_value(item, field_type, clause.field)
                if not item_ok:
                    ok = False
                    break
                converted_list.append(converted)

            if not ok:
                continue
            if not converted_list:
                logger.warning(
                    "metadata_filters.convert_clause_types: $in list for field '%s' is empty after conversion; clause discarded",
                    clause.field,
                )
                continue
            result.append(MetadataFilterClause(field=clause.field, op=clause.op, value=converted_list))
        else:
            converted, ok = _convert_single_value(clause.value, field_type, clause.field)
            if not ok:
                continue
            result.append(MetadataFilterClause(field=clause.field, op=clause.op, value=converted))

    return result


# ---------------------------------------------------------------------------
# Filter dict translation
# ---------------------------------------------------------------------------


def to_backend_filter(clauses: list[MetadataFilterClause]) -> dict[str, Any]:
    """Translate a list of clauses to a Mongo-style filter dict.

    The resulting dict is accepted by both ``PGVectorStore._build_filter_sql``
    and ``QdrantStore._translate_pgvector_filter_to_qdrant`` without further
    transformation.

    Multiple clauses on the **same field** are combined inside a single field
    dict (e.g. ``{field: {"$gte": 2020, "$lte": 2024}}``), which both stores
    interpret as AND at the field level.  Multiple distinct fields are AND-ed
    implicitly by both stores' translation loops.

    Args:
        clauses: Validated, type-converted clauses.

    Returns:
        A ``{field: {op: value}}`` dict (empty dict if *clauses* is empty).
    """
    result: dict[str, Any] = {}
    for clause in clauses:
        if clause.field not in result:
            result[clause.field] = {}
        if clause.op in result[clause.field]:
            if result[clause.field][clause.op] != clause.value:
                logger.warning(
                    "metadata_filters.to_backend_filter: duplicate (field='%s', op='%s'); keeping first value",
                    clause.field,
                    clause.op,
                )
        else:
            result[clause.field][clause.op] = clause.value
    return result


# ---------------------------------------------------------------------------
# Filter merging
# ---------------------------------------------------------------------------


def merge_filters_and(*filter_dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple Mongo-style filter dicts with AND semantics.

    Since neither PGVector nor Qdrant supports a ``$and`` top-level key in
    their current translation layers, this function merges at the
    field / operator level.

    Conflict policy — when the **same (field, op) pair** appears in two dicts
    with *different* values:
    - The value from the **first** dict that contains the key is kept.
    - The later value is discarded with a WARNING (no actual values logged).

    This ensures that pinned / agent-level filters always win over
    LLM-inferred ones when the caller passes them first.

    Args:
        *filter_dicts: Dicts in priority order (highest first).

    Returns:
        A merged ``{field: {op: value}}`` dict.
    """
    merged: dict[str, Any] = {}

    for filter_dict in filter_dicts:
        for field, spec in filter_dict.items():
            if field not in merged:
                # First occurrence of this field — copy entirely.
                merged[field] = dict(spec) if isinstance(spec, dict) else spec
                continue

            if not isinstance(spec, dict):
                # Plain equality shorthand
                if merged[field] != spec:
                    logger.warning(
                        "metadata_filters.merge_filters_and: conflict on field '%s'; keeping first value",
                        field,
                    )
                continue

            if not isinstance(merged[field], dict):
                # Existing entry was a plain value; keep it, skip all ops from later dict
                logger.warning(
                    "metadata_filters.merge_filters_and: conflict on field '%s' (plain vs op dict); keeping first value",
                    field,
                )
                continue

            for op, value in spec.items():
                if op in merged[field]:
                    if merged[field][op] != value:
                        logger.warning(
                            "metadata_filters.merge_filters_and: conflict on field '%s' op '%s'; keeping first value",
                            field,
                            op,
                        )
                else:
                    merged[field][op] = value

    return merged


# ---------------------------------------------------------------------------
# Prompt-injection sanitization
# ---------------------------------------------------------------------------


def sanitize_metadata_value(value: str, max_len: int = 80) -> str:
    """Sanitize a string metadata value to mitigate prompt-injection risk.

    **Scope**: this function is intended exclusively for descriptions and enum
    labels that are embedded verbatim in LLM tool definitions (step_005).  It
    must NOT be applied to filter values: filter values must match stored
    metadata exactly, and they are passed as SQL bind parameters so they carry
    no injection risk.  Applying this function to filter values would silently
    corrupt match results.

    Transformations applied (in order):
    1. Non-string inputs are returned as-is (caller is responsible for type
       safety; this guard prevents unexpected AttributeError).
    2. Unicode NFKC normalization — reduces full-width / homoglyph characters
       (e.g. fullwidth Latin) to their ASCII equivalents before any other check.
    3. Collapse whitespace — newlines, tabs and multiple spaces are reduced
       to a single ASCII space.
    4. Remove characters commonly used to escape context or inject markup:
       backtick `` ` ``, angle brackets ``<`` / ``>``, curly braces ``{`` / ``}``.
    5. Neutralize known prompt-injection instruction patterns (case-insensitive)
       by removing matched substrings entirely.
    6. Truncate to *max_len* characters (default 80).

    This function never raises.

    Args:
        value: Raw string from user-controlled metadata description/enum input.
        max_len: Maximum length of the returned string.

    Returns:
        Sanitized string safe to embed verbatim in an LLM tool definition.
    """
    if not isinstance(value, str):
        return value  # type: ignore[return-value]

    # Step 1: NFKC normalization — map fullwidth / homoglyph chars to ASCII
    sanitized = unicodedata.normalize("NFKC", value)

    # Step 2: collapse all whitespace (newlines, tabs, runs of spaces)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Step 3: remove injection-facilitating punctuation
    sanitized = sanitized.translate(str.maketrans("", "", "`<>{}"))

    # Step 4: neutralize instruction patterns
    for pattern in _INJECTION_PATTERNS:
        sanitized = pattern.sub("", sanitized)

    # Step 5: re-collapse any whitespace introduced by pattern removal
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Step 6: truncate
    return sanitized[:max_len]


# ---------------------------------------------------------------------------
# Composite public entrypoint
# ---------------------------------------------------------------------------


def build_filter_dict(
    clauses: list[MetadataFilterClause],
    metadata_definition: Any | None,
    vector_db_type: str,
) -> dict[str, Any]:
    """Build a backend-ready filter dict from a list of clauses.

    This is the **recommended entrypoint** for consumers (retriever tool
    builder, step_005 and later steps).  It encapsulates the full validation
    and conversion pipeline:

    1. ``ops_for_backend(vector_db_type)`` — derive the operator whitelist.
    2. ``validate_clauses(clauses, metadata_definition, backend_ops)`` — discard
       clauses with unknown fields or unsupported operators.
    3. ``convert_clause_types(valid_clauses, metadata_definition)`` — coerce
       values to the declared Python type; discard on failure.
    4. ``to_backend_filter(typed_clauses)`` — translate to the Mongo-style
       ``{field: {op: value}}`` dict accepted by both store backends.

    Invalid clauses are silently discarded with WARNING log entries (no values
    logged).  The function never raises.

    Args:
        clauses: Raw filter clauses, typically produced by the LLM tool call.
        metadata_definition: ``OutputParser`` ORM instance (with a ``fields``
            JSON attribute) or None.
        vector_db_type: The string stored in ``Silo.vector_db_type``
            (e.g. ``"PGVECTOR"`` or ``"QDRANT"``).

    Returns:
        A ``{field: {op: value}}`` filter dict, empty if no valid clauses
        remain.
    """
    backend_ops = ops_for_backend(vector_db_type)
    valid = validate_clauses(clauses, metadata_definition, backend_ops)
    typed = convert_clause_types(valid, metadata_definition)
    return to_backend_filter(typed)
