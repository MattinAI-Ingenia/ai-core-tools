"""
Pure-function builders that generate the name, description, and argument schema
for a silo's dynamic retrieval tool.

These functions are *pure*: they do not access the database, the vector store,
or any external service.  All runtime data (distinct metadata field values) must
be supplied by the caller, typically via :func:`collect_distinct_values`.

Usage pattern (step_006)::

    distinct = await asyncio.to_thread(collect_distinct_values, silo, db)
    schema   = build_retriever_args_schema(silo, distinct)
    desc     = build_retriever_description(silo, distinct)
    name     = build_retriever_tool_name(silo)

Prompt-injection hardening
--------------------------
All admin-supplied strings (field names, descriptions, enum labels) that end up
embedded in LLM tool definitions are passed through
``sanitize_metadata_value`` from :mod:`tools.vector_stores.metadata_filters`.
This applies only to text that the LLM reads — never to filter values that are
later compared against stored metadata.

Type mapping
------------
The ``type`` attribute in ``metadata_definition.fields`` is a free string set
by the UI.  The canonical mapping is::

    "str"  / "string"             → str
    "int"  / "integer"            → int
    "float"/ "number"             → float
    "bool" / "boolean"            → bool
    anything else                  → str  (WARNING logged)

Enum / Literal policy
---------------------
When a field has ≤ MAX_ENUM_VALUES (25) distinct values the schema uses a
``Literal`` type so that the LLM can only supply valid values.  When there are
more than 25 distinct values (or none at all) the schema uses the free base type
and embeds up to MAX_EXAMPLE_VALUES (10) sanitised example values in the field
description.

Filter-usage policy embedded in descriptions
---------------------------------------------
Every optional filter field carries the instruction:
  "Only use this filter if the user's question explicitly mentions it; never
  invent a value."
This implements the best practice described in the metadata-aware-retrieval spec
and guards against the LLM hallucinating filter values.
"""

from __future__ import annotations

import keyword
import logging
import re
import unicodedata
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from tools.vector_stores.metadata_filters import (
    MAX_ENUM_VALUES,
    MAX_EXAMPLE_VALUES,
    sanitize_metadata_value,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_FILTER_USAGE_POLICY = (
    "Only use this filter if the user's question explicitly mentions it; "
    "never invent a value."
)
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_SLUG_LENGTH = 40

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
}


def _resolve_python_type(type_str: str, field_name: str) -> type:
    """Map a UI type string to a Python type.

    Falls back to ``str`` and emits a WARNING for unknown type strings.

    Args:
        type_str: Raw type string from ``metadata_definition.fields[*].type``.
        field_name: Field name — included in the WARNING for diagnostics.

    Returns:
        The corresponding Python type, always one of ``str``, ``int``,
        ``float``, or ``bool``.
    """
    resolved = _TYPE_MAP.get((type_str or "").strip().lower())
    if resolved is None:
        logger.warning(
            "retriever_tool_builder: unknown type %r for field %r — falling back to str",
            type_str,
            field_name,
        )
        return str
    return resolved


def _is_valid_identifier(name: str) -> bool:
    """Return True if *name* is a valid Python identifier that does not shadow
    reserved words or the mandatory ``query`` field.
    """
    if not name or not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    if name == "query":
        return False
    return True


def _make_slug(name: str) -> str:
    """Convert *name* to a lowercase ASCII slug suitable for tool names.

    Rules:
    - NFKC-normalise, lowercase.
    - Replace any character that is not ``[a-z0-9_]`` with ``_``.
    - Collapse consecutive underscores.
    - Strip leading/trailing underscores.
    - Truncate to ``_MAX_SLUG_LENGTH`` characters.

    Returns an empty string when the input is blank or contains no
    alphanumeric characters.
    """
    normalised = unicodedata.normalize("NFKC", name or "").lower()
    slugified = re.sub(r"[^a-z0-9_]", "_", normalised)
    collapsed = re.sub(r"_+", "_", slugified).strip("_")
    return collapsed[:_MAX_SLUG_LENGTH]


def _sanitize_field_description(raw: str) -> str:
    """Sanitize an admin-supplied field description for safe LLM embedding.

    Applies ``sanitize_metadata_value`` with the wider max_len of 200
    characters to accommodate richer field descriptions.
    """
    return sanitize_metadata_value(raw, max_len=200)


def _build_literal_type(
    base_type: type,
    raw_values: list[str],
    field_name: str,
) -> tuple[type, list[str]]:
    """Build a ``Literal[...]`` type from sanitised string values.

    Only string fields can produce ``Literal`` types — numeric/bool fields
    fall through to the free-type path regardless of distinct_values length.

    Args:
        base_type: Python type resolved from the field's type string.
        raw_values: Raw distinct values from the cache service.
        field_name: Used in WARNING messages.

    Returns:
        ``(final_type, sanitised_values_used)`` where ``sanitised_values_used``
        is the deduplicated list of sanitised values (empty when no Literal
        was built).
    """
    if base_type is not str or not raw_values or len(raw_values) > MAX_ENUM_VALUES:
        return base_type, []

    sanitised: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        clean = sanitize_metadata_value(raw)
        if clean and clean not in seen:
            seen.add(clean)
            sanitised.append(clean)

    if not sanitised:
        logger.warning(
            "retriever_tool_builder: all distinct values for field %r were empty "
            "after sanitization — using free str type",
            field_name,
        )
        return str, []

    literal_type = Literal[tuple(sanitised)]  # type: ignore[valid-type]
    return literal_type, sanitised


def _build_field_description(
    raw_description: str,
    field_type_label: str,
    distinct_values: list[str],
    sanitised_literals: list[str],
) -> str:
    """Compose the Field description for an optional metadata filter parameter.

    Structure:
    1. Sanitised admin description (if non-empty).
    2. Type hint in parentheses.
    3. If free-type (no Literal): example values (up to MAX_EXAMPLE_VALUES).
    4. Filter-usage policy.

    Args:
        raw_description: Admin-supplied description from metadata_definition.
        field_type_label: Human-readable type label, e.g. ``"str"`` or ``"int"``.
        distinct_values: Raw distinct values from the cache (may be empty).
        sanitised_literals: Non-empty only when a Literal was built; used to
            suppress the example-values block (the Literal already encodes them).

    Returns:
        Composed description string.
    """
    parts: list[str] = []

    cleaned_desc = _sanitize_field_description(raw_description or "")
    if cleaned_desc:
        parts.append(cleaned_desc)

    parts.append(f"Type: {field_type_label}.")

    # Only show example values when not already encoded in a Literal
    if not sanitised_literals and distinct_values:
        examples: list[str] = []
        seen_ex: set[str] = set()
        for raw in distinct_values:
            clean = sanitize_metadata_value(raw)
            if clean and clean not in seen_ex:
                seen_ex.add(clean)
                examples.append(clean)
            if len(examples) >= MAX_EXAMPLE_VALUES:
                break
        if examples:
            parts.append(f"Example values: {', '.join(examples)}.")

    parts.append(_FILTER_USAGE_POLICY)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_retriever_args_schema(
    silo: Any,
    distinct_values: dict[str, list[str]],
) -> type[BaseModel]:
    """Build a Pydantic model for the retrieval tool's call arguments.

    The model always includes a mandatory ``query`` field.  One optional field
    is added for each entry in ``silo.metadata_definition.fields``.

    Fields whose names are not valid Python identifiers, are Python keywords,
    or collide with ``query`` are silently discarded with a WARNING.

    Only simple types (str, int, float, bool, Literal[str, ...]) are used to
    ensure compatibility with all LLM providers' function-calling schemas
    (OpenAI, Anthropic, Mistral, Azure, Google).

    Args:
        silo: ORM Silo instance.  ``silo.metadata_definition`` may be None.
            Must be attached to a live Session (reads the lazy
            ``metadata_definition`` relationship) — call at tool-construction
            time, never from inside a tool coroutine with a detached instance.
        distinct_values: Mapping ``{field_name: [raw_value, ...]}`` as
            returned by :func:`collect_distinct_values`.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass whose
        ``model_json_schema()`` is safe for all provider bind_tools calls.
    """
    field_definitions: dict[str, Any] = {
        "query": (
            str,
            Field(
                description=(
                    "Semantic search query over the document content. "
                    "Express what you are looking for in natural language; "
                    "the system will retrieve the most relevant passages."
                )
            ),
        )
    }

    metadata_def = getattr(silo, "metadata_definition", None)
    raw_fields: list[dict[str, Any]] = []
    if metadata_def is not None:
        raw_fields = metadata_def.fields or []

    for field_spec in raw_fields:
        if not isinstance(field_spec, dict):
            continue

        field_name: str = field_spec.get("name", "")
        if not _is_valid_identifier(field_name):
            logger.warning(
                "retriever_tool_builder: field name %r is not a valid Python "
                "identifier or collides with 'query' — skipping",
                field_name,
            )
            continue

        raw_type_str: str = field_spec.get("type", "str")
        base_type = _resolve_python_type(raw_type_str, field_name)
        # Canonical resolved name, never the raw editor-supplied string: the raw
        # type string is freeform text and must not reach the LLM prompt verbatim.
        type_label = base_type.__name__

        field_raw_values: list[str] = distinct_values.get(field_name, [])

        literal_type, sanitised_literals = _build_literal_type(
            base_type, field_raw_values, field_name
        )

        description = _build_field_description(
            raw_description=field_spec.get("description", ""),
            field_type_label=type_label,
            distinct_values=field_raw_values,
            sanitised_literals=sanitised_literals,
        )

        field_definitions[field_name] = (
            Optional[literal_type],
            Field(default=None, description=description),
        )

    model: type[BaseModel] = create_model(
        f"RetrieverArgs_{getattr(silo, 'silo_id', 'unknown')}",
        **field_definitions,
    )
    return model


def build_retriever_description(
    silo: Any,
    distinct_values: dict[str, list[str]],
) -> str:
    """Build the natural-language description for a silo's retrieval tool.

    The description is composed of:
    1. A purpose statement that varies by silo type (REPO / DOMAIN / other).
    2. A "Filterable metadata fields" section when the silo has a
       ``metadata_definition`` with at least one field.
    3. A usage policy (when to filter, and the zero-results retry behaviour).

    Total length is capped at :data:`_MAX_DESCRIPTION_LENGTH` characters.
    When truncation is needed the metadata-fields section is trimmed first,
    then the purpose blurb, and a trailing ellipsis is appended.

    Args:
        silo: ORM Silo instance. Must be attached to a live Session — this
            function reads the lazy relationships ``metadata_definition`` and
            ``domain``. Callers must invoke it at tool-construction time, never
            from inside a tool coroutine with a detached instance.
        distinct_values: Mapping ``{field_name: [raw_value, ...]}`` as
            returned by :func:`collect_distinct_values`.

    Returns:
        A description string safe to embed in an LLM tool definition.
    """
    silo_type_raw: str = str(getattr(silo, "silo_type", "") or "")
    silo_type_upper = silo_type_raw.upper()

    # --- Purpose blurb (varies by silo type) ---
    if silo_type_upper == "REPO":
        purpose = (
            "Search for relevant documents in the document repository. "
            "Use this tool to find specific files, passages, or structured data."
        )
    elif silo_type_upper == "DOMAIN":
        domain = getattr(silo, "domain", None)
        domain_desc = ""
        if domain is not None:
            raw_domain_desc = getattr(domain, "description", "") or ""
            domain_desc = sanitize_metadata_value(raw_domain_desc, max_len=200)
        if domain_desc:
            purpose = (
                f"Search for information from a crawled web site. "
                f"Site description: {domain_desc}"
            )
        else:
            purpose = "Search for information from a crawled web site."
    else:
        silo_desc = sanitize_metadata_value(
            getattr(silo, "description", "") or "", max_len=200
        )
        if silo_desc:
            purpose = f"Search for documents and information about {silo_desc}."
        else:
            purpose = "Search for relevant documents and information."

    # --- Filterable fields section ---
    metadata_def = getattr(silo, "metadata_definition", None)
    raw_fields: list[dict[str, Any]] = []
    if metadata_def is not None:
        raw_fields = metadata_def.fields or []

    fields_lines: list[str] = []
    for field_spec in raw_fields:
        if not isinstance(field_spec, dict):
            continue
        field_name = field_spec.get("name", "")
        if not field_name or not _is_valid_identifier(field_name):
            continue

        # Canonical resolved type name — the raw editor string never reaches the prompt.
        field_type = _resolve_python_type(field_spec.get("type", "str"), field_name).__name__
        field_desc_raw = field_spec.get("description", "") or ""
        # max_len=120 here (vs 200 in the args schema): the tool description has a
        # 2000-char global budget shared across all fields, the schema does not.
        field_desc = sanitize_metadata_value(field_desc_raw, max_len=120)

        # Build example values line (at least one real value required for AC-4)
        field_raw_values = distinct_values.get(field_name, [])
        examples: list[str] = []
        seen_ex: set[str] = set()
        for raw in field_raw_values:
            clean = sanitize_metadata_value(raw)
            if clean and clean not in seen_ex:
                seen_ex.add(clean)
                examples.append(clean)
            if len(examples) >= MAX_EXAMPLE_VALUES:
                break

        line = f"- {field_name} ({field_type}): {field_desc}."
        if examples:
            line += f" Example values: {', '.join(examples)}."

        fields_lines.append(line)

    # --- Usage policy ---
    usage_policy = (
        "When to filter: apply a metadata filter only when the user's question "
        "explicitly mentions a value for that field. "
        "If a filtered search returns zero results, retry without filters and "
        "show the available field values to help the user refine their query."
    )

    # --- Assemble and truncate ---
    sections: list[str] = [purpose]

    if fields_lines:
        fields_block = "Filterable metadata fields:\n" + "\n".join(fields_lines)
        sections.append(fields_block)

    sections.append(usage_policy)

    full_text = "\n\n".join(sections)
    if len(full_text) <= _MAX_DESCRIPTION_LENGTH:
        return full_text

    # Truncate: shorten the fields block first, then the purpose blurb
    budget = _MAX_DESCRIPTION_LENGTH - len(usage_policy) - 4  # "\n\n" separators

    if fields_lines:
        truncated_fields: list[str] = []
        used = len("Filterable metadata fields:\n")
        for line in fields_lines:
            candidate = used + len(line) + 1  # +1 for newline
            if candidate > budget - len(purpose) - 4:
                truncated_fields.append("...")
                break
            truncated_fields.append(line)
            used = candidate
        fields_block = "Filterable metadata fields:\n" + "\n".join(truncated_fields)
        result = "\n\n".join([purpose, fields_block, usage_policy])
    else:
        result = "\n\n".join([purpose, usage_policy])

    return result[:_MAX_DESCRIPTION_LENGTH]


def build_retriever_tool_name(silo: Any) -> str:
    """Build a stable, unique tool name for the silo's retrieval tool.

    Pattern: ``search_{slug}_{silo_id}``

    The slug is derived from ``silo.name``: lowercase, non-alphanumeric
    characters replaced with ``_``, consecutive underscores collapsed, leading/
    trailing underscores trimmed, truncated to ``_MAX_SLUG_LENGTH`` (40) chars.

    If the slug is empty (blank name, emoji-only, etc.) the name falls back to
    ``search_silo_{silo_id}`` to guarantee a non-empty, unique identifier.

    Args:
        silo: ORM Silo instance with ``silo_id`` and ``name`` attributes.

    Returns:
        A non-empty tool name string unique per silo_id.
    """
    silo_id = getattr(silo, "silo_id", "unknown")
    raw_name: str = getattr(silo, "name", "") or ""
    slug = _make_slug(raw_name)

    if not slug:
        return f"search_silo_{silo_id}"
    return f"search_{slug}_{silo_id}"


def collect_distinct_values(silo: Any, db: Any) -> dict[str, list[str]]:
    """Collect distinct metadata field values for all fields in a silo.

    This is a convenience helper for step_006 (the wiring layer).  It iterates
    ``silo.metadata_definition.fields`` and calls
    :meth:`~services.metadata_values_cache_service.MetadataValuesCacheService.get_distinct_values`
    for each field.  Errors on individual fields are caught and logged; the
    corresponding key is simply absent from the result dict.

    Args:
        silo: ORM Silo instance.
        db: SQLAlchemy Session forwarded to the cache service.

    Returns:
        ``{field_name: [value, ...]}`` mapping.  Fields for which the cache
        service raised are absent (not present with an empty list).
    """
    # Deferred import to mirror the pattern used by MetadataValuesCacheService
    # and avoid circular dependency risks at module load time.
    from services.metadata_values_cache_service import MetadataValuesCacheService  # noqa: PLC0415

    metadata_def = getattr(silo, "metadata_definition", None)
    if metadata_def is None:
        return {}

    raw_fields: list[dict[str, Any]] = metadata_def.fields or []
    result: dict[str, list[str]] = {}

    for field_spec in raw_fields:
        if not isinstance(field_spec, dict):
            continue
        field_name = field_spec.get("name", "")
        if not field_name:
            continue

        try:
            values = MetadataValuesCacheService.get_distinct_values(
                silo_id=silo.silo_id,
                field=field_name,
                db=db,
            )
            result[field_name] = values
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "retriever_tool_builder.collect_distinct_values: "
                "error fetching values for silo=%s field=%r: %s — key omitted",
                getattr(silo, "silo_id", "?"),
                field_name,
                exc,
            )

    return result
