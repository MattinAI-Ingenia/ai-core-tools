from typing import Optional, List, Dict, Any
import functools
import math
import os
import re
import uuid
from models.media import Media
from models.silo import Silo
from models.resource import Resource
from db.database import SessionLocal
from db.database import db as db_obj
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from utils.logger import get_logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.vector_store_factory import VectorStoreFactory
from tools.vector_stores.vector_store_interface import VectorStoreInterface
from models.indexing_metric import IndexingMetric
from models.silo import SiloType
from services.output_parser_service import OutputParserService
from langchain_core.vectorstores.base import VectorStoreRetriever
from utils.error_handlers import (
    handle_database_errors, NotFoundError, ValidationError,
    validate_required_fields
)
from utils.vector_db_immutability import assert_vector_db_type_immutable, assert_embedding_service_immutable
from schemas.silo_schemas import SiloListItemSchema, SiloDetailSchema, CreateUpdateSiloSchema
from repositories.silo_repository import SiloRepository
from services.folder_service import FolderService

REPO_BASE_FOLDER = os.path.abspath(os.getenv("REPO_BASE_FOLDER"))
COLLECTION_PREFIX = 'silo_'
DEFAULT_SEARCH_LIMIT = 100
MAX_SEARCH_LIMIT = 200
LIGHTRAG_VECTOR_DB_TYPES = ('PGVECTOR', 'QDRANT')

# Per-chunk marker stamping each indexing run; lets reindex delete only stale chunks.
INDEX_BATCH_FIELD = 'index_batch'

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LightRAG 2026.05 role-specific model recommendations
#
# Per-role minimum specs based on the LightRAG release notes:
#   EXTRACT  — entity/relationship extraction. Mid-tier, non-reasoning.
#   QUERY    — final answer generation. Large, optionally reasoning.
#   KEYWORDS — query keyword extraction. Small, fast. Only context matters.
#   VLM      — vision-language for images/tables. MUST be multimodal.
# ---------------------------------------------------------------------------

_ROLE_MIN_SPECS = {
    'extract':  {'params_b': 30, 'context_kb': 32},
    'query':    {'params_b': 32, 'context_kb': 32},
    'keywords': {'params_b': None, 'context_kb': 8},
    'vlm':      {'params_b': None, 'context_kb': None},  # validated via vision flag
}

# Known model specs. Tuple is (context_kb, params_b, supports_vision).
# Numbers are conservative public estimates — when uncertain, the value
# is biased toward the lower bound so the warning errs on the safe side.
_MODEL_SPECS = {
    # OpenAI
    'gpt-4o':            (128, 200, True),
    'gpt-4o-mini':       (128, 15, True),
    'gpt-4-turbo':       (128, 175, True),
    'gpt-4':             (8,   175, False),
    'gpt-3.5-turbo':     (16,  175, False),
    # Anthropic
    'claude-opus-4-7':   (200, 175, True),
    'claude-sonnet-4-6': (200, 100, True),
    'claude-opus':       (200, 175, True),
    'claude-3.5-sonnet': (200, 100, True),
    'claude-3-haiku':    (200, 30,  True),
    # Mistral
    'mistral-large':     (128, 123, False),
    'mistral-medium':    (32,  60,  False),
    'mistral-small':     (32,  10,  False),
    'pixtral':           (128, 12,  True),
    # Google
    'gemini-2.0-flash':  (1000, 1000, True),
    'gemini-1.5-pro':    (1000, 1000, True),
    'gemini-1.5-flash':  (1000, 1000, True),
}

# ---------------------------------------------------------------------------
# Pricing catalog — USD per 1 million tokens (public list prices, May 2026).
# Tuples are (input_usd_per_1m, output_usd_per_1m).
# Use the same fuzzy prefix/substring matching as _MODEL_SPECS.
# ---------------------------------------------------------------------------

def _ensure_pricing_catalog(db) -> None:
    """Populate the pricing_catalog table from providers if empty."""
    from models.pricing_catalog import PricingCatalog
    if db.query(PricingCatalog).first() is not None:
        return
    from services.pricing_service import PricingService
    PricingService.update_pricing_catalog(db)


def _lookup_model_specs(model_name: str):
    """Return (context_kb, params_b, supports_vision) or None if unknown."""
    if not model_name:
        return None
    lower = model_name.lower()
    for known_id, specs in _MODEL_SPECS.items():
        if known_id in lower or lower in known_id:
            return specs
    return None


def model_supports_vision(model_name: str) -> bool:
    """True only when the model is known to handle image inputs."""
    specs = _lookup_model_specs(model_name)
    return bool(specs and specs[2])


def _validate_model_for_role(role: str, model_name: str) -> Optional[str]:
    """Return a non-blocking warning for ``role``/``model_name`` or None.

    ``role`` must be one of ``extract|query|keywords|vlm``. VLM is validated
    elsewhere (it's a blocking error, not a warning) — see
    :func:`SiloService._validate_vlm_service`.
    """
    if not model_name or role not in _ROLE_MIN_SPECS:
        return None

    min_spec = _ROLE_MIN_SPECS[role]
    specs = _lookup_model_specs(model_name)

    if specs is None:
        # Unknown model — no warning (closed models lack public specs, can't assume size).
        return None

    context_kb, params_b, _ = specs
    issues = []
    if min_spec['context_kb'] is not None and context_kb < min_spec['context_kb']:
        issues.append(f"context {context_kb}K < {min_spec['context_kb']}K")
    if min_spec['params_b'] is not None and params_b < min_spec['params_b']:
        issues.append(f"params ~{params_b}B < {min_spec['params_b']}B")
    # Keywords: warn if model is unnecessarily large (>30B)
    if role == 'keywords' and params_b > 30:
        issues.append(f"params ~{params_b}B is overkill for keyword extraction")

    if not issues:
        return None
    return f"{role.upper()}: '{model_name}' {', '.join(issues)}"


def _validate_model_for_lightrag(model_name: str) -> Optional[str]:
    """Backward-compatible single-model validator (treats input as EXTRACT)."""
    return _validate_model_for_role('extract', model_name)

def _resolve_vector_db_type(silo: Optional[Silo] = None, override: Optional[str] = None) -> str:
    """Determine the vector DB type for a silo, falling back to PGVECTOR."""

    if override:
        return override.upper()

    if silo and getattr(silo, 'vector_db_type', None):
        return silo.vector_db_type.upper()

    return 'PGVECTOR'


def _get_vector_store(silo: Optional[Silo] = None, vector_db_type: Optional[str] = None) -> VectorStoreInterface:
    """Return the vector store implementation bound to the silo's backend."""

    resolved_type = _resolve_vector_db_type(silo, vector_db_type)
    if resolved_type == 'LIGHTRAG' and silo is not None:
        # EXTRACT is the role that drives indexing; prefer the new column
        # and fall back to the legacy ``indexing_service`` for old silos.
        extract_service = getattr(silo, 'extract_service', None) or silo.indexing_service
        return VectorStoreFactory.get_vector_store(
            db_obj, resolved_type,
            ai_service=extract_service,
            embedding_service=silo.embedding_service,
            keywords_service=getattr(silo, 'keywords_service', None),
            vlm_service=getattr(silo, 'vlm_service', None),
            lightrag_vector_db_type=getattr(silo, 'lightrag_vector_db_type', None) or 'QDRANT',
            lightrag_chunk_token_size=getattr(silo, 'lightrag_chunk_token_size', None),
            lightrag_chunk_overlap_token_size=getattr(silo, 'lightrag_chunk_overlap_token_size', None),
            lightrag_chunk_strategy=getattr(silo, 'lightrag_chunk_strategy', None),
            lightrag_language=getattr(silo, 'lightrag_language', None),
            lightrag_entity_extract_max_gleaning=getattr(silo, 'lightrag_entity_extract_max_gleaning', None),
            lightrag_max_source_ids_per_entity=getattr(silo, 'lightrag_max_source_ids_per_entity', None),
            lightrag_max_source_ids_per_relation=getattr(silo, 'lightrag_max_source_ids_per_relation', None),
            lightrag_entity_types=getattr(silo, 'lightrag_entity_types', None),
        )
    return VectorStoreFactory.get_vector_store(db_obj, resolved_type)


# Patterns produced by PDF loaders when they encounter fonts with non-standard
# encodings (e.g. Type3 fonts where codepoint 64 maps to a drawn glyph):
#   PyPDFLoader  → outputs raw decimal codepoints:  "64/64/64/64/..."
#   PyMuPDFLoader → decodes codepoint 64 as ASCII @: "@@@@@@@@@..."
# Both represent the same underlying artifact and must be stripped.
_GARBAGE_PATTERNS = [
    re.compile(r'(\d+/){8,}'),  # raw decimal codepoint sequences
    re.compile(r'@{5,}'),       # decoded Type3 font artifacts
]

# Detects real words: 3+ consecutive Unicode letters (isalpha-equivalent via \w without digits/_)
# Used to distinguish OCR artifact lines from lines with actual text content.
_WORD_RE = re.compile(r'[^\W\d_]{3,}', re.UNICODE)

# Matches runs of single Unicode letters each separated by exactly one space:
# e.g. "V I T O R I A" or "p a s a d o". Used to detect and collapse
# spaced-out OCR output back into normal words.
_SPACED_WORD_RE = re.compile(r'[^\W\d_](?: [^\W\d_])+', re.UNICODE)


@functools.lru_cache(maxsize=1)
def _token_encoder():
    """tiktoken encoder matching LightRAG's chunker (TiktokenTokenizer('gpt-4o-mini') → o200k_base)."""
    import tiktoken
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str) -> int:
    """Real token count via the same tiktoken encoding LightRAG uses."""
    if not text:
        return 0
    try:
        return len(_token_encoder().encode(text))
    except Exception:
        return len(text) // 4  # ponytail: fallback if tiktoken unavailable


# Providers whose inference runs on our own hardware: no per-token price exists,
# so they contribute 0 to a cost estimate (see estimate_indexing_cost).
_SELF_HOSTED_PROVIDERS = frozenset({'Custom', 'Ollama'})


def _chunks_from_tokens(doc_tokens: int, chunk_token_size: int, overlap_token_size: int) -> int:
    """Token-window chunk count for a document of *doc_tokens* tokens."""
    if doc_tokens <= 0:
        return 0
    if overlap_token_size >= chunk_token_size or overlap_token_size == 0:
        return max(1, math.ceil(doc_tokens / chunk_token_size))
    stride = chunk_token_size - overlap_token_size
    if doc_tokens <= chunk_token_size:
        return 1
    return 1 + math.ceil((doc_tokens - chunk_token_size) / stride)


def _collapse_spaced_letters(line: str) -> str:
    """Collapse spaced-out single-letter OCR output into normal words.

    When > 60 % of the whitespace-delimited tokens on a line are single
    Unicode letters, the line is treated as spaced-out text (e.g. from
    low-resolution OCR) and consecutive single-letter runs are joined::

        "V I T O R I A .  E l  p a s a d o" → "VITORIA . El pasado"

    Lines that do not meet the threshold are returned unchanged.
    """
    tokens = line.split()
    if len(tokens) < 3:
        return line
    single_alpha = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if single_alpha / len(tokens) < 0.6:
        return line
    collapsed = _SPACED_WORD_RE.sub(lambda m: m.group(0).replace(' ', ''), line)
    # Normalize multiple spaces left after joining (e.g. between sections)
    collapsed = re.sub(r' {2,}', ' ', collapsed)
    return collapsed.strip()


def _clean_chunk_text(text: str) -> str:
    """Remove known PDF extraction artifacts from *text*.

    Two complementary strategies are applied:

    1. **Span-level:** Replace long runs of repeated glyph sequences in a single
       span (e.g. ``@@@@@`` or ``64/64/64/...``) with a space.
    2. **Line-level:** Drop entire lines that consist almost entirely of
       non-alphabetic characters and are very short (≤ 6 chars, < 2 alpha).
       This catches multi-line Type3 font artifacts such as ``@@h?``, ``@@g``,
       ``??``, ``?``, ``@@`` — which are Unicode codepoints decoded from glyphs
       that have no real text mapping (e.g. cp64→@, cp104→h, cp63→?).
    """
    # 1. Span-level: strip long repetitive sequences
    for pattern in _GARBAGE_PATTERNS:
        text = pattern.sub(' ', text)

    # 2. Per-line: collapse spaced-out single-letter OCR text BEFORE filtering
    #    so that "V I T O R I A" → "VITORIA" survives Rule B below.
    lines = text.split('\n')
    lines = [_collapse_spaced_letters(line) for line in lines]

    # 3. Line-level: drop lines that are clearly artifact/garbage
    #    Rule A — very short lines with minimal alpha (@@h?, @@g, ??, @@, …)
    #    Rule B — short-medium lines with no real word ≥ 3 letters; catches OCR
    #             image noise such as "r . * * * ! ? *", "^ i * * : * ;",
    #             "-'Jl", "i i ¥", etc.
    clean_lines = []
    for line in lines:
        s = line.strip()
        n = len(s)
        alpha = sum(c.isalpha() for c in s)
        if n <= 8 and alpha < 3:          # Rule A
            continue
        if n < 50 and not _WORD_RE.search(s):  # Rule B
            continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # Collapse whitespace runs introduced by replacements
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _is_meaningful_chunk(text: str) -> bool:
    """Return True if *text* (already cleaned) contains real textual content.

    Rejects chunks that are too short or still dominated by non-alphabetic
    characters after ``_clean_chunk_text`` has been applied.
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    # Generic: fewer than 40 % of characters are letters or spaces → garbage
    alpha_space = sum(1 for c in stripped if c.isalpha() or c.isspace())
    if alpha_space / len(stripped) < 0.40:
        return False
    return True


def _scale_k_per_100_chunks(k_per_100: int, silo: Any) -> int:
    """Scale ``k_per_100`` (documents per 100 indexed chunks) by the silo's chunk count.

    Opens its own short-lived session — ``resolve_search_params`` runs via
    ``asyncio.to_thread``, so a blocking count query here is safe. No upper clamp:
    a large knowledge base is meant to retrieve proportionally more.
    """
    session = SessionLocal()
    try:
        chunk_count = SiloService.count_docs_in_silo(silo.silo_id, session)
    except Exception as exc:
        logger.warning("resolve_search_params: could not count chunks for silo %s: %s", silo.silo_id, exc)
        return k_per_100
    finally:
        session.close()

    if chunk_count <= 0:
        return k_per_100
    return max(1, math.ceil(k_per_100 * chunk_count / 100))


def resolve_search_params(agent: Any, caller_search_params: Optional[dict]) -> tuple[dict, dict]:
    """Materialize RAG precedence: caller > agent > system.

    Returns ``(search_params, pinned_filter)`` where:
    - ``search_params``: tuning-only dict (no 'filter' key) — k, search_type,
      score_threshold, fetch_k, lambda_mult.
    - ``pinned_filter``: backend-ready ``{field: {op: value}}`` dict — the AND
      combination of caller flat filter and ``agent.rag_fixed_filters``. The
      agent's fixed filters WIN on (field, op) conflict: they are an
      admin-configured scoping floor a caller must not be able to loosen
      (security). Tuning params (k/search_type/...) keep caller > agent.

    Never raises: filter-building errors are caught and logged as WARNING.

    Args:
        agent: Agent ORM instance (or SimpleNamespace in tests).  Must expose
            ``rag_k``, ``rag_search_type``, ``rag_score_threshold``,
            ``rag_fixed_filters``, ``rag_max_retrieval_calls`` and
            optionally ``.silo`` (for metadata_definition + vector_db_type).
        caller_search_params: Optional search-param dict supplied by the call
            site (e.g. the public API chat endpoint).
    """
    caller: dict = caller_search_params or {}
    silo = getattr(agent, "silo", None)

    # ---- Tuning params (caller > agent > server_default) ----
    resolved: dict = {}

    # k: caller explicit value wins; otherwise use agent column (server_default 30),
    # scaled against the silo's chunk count when rag_k_mode='per_100_chunks'.
    if "k" in caller:
        resolved["k"] = caller["k"]
    else:
        agent_k = getattr(agent, "rag_k", None) or 30
        k_mode = getattr(agent, "rag_k_mode", None) or "fixed"
        resolved["k"] = (
            _scale_k_per_100_chunks(agent_k, silo) if k_mode == "per_100_chunks" and silo is not None else agent_k
        )

    # search_type: caller wins; fall back to agent column (server_default 'similarity')
    resolved["search_type"] = (
        caller.get("search_type") or getattr(agent, "rag_search_type", None) or "similarity"
    )

    # score_threshold: caller wins (explicit None clears). Only concrete values are stored,
    # so absent key == no threshold; never forward None to as_retriever.
    if "score_threshold" in caller:
        if caller["score_threshold"] is not None:
            resolved["score_threshold"] = caller["score_threshold"]
    else:
        agent_threshold = getattr(agent, "rag_score_threshold", None)
        if agent_threshold is not None:
            resolved["score_threshold"] = agent_threshold

    # fetch_k / lambda_mult: caller pass-through only
    if "fetch_k" in caller:
        resolved["fetch_k"] = caller["fetch_k"]
    if "lambda_mult" in caller:
        resolved["lambda_mult"] = caller["lambda_mult"]

    # Threshold search without a threshold silently degrades to plain similarity — normalize
    # and warn instead of a silent no-op (legacy agents, or a caller that cleared the value).
    if resolved["search_type"] == "similarity_score_threshold" and "score_threshold" not in resolved:
        logger.warning(
            "resolve_search_params: 'similarity_score_threshold' requested without a "
            "score_threshold; falling back to plain similarity search"
        )
        resolved["search_type"] = "similarity"

    # ---- Pinned filters ----
    if silo is None:
        return resolved, {}

    metadata_definition = getattr(silo, "metadata_definition", None)

    def _build_backend_filter(clauses_input: list) -> dict:
        """Build backend filter from a list of MetadataFilterClause-like dicts/instances."""
        from tools.vector_stores.metadata_filters import (  # local import avoids cycles
            MetadataFilterClause,
            convert_clause_types,
            to_backend_filter,
        )
        clauses = []
        for item in clauses_input:
            try:
                if isinstance(item, MetadataFilterClause):
                    clauses.append(item)
                else:
                    clauses.append(MetadataFilterClause(**item))
            except Exception as exc:
                logger.warning(
                    "resolve_search_params: could not build filter clause from %r: %s",
                    {k: v for k, v in item.items() if k != "value"} if isinstance(item, dict) else "?",
                    exc,
                )
        if not clauses:
            return {}
        typed = convert_clause_types(clauses, metadata_definition)
        return to_backend_filter(typed)

    # Caller flat filter: {field: value} → convert to $eq clauses
    caller_backend: dict = {}
    raw_caller_filter = caller.get("filter") or {}
    if raw_caller_filter:
        try:
            caller_clauses = [{"field": f, "op": "$eq", "value": v} for f, v in raw_caller_filter.items()]
            caller_backend = _build_backend_filter(caller_clauses)
        except Exception as exc:
            logger.warning("resolve_search_params: failed to build caller filter: %s", exc)

    # Agent rag_fixed_filters: list of {field, op, value}
    agent_backend: dict = {}
    raw_agent_filters: Optional[list] = getattr(agent, "rag_fixed_filters", None)
    if raw_agent_filters:
        try:
            agent_backend = _build_backend_filter(raw_agent_filters)
        except Exception as exc:
            logger.warning("resolve_search_params: failed to build agent fixed filters: %s", exc)

    # AND merge. Agent fixed filters passed FIRST so they win on (field, op)
    # conflict — an admin scoping floor the caller cannot loosen (security).
    # merge_filters_and is first-wins per (field, op). Caller filters on other
    # fields still apply (AND).
    if caller_backend or agent_backend:
        from tools.vector_stores.metadata_filters import merge_filters_and  # local import
        pinned_filter = merge_filters_and(agent_backend, caller_backend)
    else:
        pinned_filter = {}

    return resolved, pinned_filter


class SiloService:

    '''SILO CRUD Operations'''
    @staticmethod
    def get_silo(silo_id: int, db: Session) -> Optional[Silo]:
        """
        Retrieve a silo by its ID
        """
        return SiloRepository.get_by_id(silo_id, db)

    @staticmethod
    def _validate_vlm_service(service_id: int, db: Session) -> None:
        """Raise ``ValidationError`` if the AIService isn't a multimodal model.

        Determined heuristically from the model name (see ``_MODEL_SPECS``);
        unknown model names are treated as non-multimodal so the operator
        gets a clear failure rather than a silent runtime error later.
        """
        from repositories.ai_service_repository import AIServiceRepository

        ai_service = AIServiceRepository.get_by_id(db, service_id)
        if ai_service is None:
            raise ValidationError(f"VLM AI service {service_id} not found")

        model_name = (
            getattr(ai_service, 'description', None)
            or getattr(ai_service, 'model_name', None)
            or getattr(ai_service, 'name', None)
            or ''
        )
        if not model_supports_vision(model_name):
            raise ValidationError(
                f"VLM role requires a multimodal model — '{model_name}' is not known to support vision. "
                "Select a vision-capable model (e.g. gpt-4o, claude-3.5-sonnet, gemini-1.5-pro) "
                "or leave the VLM service empty if your documents have no images."
            )
    
    @staticmethod
    def get_silo_retriever(silo_id: int, search_params=None, **kwargs) -> Optional[VectorStoreRetriever]:
        """
        Get retriever for a silo with its corresponding embedding service
        
        Args:
            silo_id: ID of the silo
            search_params: Optional search parameters for filtering
            
        Returns:
            VectorStoreRetriever if silo exists and has embedding service, None otherwise
            
        Raises:
            NotFoundError: If silo doesn't exist
            ValidationError: If silo has no embedding service
        """
        if not isinstance(silo_id, int) or silo_id <= 0:
            raise ValidationError(f"Invalid silo_id: {silo_id}")
        
        # For retriever operations, we need a fresh session since this might be called from other contexts
        session = SessionLocal()
        try:
            silo = SiloService.get_silo(silo_id, session)
            if not silo:
                raise NotFoundError(f"Silo with ID {silo_id} not found", "silo")

            if not silo.embedding_service:
                raise ValidationError(f"Silo {silo_id} has no embedding service configured")

            logger.debug(f"Getting retriever for silo {silo_id} with embedding service: {silo.embedding_service.name}")
            
            collection_name = COLLECTION_PREFIX + str(silo_id)

            # Known retriever parameters that should not be wrapped in 'filter'
            known_params = {'k', 'filter', 'score_threshold', 'fetch_k', 'lambda_mult', 'search_type', 'lightrag_query_mode'}

            # --- Layer 1: system defaults ---
            merged_search_kwargs: dict = {'k': 30}

            # --- Layer 2: per-call search_params (highest priority) ---
            if search_params:
                # Separate known params from filter fields
                filter_fields = {}
                direct_params = {}
                
                for key, value in search_params.items():
                    if key in known_params:
                        direct_params[key] = value
                    else:
                        # Any unknown key is treated as a filter field
                        filter_fields[key] = value
                
                # Update merged_search_kwargs with direct params
                merged_search_kwargs.update(direct_params)
                
                # If there are filter fields, wrap them in 'filter' key
                if filter_fields:
                    if 'filter' in merged_search_kwargs:
                        # Merge with existing filter
                        merged_search_kwargs['filter'].update(filter_fields)
                    else:
                        # Create new filter with these fields
                        merged_search_kwargs['filter'] = filter_fields
                
                logger.debug(f"Merged search_kwargs: {merged_search_kwargs}")
            
            # Extract search_type before passing search_kwargs — it must be a top-level
            # kwarg to `as_retriever`, not nested inside search_kwargs.
            retriever_search_type = merged_search_kwargs.pop("search_type", "similarity")

            # Use async engine with psycopg (not asyncpg) for async operations
            # psycopg supports async natively and handles multiple SQL statements properly
            return _get_vector_store(silo).get_retriever(
                collection_name,
                silo.embedding_service,
                merged_search_kwargs,
                search_type=retriever_search_type,
                use_async=True  # Use async psycopg engine for LangGraph compatibility
            )
        except Exception as e:
            logger.error(f"Failed to create retriever for silo {silo_id}: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
    
    @staticmethod
    def get_silos_by_app_id(app_id: int, db: Session) -> List[Silo]:
        """
        Retrieve all silos by app_id
        """
        return SiloRepository.get_by_app_id(app_id, db)
    
    @staticmethod
    @handle_database_errors("create_or_update_silo")
    def create_or_update_silo(silo_data: dict, silo_type: Optional[SiloType] = None, db: Session = None) -> Silo:
        """
        Create a new silo or update an existing one
        
        Args:
            silo_data: Dictionary containing silo data
            silo_type: Optional silo type to set
            db: Database session to use
            
        Returns:
            Created or updated Silo instance
            
        Raises:
            ValidationError: If required fields are missing or invalid
            DatabaseError: If database operation fails
        """
        logger.info(f"Received silo data: {silo_data}")
        
        # Convert ImmutableMultiDict to regular dict if needed
        if hasattr(silo_data, 'to_dict'):
            silo_data = silo_data.to_dict()
        else:
            silo_data = dict(silo_data)
        
        requested_vector_db_type = silo_data.get('vector_db_type')
        if requested_vector_db_type is not None:
            if not isinstance(requested_vector_db_type, str):
                raise ValidationError("vector_db_type must be a string")
            requested_vector_db_type = requested_vector_db_type.strip().upper()
            if not requested_vector_db_type:
                requested_vector_db_type = None
            elif requested_vector_db_type not in VectorStoreFactory.IMPLEMENTED_TYPES:
                supported = ', '.join(VectorStoreFactory.IMPLEMENTED_TYPES)
                raise ValidationError(
                    f"Unsupported vector_db_type '{requested_vector_db_type}'. Supported types: {supported}"
                )

        requested_lightrag_vector_db_type = silo_data.get('lightrag_vector_db_type')
        if requested_lightrag_vector_db_type is not None:
            if not isinstance(requested_lightrag_vector_db_type, str):
                raise ValidationError("lightrag_vector_db_type must be a string")
            requested_lightrag_vector_db_type = requested_lightrag_vector_db_type.strip().upper()
            if not requested_lightrag_vector_db_type:
                requested_lightrag_vector_db_type = None
            elif requested_lightrag_vector_db_type not in LIGHTRAG_VECTOR_DB_TYPES:
                supported = ', '.join(LIGHTRAG_VECTOR_DB_TYPES)
                raise ValidationError(
                    "Unsupported lightrag_vector_db_type "
                    f"'{requested_lightrag_vector_db_type}'. Supported types: {supported}"
                )

        # Validate required fields
        required_fields = ['name', 'app_id']
        validate_required_fields(silo_data, required_fields)
        
        # Validate field types
        field_types = {'app_id': int}
        if 'silo_id' in silo_data and silo_data['silo_id']:
            field_types['silo_id'] = int
        if 'embedding_service_id' in silo_data and silo_data['embedding_service_id']:
            field_types['embedding_service_id'] = int
        
        # Convert string values to int where needed
        for field in [
            'silo_id', 'app_id', 'embedding_service_id', 'indexing_service_id',
            'extract_service_id', 'keywords_service_id', 'vlm_service_id',
        ]:
            if field in silo_data and silo_data[field] and isinstance(silo_data[field], str):
                try:
                    silo_data[field] = int(silo_data[field])
                except ValueError:
                    raise ValidationError(f"Invalid integer value for {field}: {silo_data[field]}")
        
        silo_id = silo_data.get('silo_id')
        
        # Use provided session or create a new one
        session = db if db is not None else SessionLocal()
        should_close = db is None
        
        try:
            # Get existing silo or create new one
            if silo_id:
                silo = SiloService.get_silo(silo_id, session)
                if not silo:
                    raise NotFoundError(f"Silo with ID {silo_id} not found", "silo")
                logger.info(f"Updating existing silo {silo_id}")
                assert_vector_db_type_immutable(
                    silo.vector_db_type, requested_vector_db_type, "silo"
                )
                assert_embedding_service_immutable(
                    silo.embedding_service_id, silo_data.get('embedding_service_id'), "silo"
                )
            else:
                # Enforce per-app silo limit before creation (SaaS mode only)
                app_id = silo_data.get('app_id')
                if app_id:
                    from services.tier_enforcement_service import TierEnforcementService
                    TierEnforcementService.check_resource_limit(session, int(app_id), 'silos')

                silo = Silo()
                # Set default type to CUSTOM, but allow override from form data
                silo.silo_type = SiloType.CUSTOM.value
                logger.info("Creating new silo")

            silo.vector_db_type = _resolve_vector_db_type(silo, requested_vector_db_type)

            if silo.vector_db_type == 'LIGHTRAG':
                resolved_lightrag_vector_db_type = (
                    requested_lightrag_vector_db_type
                    or getattr(silo, 'lightrag_vector_db_type', None)
                    or 'QDRANT'
                )
                silo.lightrag_vector_db_type = resolved_lightrag_vector_db_type
            else:
                silo.lightrag_vector_db_type = None

            # LightRAG validation: require at least one extraction LLM (query
            # is the primary; extract is auto-filled from query in the UI but
            # both can be provided) and an embedding service.
            if silo.vector_db_type == 'LIGHTRAG' and not silo_id:
                has_extract_llm = (
                    silo_data.get('extract_service_id')
                    or silo_data.get('indexing_service_id')  # legacy
                )
                if not has_extract_llm:
                    raise ValidationError(
                        "LightRAG silos require at least an Extract AI service"
                    )
                if not silo_data.get('embedding_service_id'):
                    raise ValidationError("LightRAG silos require an embedding_service_id")

                # Block: VLM service, if provided, MUST be multimodal.
                vlm_service_id = silo_data.get('vlm_service_id')
                if vlm_service_id:
                    SiloService._validate_vlm_service(int(vlm_service_id), session)

            # extract/vlm/indexing are set on creation only — immutable after,
            # since changing them mid-way would mix entities extracted with
            # different models in the same graph. keywords_service_id is
            # query-time only (never touches indexed data) so it stays
            # editable on update too.
            if not silo_id:
                role_fields = (
                    'indexing_service_id',
                    'extract_service_id',
                    'vlm_service_id',
                )
                for field in role_fields:
                    if silo_data.get(field):
                        setattr(silo, field, silo_data[field])

            if silo_data.get('keywords_service_id'):
                silo.keywords_service_id = silo_data['keywords_service_id']

            # Compatibility shim: when the UI sent only the new
            # extract_service_id, mirror it into the legacy column so
            # downstream code that still reads indexing_service_id keeps
            # working until it is migrated.
            if silo.extract_service_id and not silo.indexing_service_id:
                silo.indexing_service_id = silo.extract_service_id
            elif silo.indexing_service_id and not silo.extract_service_id:
                silo.extract_service_id = silo.indexing_service_id

            # Set LightRAG config columns on creation
            if not silo_id:
                for field in ('lightrag_chunk_strategy', 'lightrag_chunk_token_size',
                              'lightrag_chunk_overlap_token_size', 'lightrag_language',
                              'lightrag_entity_extract_max_gleaning',
                              'lightrag_max_source_ids_per_entity',
                              'lightrag_max_source_ids_per_relation',
                              'lightrag_entity_types'):
                    if field in silo_data and silo_data[field] is not None:
                        setattr(silo, field, silo_data[field])

            # Set silo type from form data if provided
            if silo_data.get('type') and silo_data['type'].strip():
                silo.silo_type = silo_data['type'].strip()
            # Set silo type if provided via parameter (for backward compatibility)
            elif silo_type:
                silo.silo_type = silo_type.value
                if silo_type == SiloType.REPO:
                    silo.metadata_definition_id = 0

            # Set embedding service on creation only — immutable after creation
            if not silo_id and silo_data.get('embedding_service_id'):
                silo.embedding_service_id = silo_data['embedding_service_id']
            
            # Handle metadata definition (output parser) - explicitly handle None to clear it
            if 'output_parser_id' in silo_data:
                # If output_parser_id is explicitly provided (even if None), use it
                logger.info(f"Setting metadata_definition_id to: {silo_data['output_parser_id']}")
                silo.metadata_definition_id = silo_data['output_parser_id']
            elif 'metadata_definition_id' in silo_data:
                # Fallback to metadata_definition_id for backward compatibility
                logger.info(f"Setting metadata_definition_id from fallback to: {silo_data['metadata_definition_id']}")
                silo.metadata_definition_id = silo_data['metadata_definition_id']
            
            # Update silo attributes
            SiloService._update_silo(silo, silo_data)
            
            # Save to database
            session.add(silo)
            session.commit()
            
            logger.info(f"Successfully {'updated' if silo_id else 'created'} silo {silo.silo_id}")
            return silo
        finally:
            if should_close:
                session.close()
    
    @staticmethod
    def _update_silo(silo: Silo, data: dict):
        """
        Update silo attributes from input data
        
        Args:
            silo: Silo instance to update
            data: Dictionary containing update data
            
        Raises:
            ValidationError: If data validation fails
        """
        # Validate silo name
        name = data['name'].strip() if data['name'] else None
        if not name:
            raise ValidationError("Silo name cannot be empty")
        
        silo.name = name
        silo.description = data.get('description', '').strip() or None
        silo.status = data.get('status')
        silo.app_id = data['app_id']
        silo.fixed_metadata = bool(data.get('fixed_metadata', False))
        # Don't override metadata_definition_id here as it's handled above
        # silo.metadata_definition_id = data.get('metadata_definition_id') or None
            
    @staticmethod
    def delete_silo(silo_id: int, db: Session):
        """
        Delete a silo by its ID
        """
        silo = SiloRepository.get_by_id(silo_id, db)
        if silo:
            # Store metadata_definition_id before deleting (avoid DetachedInstanceError)
            metadata_definition_id = silo.metadata_definition_id
            
            SiloService.delete_collection(silo.silo_id, db)
            
            silo.embedding_service_id = None
            db.add(silo)
            db.commit()
            
            # Now delete the silo using repository
            SiloRepository.delete(silo_id, db)

            # Finally delete the output parser if it exists
            if metadata_definition_id:
                output_parser_service = OutputParserService()
                output_parser_service.delete_parser(db, metadata_definition_id)



    '''SILO and DATA Operations'''

    @staticmethod
    def check_silo_collection_exists(silo_id: int, db: Session) -> bool:
        collection_name = COLLECTION_PREFIX + str(silo_id)
        try:
            silo = SiloRepository.get_by_id(silo_id, db)
            if not silo:
                logger.warning("Silo %s not found while checking collection existence", silo_id)
                return False
            return _get_vector_store(silo).collection_exists(collection_name)
        except Exception as exc:
            logger.error(f"Error checking collection for silo {silo_id}: {exc}")
            return False
    
    @staticmethod
    def count_docs_in_silo(silo_id: int, db: Session) -> int:
        collection_name = COLLECTION_PREFIX + str(silo_id)
        try:
            silo = SiloRepository.get_by_id(silo_id, db)
            if not silo:
                logger.warning("Silo %s not found while counting documents", silo_id)
                return 0

            count = _get_vector_store(silo).count_documents(collection_name)
            if count > 0:
                return count

            if _resolve_vector_db_type(silo) == 'LIGHTRAG':
                # LightRAG environments may not expose PGDocStatusStorage
                # tables consistently. When that happens, fall back to the
                # number of successfully indexed source resources recorded in
                # indexing metrics so the silo UI reflects uploaded files.
                metric_count = (
                    db.query(func.count(func.distinct(IndexingMetric.resource_id)))
                    .filter(
                        IndexingMetric.silo_id == silo_id,
                        IndexingMetric.status == 'success',
                        IndexingMetric.resource_id.isnot(None),
                    )
                    .scalar()
                )
                if metric_count:
                    return int(metric_count)

            return 0
        except Exception as exc:
            logger.error(f"Error counting docs in silo {silo_id}: {exc}")
            return 0

    @staticmethod
    def count_docs_with_filter(
        silo_id: int,
        filter_metadata: Optional[dict] = None,
        db: Session = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
    ) -> int:
        """
        Count documents in a silo collection, optionally filtered by metadata
        and/or content length. Returns 0 gracefully if the silo or collection
        does not exist.
        """
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo or not SiloService.check_silo_collection_exists(silo_id, db):
            return 0

        collection_name = COLLECTION_PREFIX + str(silo_id)
        try:
            return _get_vector_store(silo).count_documents(
                collection_name,
                filter_metadata=filter_metadata,
                min_content_length=min_content_length,
                max_content_length=max_content_length,
            )
        except Exception as exc:
            logger.error("count_docs_with_filter error for silo %s: %s", silo_id, exc)
            return 0

    @staticmethod
    def _get_silo_for_indexing(silo_id: int, db: Session):
        """Helper method to get silo and validate it exists"""
        silo = SiloService.get_silo(silo_id, db)
        if not silo:
            logger.error(f"Silo con id {silo_id} no existe")
            raise ValueError(f"Silo with id {silo_id} does not exist")
        return silo

    @staticmethod
    def _create_documents_for_indexing(silo_id: int, contents: List[dict]) -> List[Document]:
        """Split each content item into chunks and attach metadata.

        File-upload callers pre-split via ``extract_documents_from_file`` before
        calling ``index_multiple_content``, so chunks already ≤ CHUNK_SIZE produce
        exactly one chunk per input item (idempotent).
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        documents: List[Document] = []
        for item in contents:
            # silo_id last — a caller-supplied key must not override the parameter.
            base_metadata = {**(item.get('metadata', {})), "silo_id": silo_id}
            chunks = splitter.split_text(item['content'])
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=dict(base_metadata)))
        return documents

    @staticmethod
    def index_single_content(silo_id: int, content: str, metadata: dict, db: Session):
        """Index single content in a silo"""
        SiloService.index_multiple_content(silo_id, [{'content': content, 'metadata': metadata}], db)

    @staticmethod
    def index_multiple_content(silo_id: int, documents: List[dict], db: Session):
        """Index multiple documents in a silo with the corresponding embedding service"""
        logger.info(f"Indexando documentos en silo {silo_id}")
        
        collection_name = COLLECTION_PREFIX + str(silo_id)
        
        # Get silo within this session to avoid detached instance
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            logger.error(f"Silo con id {silo_id} no existe")
            raise ValueError(f"Silo with id {silo_id} does not exist")
        
        # Get embedding service within the same session
        embedding_service = None
        if silo.embedding_service_id:
            embedding_service = SiloRepository.get_embedding_service_by_id(silo.embedding_service_id, db)
        
        logger.debug(f"Usando embedding service: {embedding_service.name if embedding_service else 'None'}")
        
        docs = SiloService._create_documents_for_indexing(silo_id, documents)
        _get_vector_store(silo).index_documents(
            collection_name,
            docs,
            embedding_service=embedding_service
        )
        logger.info(f"Documentos indexados correctamente en silo {silo_id}")
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService  # noqa: PLC0415
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after index_multiple_content for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def extract_documents_from_file(file_path: str, file_extension: str, base_metadata: dict = None, split: bool = True):
        """
        Extracts (and optionally splits) documents from a file, attaching base metadata.
        Args:
            file_path: Path to the file to extract from
            file_extension: File extension (e.g., '.pdf', '.docx', '.txt')
            base_metadata: Metadata dict to attach to each document
            split: When True (default, for PGVector/Qdrant), split into ~1000-char
                chunks. When False (LightRAG), return one Document per page/file so
                LightRAG applies its own token-based chunking (``lightrag_chunk_token_size``).
        Returns:
            List[Document]: List of Document objects
        """
        from langchain_core.documents import Document
        from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

        if base_metadata is None:
            base_metadata = {}

        # Determine file type and use appropriate loader
        if file_extension == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension in ('.txt', '.md'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")

        pages = loader.load()
        if split:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            docs = text_splitter.split_documents(pages)
        else:
            # LightRAG path: keep one Document per page/file so LightRAG can
            # chunk by its own configured token size instead of being fed
            # pre-split text (which would bypass lightrag_chunk_token_size).
            docs = pages

        # Clean known PDF extraction artifacts (e.g. @@@@, 64/64/64/...) from
        # each chunk's content, then discard chunks that are empty or still
        # dominated by non-textual characters after cleaning.
        for doc in docs:
            doc.page_content = _clean_chunk_text(doc.page_content)

        original_count = len(docs)
        docs = [doc for doc in docs if _is_meaningful_chunk(doc.page_content)]
        filtered = original_count - len(docs)
        if filtered:
            logger.debug(f"Filtered {filtered} garbage chunk(s) from {file_path}")

        for doc in docs:
            doc.metadata.update(base_metadata)
            # Only add page number if it exists (PDFs have page metadata, DOCX/TXT don't)
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1
        return docs

    @staticmethod
    def update_resource_metadata(resource: Resource, db_session: Session = None):
        """
        Update only the metadata of a resource in the vector database without re-indexing content.
        This is more efficient for operations like moving files between folders.

        Args:
            resource: Resource instance with updated folder information
            db_session: Optional database session to use (if None, creates new session)
        """
        session = db_session if db_session else SessionLocal()
        try:
            resource_with_relations = session.query(Resource).filter(
                Resource.resource_id == resource.resource_id
            ).first()

            if not resource_with_relations:
                logger.error(f"Resource {resource.resource_id} not found for metadata update")
                return

            silo = resource_with_relations.repository.silo
            collection_name = COLLECTION_PREFIX + str(silo.silo_id)

            file_extension = os.path.splitext(resource_with_relations.uri)[1].lower()
            base_metadata = {
                "repository_id": resource_with_relations.repository_id,
                "resource_id": resource_with_relations.resource_id,
                "silo_id": silo.silo_id,
                "name": resource_with_relations.uri,
                "file_type": file_extension,
            }

            if resource_with_relations.folder_id:
                folder_path = FolderService.get_folder_path(resource_with_relations.folder_id, session)
                base_metadata["folder_id"] = resource_with_relations.folder_id
                base_metadata["folder_path"] = folder_path
                base_metadata["ref"] = os.path.join(
                    str(resource_with_relations.repository_id), folder_path, resource_with_relations.uri
                )
            else:
                base_metadata["folder_id"] = None
                base_metadata["folder_path"] = ""
                base_metadata["ref"] = os.path.join(
                    str(resource_with_relations.repository_id), resource_with_relations.uri
                )

            updated = _get_vector_store(silo).update_documents_metadata(
                collection_name,
                filter_metadata={"resource_id": {"$eq": str(resource.resource_id)}},
                metadata_updates=base_metadata,
                replace=True,
            )

            logger.info(
                "Updated metadata for resource %s in collection %s (%s rows)",
                resource.resource_id, collection_name, updated,
            )

        except Exception as e:
            logger.error(f"Error updating resource metadata: {str(e)}")
            raise
        finally:
            if not db_session:
                session.close()

    @staticmethod
    def update_media_metadata(media: Media, db_session: Session = None):
        """
        Update only the metadata of media chunks in the vector database without re-indexing content.
        Updates folder information for all chunks belonging to this media.

        Args:
            media: Media instance with updated folder information
            db_session: Optional database session
        """
        session = db_session if db_session else SessionLocal()
        try:
            media_with_relations = session.query(Media).filter(
                Media.media_id == media.media_id
            ).first()

            if not media_with_relations:
                logger.error(f"Media {media.media_id} not found for metadata update")
                return

            silo = media_with_relations.repository.silo
            collection_name = COLLECTION_PREFIX + str(silo.silo_id)

            metadata_updates = {
                "source_type": media_with_relations.source_type,
                "source_url": media_with_relations.source_url,
                "language": media_with_relations.language,
                "name": media_with_relations.name,
            }

            if media_with_relations.folder_id:
                folder_path = FolderService.get_folder_path(media_with_relations.folder_id, session)
                metadata_updates["folder_id"] = media_with_relations.folder_id
                metadata_updates["folder_path"] = folder_path
            else:
                metadata_updates["folder_id"] = None
                metadata_updates["folder_path"] = ""

            updated = _get_vector_store(silo).update_documents_metadata(
                collection_name,
                filter_metadata={
                    "media_id": {"$eq": str(media.media_id)},
                    "content_type": {"$eq": "media_chunk"},
                },
                metadata_updates=metadata_updates,
                replace=False,
            )

            logger.info(
                "Updated metadata for media %s in collection %s (%s chunks)",
                media.media_id, collection_name, updated,
            )

        except Exception as e:
            logger.error(f"Error updating media metadata: {str(e)}")
            raise
        finally:
            if not db_session:
                session.close()

    @staticmethod
    def count_resource_chunks(resource: Resource) -> int:
        """Return the number of document chunks that would be extracted from a resource.

        Uses the same extraction path as index_resource but discards the result.
        Returns 0 on any error so the caller can continue safely.
        """
        session = SessionLocal()
        try:
            resource_with_relations = session.query(Resource).filter(Resource.resource_id == resource.resource_id).first()
            if not resource_with_relations:
                return 0

            if resource_with_relations.folder_id:
                from services.folder_service import FolderService
                folder_path = FolderService.get_folder_path(resource_with_relations.folder_id, session)
                path = os.path.join(REPO_BASE_FOLDER, str(resource_with_relations.repository_id), folder_path, resource_with_relations.uri)
            else:
                path = os.path.join(REPO_BASE_FOLDER, str(resource_with_relations.repository_id), resource_with_relations.uri)

            file_extension = os.path.splitext(resource_with_relations.uri)[1].lower()
            silo = resource_with_relations.repository.silo
            is_lightrag = getattr(silo, 'vector_db_type', None) == 'LIGHTRAG'

            docs = SiloService.extract_documents_from_file(
                path, file_extension, {}, split=not is_lightrag,
            )
            if not docs:
                return 0
            # One unit = one item fed to the vector store: a pre-split chunk for
            # the other backends, a whole page/file for LightRAG (split=False).
            # This has to be the same unit the progress callback reports in, or
            # the bar shows things like "3/14" for a 13-page PDF — LightRAG's
            # own token-chunking is not observable per page beforehand.
            return len(docs)
        except Exception:
            return 0
        finally:
            session.close()

    @staticmethod
    def index_resource(resource: Resource, index_batch: Optional[str] = None, progress_callback=None) -> str:
        """Index a resource into its silo's vector store.

        Every chunk is stamped with ``index_batch`` (generated if not supplied) so a
        later reindex can delete only the stale chunks. Returns the batch used.
        """
        index_batch = index_batch or uuid.uuid4().hex
        session = SessionLocal()
        try:
            resource_with_relations = session.query(Resource).filter(Resource.resource_id == resource.resource_id).first()
            if not resource_with_relations:
                logger.error(f"Resource {resource.resource_id} not found for indexing")
                return index_batch

            collection_name = COLLECTION_PREFIX + str(resource_with_relations.repository.silo_id)

            if resource_with_relations.folder_id:
                from services.folder_service import FolderService
                folder_path = FolderService.get_folder_path(resource_with_relations.folder_id, session)
                path = os.path.join(REPO_BASE_FOLDER, str(resource_with_relations.repository_id), folder_path, resource_with_relations.uri)
            else:
                path = os.path.join(REPO_BASE_FOLDER, str(resource_with_relations.repository_id), resource_with_relations.uri)

            file_extension = os.path.splitext(resource_with_relations.uri)[1].lower()

            base_metadata = {
                "repository_id": resource_with_relations.repository_id,
                "resource_id": resource_with_relations.resource_id,
                "silo_id": resource_with_relations.repository.silo_id,
                "name": resource_with_relations.uri,
                "file_type": file_extension,
                INDEX_BATCH_FIELD: index_batch,
            }

            if resource_with_relations.folder_id:
                from services.folder_service import FolderService
                folder_path = FolderService.get_folder_path(resource_with_relations.folder_id, session)
                base_metadata["folder_id"] = resource_with_relations.folder_id
                base_metadata["folder_path"] = folder_path
                base_metadata["ref"] = os.path.join(str(resource_with_relations.repository_id), folder_path, resource_with_relations.uri)
            else:
                base_metadata["folder_id"] = None
                base_metadata["folder_path"] = ""
                base_metadata["ref"] = os.path.join(str(resource_with_relations.repository_id), resource_with_relations.uri)

            if resource_with_relations.extra_metadata:
                base_metadata = {**resource_with_relations.extra_metadata, **base_metadata}

            # LightRAG chunks internally by token size, so feed it whole
            # pages/files; other backends still need pre-split chunks.
            _is_lightrag = getattr(resource_with_relations.repository.silo, 'vector_db_type', None) == 'LIGHTRAG'
            docs = SiloService.extract_documents_from_file(
                path, file_extension, base_metadata, split=not _is_lightrag,
            )

            if not docs:
                logger.warning(f"No content extracted from resource {resource_with_relations.resource_id} ({resource_with_relations.uri}). The file may be empty or contain only images/scans without text.")
                return index_batch

            embedding_service = resource_with_relations.repository.silo.embedding_service

            if not embedding_service:
                logger.warning(f"Silo {resource_with_relations.repository.silo_id} has no embedding service, skipping indexing for resource {resource_with_relations.resource_id}")
                return index_batch

            silo = resource_with_relations.repository.silo
            app_id = silo.app_id
            silo_id = silo.silo_id
            resource_id = resource_with_relations.resource_id
            content_ref = resource_with_relations.uri
            ai_service = silo.embedding_service  # embedding_service for model name lookup

            # Resolve LLM service for model name (LightRAG only)
            vector_store = _get_vector_store(silo)

            # ----- Metric scaffolding -----
            from repositories.indexing_metric_repository import IndexingMetricRepository
            from services.pricing_service import PricingService

            _metric_kwargs: dict = {}
            _index_failed = False

            try:
                # Index all chunks in one call (batch is more efficient for most backends)
                usage = vector_store.index_documents(
                    collection_name,
                    docs,
                    embedding_service,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                _index_failed = True
                _metric_kwargs["status"] = "failed"
                logger.error(f"Error indexing resource {resource_id}: {exc}")
                raise
            else:
                _metric_kwargs["status"] = "success"
                if usage and isinstance(usage, dict):
                    _metric_kwargs.update({
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "tokens_source": usage.get("tokens_source", "estimated"),
                        "llm_calls": usage.get("llm_calls", 0),
                        "duration_seconds": usage.get("duration_seconds"),
                        "embedding_tokens": usage.get("embedding_tokens") or None,
                    })
                    # Attempt to resolve cost via PricingService
                    try:
                        # Resolve the primary LLM model name from the LightRAG
                        # role services (extract > indexing fallback).
                        lightrag_ai_service = (
                            getattr(silo, "extract_service", None)
                            or getattr(silo, "indexing_service", None)
                        )
                        model_name = getattr(lightrag_ai_service, "description", None)
                        emb_model_name = getattr(embedding_service, "description", None)
                        _metric_kwargs["model_name"] = model_name
                        _metric_kwargs["embedding_model_name"] = emb_model_name

                        total_cost = 0.0
                        has_cost = False

                        if model_name:
                            pricing = PricingService.get_llm_pricing(session, model_name)
                            if pricing and _metric_kwargs.get("total_tokens"):
                                input_per_1m, output_per_1m = pricing
                                total_cost += (
                                    _metric_kwargs["prompt_tokens"] / 1_000_000 * input_per_1m
                                    + _metric_kwargs["completion_tokens"] / 1_000_000 * output_per_1m
                                )
                                has_cost = True

                        emb_tokens = _metric_kwargs.get("embedding_tokens") or 0
                        if emb_model_name and emb_tokens:
                            emb_pricing = PricingService.get_embedding_pricing(session, emb_model_name)
                            if emb_pricing:
                                total_cost += emb_tokens / 1_000_000 * emb_pricing
                                has_cost = True

                        if has_cost:
                            _metric_kwargs["cost"] = round(total_cost, 8)
                            _metric_kwargs["currency"] = "USD"
                    except Exception as pricing_exc:
                        logger.debug(f"Could not resolve pricing for metric: {pricing_exc}")

            finally:
                # Persist metric regardless of success/failure
                try:
                    metric_record = IndexingMetricRepository.create(
                        session,
                        app_id=app_id,
                        silo_id=silo_id,
                        resource_id=resource_id,
                        content_ref=content_ref,
                        **_metric_kwargs,
                    )
                    session.flush()
                    logger.debug(
                        "Persisted IndexingMetric %s for resource %s (status=%s, tokens=%s)",
                        metric_record.metric_id,
                        resource_id,
                        _metric_kwargs.get("status"),
                        _metric_kwargs.get("total_tokens", 0),
                    )
                except Exception as metric_exc:
                    logger.warning(f"Failed to persist indexing metric for resource {resource_id}: {metric_exc}")

            if not _index_failed:
                logger.info(
                    "Indexed resource %s in silo %s (%s chunks)",
                    resource_id,
                    silo_id,
                    len(docs),
                )
                try:
                    from services.metadata_values_cache_service import MetadataValuesCacheService
                    MetadataValuesCacheService.invalidate(silo_id)
                except Exception as _cache_exc:
                    logger.warning("metadata_values_cache: invalidation failed after index_resource for silo=%d: %s", silo_id, _cache_exc)
        except Exception as e:
            logger.error(f"Error indexing resource {resource.resource_id}: {str(e)}")
            raise
        finally:
            session.close()

        return index_batch

    @staticmethod
    def _delete_resource_chunks(resource_id: int, collection_name: str, silo) -> None:
        """Delete all vector chunks for a resource. Propagates vector-store exceptions.

        Single authoritative point for the ``resource_id`` filter; both
        ``delete_resource`` (tolerant) and ``reindex_resource`` (fail-fast) delegate here.
        """
        embedding_service = silo.embedding_service
        _get_vector_store(silo).delete_documents(
            collection_name,
            ids={"resource_id": {"$eq": resource_id}},
            embedding_service=embedding_service,
        )

    @staticmethod
    def reindex_resource(resource: Resource) -> None:
        """Re-index a resource using index-then-swap (no empty-collection window).

        The fresh batch is written first; only after it succeeds are the resource's
        other chunks deleted. If indexing fails, the previous chunks are untouched
        (worst case: transient duplicates that the next successful reindex resolves —
        never zero chunks).

        Raises:
            ValueError: If the silo has no embedding service configured.
            Exception: If the vector-store delete/index fails.
        """
        session = SessionLocal()
        try:
            resource_with_relations = session.scalars(
                select(Resource).where(Resource.resource_id == resource.resource_id)
            ).first()

            if not resource_with_relations:
                logger.warning(
                    "reindex_resource: resource %s not found in database; aborting",
                    resource.resource_id,
                )
                return

            silo = resource_with_relations.repository.silo
            silo_id = resource_with_relations.repository.silo_id

            if not silo or not silo.embedding_service:
                raise ValueError(f"Silo {silo_id} has no embedding service configured")

            # Capture everything needed for the swap before the session closes.
            collection_name = COLLECTION_PREFIX + str(silo_id)
            embedding_service = silo.embedding_service
            vector_store = _get_vector_store(silo)
        finally:
            session.close()

        # 1) Write the fresh batch first — previous chunks remain searchable meanwhile.
        new_batch = SiloService.index_resource(resource)

        # 2) Swap: drop this resource's chunks that are not the fresh batch.
        logger.info(
            "reindex_resource: swapping stale chunks for resource %s in collection %s (batch %s)",
            resource.resource_id, collection_name, new_batch,
        )
        vector_store.delete_documents_excluding(
            collection_name,
            filter_metadata={"resource_id": {"$eq": resource.resource_id}},
            exclude={INDEX_BATCH_FIELD: new_batch},
            embedding_service=embedding_service,
        )

    @staticmethod
    def index_media_chunk(chunk: dict, media: Media, db: Session = None):
        """
        Index a single media chunk in the vector database using chunk data dict and the Media instance.

        `chunk` should be a dict with keys: `text`, `start_time`, `end_time`, `chunk_index` and optionally `chunk_id`.
        The chunk information will be stored inside the embedding metadata (no DB row required).
        """
        if not media:
            logger.error("Media not provided for chunk indexing")
            return

        collection_name = COLLECTION_PREFIX + str(media.repository.silo_id)

        # Build metadata from chunk dict and media
        chunk_type = chunk.get('chunk_type', 'audio')  # 'audio' or 'visual'
        metadata = {
            "repository_id": media.repository_id,
            "media_id": media.media_id,
            "silo_id": media.repository.silo_id,
            "content_type": "media_chunk",
            "chunk_type": chunk_type,
            
            # Chunk-specific data
            "chunk_index": chunk.get('chunk_index'),
            "start_time": chunk.get('start_time'),
            "end_time": chunk.get('end_time'),
            "duration": chunk.get('end_time', 0) - chunk.get('start_time', 0),
            
            # Media file information
            "name": media.name,
            "source_type": media.source_type,
            "source_url": media.source_url,
            "language": media.language,
            "file_type": os.path.splitext(media.file_path)[1].lower() if media.file_path else None,
            "source": media.file_path,
            "processing_mode": media.processing_mode or "basic",
            
            # Folder information
            "folder_id": media.folder_id,
            "folder_path": FolderService.get_folder_path(media.folder_id, db) if media.folder_id else "",
            
            # Reference path (similar to resource 'ref')
            "ref": os.path.join(
                str(media.repository_id),
                FolderService.get_folder_path(media.folder_id, db) if media.folder_id else "",
                f"{media.media_id}{os.path.splitext(media.file_path)[1]}" if media.file_path else ""
            ).replace("\\", "/"),
            
            # Media metadata
            "media_duration": media.duration
        }

        # Add folder information if media is in a folder
        if media.folder_id:
            folder_path = FolderService.get_folder_path(media.folder_id, db)
            metadata["folder_id"] = media.folder_id
            metadata["folder_path"] = folder_path
        else:
            metadata["folder_id"] = None
            metadata["folder_path"] = ""

        # Create Document and index
        page_content = chunk.get('text', '')
        
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )

        embedding_service = media.repository.silo.embedding_service
        if not embedding_service:
            logger.warning(f"Silo {media.repository.silo_id} has no embedding service, skipping indexing for media {media.media_id}")
            return

        try:
            _get_vector_store(media.repository.silo).index_documents(
                collection_name,
                [doc],
                embedding_service
            )
            logger.info(f"Indexed media chunk (media {media.media_id}) in silo {media.repository.silo_id}")
            try:
                from services.metadata_values_cache_service import MetadataValuesCacheService
                MetadataValuesCacheService.invalidate(media.repository.silo_id)
            except Exception as _cache_exc:
                logger.warning("metadata_values_cache: invalidation failed after index_media_chunk for silo=%d: %s", media.repository.silo_id, _cache_exc)
        except Exception as e:
            logger.error(f"Error indexing media chunk for media {media.media_id}: {str(e)}")
            raise

    @staticmethod
    def delete_media(media: Media):
        """Delete all chunks for a media"""
        logger.info(f"Eliminando recurso {media.media_id} del silo {media.repository.silo_id}")
        collection_name = COLLECTION_PREFIX + str(media.repository.silo_id)
        
        # For resource operations, we need a fresh session since this might be called from other contexts
        session = SessionLocal()
        try:
            # Load silo within the session to avoid detached instance issues
            silo = SiloRepository.get_by_id(media.repository.silo_id, session)
            if not silo:
                logger.error(f"Silo no encontrado para la media {media.media_id}")
                return

            # Check if silo has embedding service
            if not silo.embedding_service:
                logger.warning(f"Silo {silo.silo_id} has no embedding service, skipping vector deletion for media {media.media_id}")
                return

            _get_vector_store(silo).delete_documents(
                collection_name,
                ids={"media_id": {"$eq": media.media_id}},
                embedding_service=silo.embedding_service
            )
        except Exception as e:
            logger.error(f"Error deleting media {media.media_id} from vector store: {str(e)}")
            # Don't raise the exception - allow the media to be deleted from database and disk
        finally:
            session.close()

    @staticmethod
    def delete_resource(resource: Resource):
        """
        Delete a resource using its silo's embedding service
        """
        logger.info(f"Eliminando recurso {resource.resource_id} del silo {resource.repository.silo_id}")
        collection_name = COLLECTION_PREFIX + str(resource.repository.silo_id)
        
        # For resource operations, we need a fresh session since this might be called from other contexts
        session = SessionLocal()
        try:
            # Load silo within the session to avoid detached instance issues
            silo = SiloRepository.get_by_id(resource.repository.silo_id, session)
            if not silo:
                logger.error(f"Silo no encontrado para el recurso {resource.resource_id}")
                return

            # Check if silo has embedding service
            if not silo.embedding_service:
                logger.warning(f"Silo {silo.silo_id} has no embedding service, skipping vector deletion for resource {resource.resource_id}")
                return

            SiloService._delete_resource_chunks(resource.resource_id, collection_name, silo)
            # Only reached when _delete_resource_chunks succeeded — safe to invalidate.
            try:
                from services.metadata_values_cache_service import MetadataValuesCacheService
                MetadataValuesCacheService.invalidate(silo.silo_id)
            except Exception as _cache_exc:
                logger.warning("metadata_values_cache: invalidation failed after delete_resource for silo=%d: %s", silo.silo_id, _cache_exc)
        except Exception as e:
            logger.error(f"Error deleting resource {resource.resource_id} from vector store: {str(e)}")
            # Don't raise the exception - allow the resource to be deleted from database and disk
        finally:
            session.close()

    @staticmethod
    def delete_url(silo_id: int, url: str, db: Session):
        """
        Delete a resource using its silo's embedding service
        """
        logger.info(f"Eliminando URL {url} del silo {silo_id}")
        collection_name = COLLECTION_PREFIX + str(silo_id)
        
        # Get silo within this session to avoid detached instance
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            logger.error(f"Silo no encontrado para la url {url}")
            return

        # Get embedding service within the same session
        embedding_service = None
        if silo.embedding_service_id:
            embedding_service = SiloRepository.get_embedding_service_by_id(silo.embedding_service_id, db)
        
        _get_vector_store(silo).delete_documents(
            collection_name,
            ids={"url": {"$eq": url}},
            embedding_service=embedding_service
        )
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_url for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def delete_content(silo_id: int, content_id: str, db: Session):
        """
        Delete content from a silo using its embedding service
        """
        logger.info(f"Eliminando contenido {content_id} del silo {silo_id}")
        
        if not SiloService.check_silo_collection_exists(silo_id, db):
            logger.warning(f"La colección para el silo {silo_id} no existe")
            return

        silo = SiloService.get_silo(silo_id, db)
        if not silo:
            logger.error(f"Silo {silo_id} no encontrado")
            return

        collection_name = COLLECTION_PREFIX + str(silo_id)
        _get_vector_store(silo).delete_documents(
            collection_name,
            filter_metadata={"id": {"$eq": content_id}},
            embedding_service=silo.embedding_service
        )
        logger.info(f"Contenido {content_id} eliminado correctamente del silo {silo_id}")
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_content for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def delete_collection(silo_id: int, db: Session):
        """Delete a collection using its silo's embedding service"""
        if not SiloService.check_silo_collection_exists(silo_id, db):
            return

        # Get silo within the session to ensure relationships are loaded
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            return

        collection_name = COLLECTION_PREFIX + str(silo_id)
        _get_vector_store(silo).delete_collection(collection_name, silo.embedding_service)
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_collection for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def delete_docs_in_collection(silo_id: int, ids: List[str], db: Session):
        """
        Delete documents from a silo using its embedding service
        """
        logger.info(f"Eliminando documentos {ids} del silo {silo_id}")
        
        if not SiloService.check_silo_collection_exists(silo_id, db):
            logger.warning(f"La colección para el silo {silo_id} no existe")
            return

        # Get silo within the session to ensure relationships are loaded
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            logger.error(f"Silo {silo_id} no encontrado")
            return

        collection_name = COLLECTION_PREFIX + str(silo_id)
        _get_vector_store(silo).delete_documents(
            collection_name,
            ids=ids,
            embedding_service=silo.embedding_service
        )
        logger.info(f"Documentos eliminados correctamente del silo {silo_id}")
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_docs_in_collection for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def delete_all_docs_in_collection(silo_id: int, db: Session):
        """
        Delete all documents from a silo collection.
        """
        logger.info(f"Deleting all documents from silo {silo_id}")
        
        if not SiloService.check_silo_collection_exists(silo_id, db):
            logger.warning(f"Collection for silo {silo_id} does not exist, nothing to delete")
            return

        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            logger.error(f"Silo {silo_id} not found")
            return

        collection_name = COLLECTION_PREFIX + str(silo_id)
        _get_vector_store(silo).delete_collection(collection_name, silo.embedding_service)
        logger.info(f"All documents deleted from silo {silo_id}")
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_all_docs_in_collection for silo=%d: %s", silo_id, _cache_exc)

    @staticmethod
    def delete_docs_by_metadata(silo_id: int, filter_metadata: Dict[str, Any], db: Session) -> int:
        """
        Delete documents from a silo using metadata filters.
        
        Args:
            silo_id: ID of the silo
            filter_metadata: Metadata filter dict (MongoDB-style, e.g., {"field": {"$eq": "value"}})
            db: Database session
            
        Returns:
            Number of documents deleted
        """
        logger.info(f"Deleting documents from silo {silo_id} with filter: {filter_metadata}")
        
        if not SiloService.check_silo_collection_exists(silo_id, db):
            logger.warning(f"Collection for silo {silo_id} does not exist")
            return 0

        # Get silo within the session to ensure relationships are loaded
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            logger.error(f"Silo {silo_id} not found")
            return 0

        if not silo.embedding_service:
            logger.warning(f"Silo {silo_id} has no embedding service configured")
            return 0

        collection_name = COLLECTION_PREFIX + str(silo_id)
        store = _get_vector_store(silo)

        # Count matches first via the vector store abstraction (faster than search).
        doc_count = store.count_documents(collection_name, filter_metadata=filter_metadata)

        if doc_count == 0:
            logger.info(f"No documents found matching the filter in silo {silo_id}")
            return 0

        # Delete the matched documents using metadata filter
        store.delete_documents(
            collection_name,
            ids=filter_metadata,  # Pass metadata filter as dict
            embedding_service=silo.embedding_service
        )
        logger.info(f"Successfully deleted {doc_count} document(s) from silo {silo_id}")
        try:
            from services.metadata_values_cache_service import MetadataValuesCacheService
            MetadataValuesCacheService.invalidate(silo_id)
        except Exception as _cache_exc:
            logger.warning("metadata_values_cache: invalidation failed after delete_docs_by_metadata for silo=%d: %s", silo_id, _cache_exc)
        return doc_count

    @staticmethod
    def find_docs_in_collection(
        silo_id: int,
        query: str,
        filter_metadata: Optional[dict] = None,
        limit: Optional[int] = None,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
        db: Session = None,
    ) -> List[Document]:
        # Get silo within the session to ensure relationships are loaded
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo or not SiloService.check_silo_collection_exists(silo_id, db):
            return []
        
        collection_name = COLLECTION_PREFIX + str(silo_id)
        
        # Get embedding service within the same session
        embedding_service = None
        if silo.embedding_service_id:
            embedding_service = SiloRepository.get_embedding_service_by_id(silo.embedding_service_id, db)
        
        results_limit = limit if limit and limit > 0 else DEFAULT_SEARCH_LIMIT
        if results_limit > MAX_SEARCH_LIMIT:
            results_limit = MAX_SEARCH_LIMIT

        docs = _get_vector_store(silo).search_similar_documents(
            collection_name,
            query,
            embedding_service=embedding_service,
            filter_metadata=filter_metadata or {},
            k=results_limit,
            search_type=search_type,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        if min_content_length is not None or max_content_length is not None:
            docs = [
                d for d in docs
                if (min_content_length is None or len(d.page_content) >= min_content_length)
                and (max_content_length is None or len(d.page_content) <= max_content_length)
            ]
        return docs

    @staticmethod
    def search_in_silo(
        silo_id: int,
        query: str,
        filter_metadata: Optional[dict] = None,
        limit: Optional[int] = None,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
        db: Session = None,
    ) -> List[Document]:
        """
        Search for documents in a silo using semantic search
        
        Args:
            silo_id: ID of the silo to search in
            query: Search query text
            filter_metadata: Optional metadata filters
            limit: Maximum number of results to return
            db: Database session
            
        Returns:
            List of Document objects with page_content and metadata
        """
        # Use find_docs_in_collection as the base implementation
        return SiloService.find_docs_in_collection(
            silo_id,
            query,
            filter_metadata,
            limit=limit,
            search_type=search_type,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            db=db,
        )

    @staticmethod
    def _get_filter_value_by_type(field_value: str, field_type: str) -> dict:
        """Helper method to convert field value to the appropriate type for filtering"""
        if field_type == 'int':
            return {"$eq": int(field_value)}
        elif field_type == 'float':
            return {"$eq": float(field_value)}
        elif field_type == 'bool':
            return {"$eq": field_value}
        elif field_type in ['str', 'date']:
            return {"$eq": field_value}
        return {"$eq": field_value}  # default case

    @staticmethod
    def get_metadata_filter_from_form(silo: Silo, form_data: dict) -> dict:
        filter_dict = {}
        if not silo.metadata_definition:
            return filter_dict

        field_definitions = {f['name']: f for f in silo.metadata_definition.fields}
        filter_prefix = 'filter_'
        
        for field_name, field_value in form_data.items():
            if not field_value or field_value == '':
                continue
                
            if not field_name.startswith(filter_prefix):
                continue
                
            name = field_name[len(filter_prefix):]
            if name not in field_definitions:
                continue
                
            field_definition = field_definitions[name]
            filter_dict[name] = SiloService._get_filter_value_by_type(
                field_value, 
                field_definition['type']
            )
        
        return filter_dict 

    # ==================== ROUTER SERVICE METHODS ====================
    
    @staticmethod
    def get_silos_list(app_id: int, db: Session) -> List[SiloListItemSchema]:
        """
        Get list of silos for a specific app with document counts
        """
        # Get silos using the existing service
        silos = SiloService.get_silos_by_app_id(app_id, db)
        
        result = []
        for silo in silos:
            # Get document count
            docs_count = SiloService.count_docs_in_silo(silo.silo_id, db)
            
            result.append(SiloListItemSchema(
                silo_id=silo.silo_id,
                name=silo.name,
                description=silo.description,
                type=silo.silo_type if silo.silo_type else None,
                created_at=silo.create_date,
                docs_count=docs_count,
                vector_db_type=silo.vector_db_type
            ))
        
        return result
    
    @staticmethod
    def get_silo_detail(app_id: int, silo_id: int, db: Session) -> SiloDetailSchema:
        """
        Get detailed silo information including form data for editing
        """
        logger.info(f"Getting silo detail for app_id: {app_id}, silo_id: {silo_id}")
        
        if silo_id == 0:
            # New silo
            logger.info("Returning new silo template")
            vector_db_options = VectorStoreFactory.get_available_type_options()
            default_vector_db_type = _resolve_vector_db_type()
            # Get AI services for indexing LLM selector
            try:
                form_data = SiloRepository.get_form_data_for_silo(app_id, 0, db)
                ai_services = [
                    {"service_id": s.service_id, "name": s.name, "provider": s.provider}
                    for s in form_data.get('ai_services', [])
                ]
            except Exception:
                ai_services = []
            return SiloDetailSchema(
                silo_id=0,
                name="",
                description=None,
                type=None,
                created_at=None,
                docs_count=0,
                vector_db_type=default_vector_db_type,
                lightrag_vector_db_type='QDRANT',
                # Form data
                output_parsers=[],
                embedding_services=[],
                ai_services=ai_services,
                vector_db_options=vector_db_options
            )
        
        # Existing silo
        logger.info(f"Getting existing silo {silo_id}")
        silo = SiloService.get_silo(silo_id, db)
        if not silo:
            logger.error(f"Silo {silo_id} not found")
            return None
        
        logger.info(f"Found silo: {silo.name}, app_id: {silo.app_id}")
        
        # Get document count
        try:
            logger.info(f"Counting docs in silo {silo_id}")
            docs_count = SiloService.count_docs_in_silo(silo_id, db)
            logger.info(f"Docs count: {docs_count}")
        except Exception as e:
            logger.error(f"Error counting docs in silo {silo_id}: {str(e)}")
            docs_count = 0
        
        # Get form data using repository consolidation
        try:
            logger.info(f"Getting form data for app_id: {app_id}")
            form_data = SiloRepository.get_form_data_for_silo(app_id, 0, db)  # We already have the silo
            output_parsers = [{"parser_id": p.parser_id, "name": p.name} for p in form_data['output_parsers']]
            from schemas.embedding_service_schemas import EmbeddingServiceOptionSchema
            embedding_services = (
                [EmbeddingServiceOptionSchema(service_id=s.service_id, name=s.name, provider=s.provider.value if hasattr(s.provider, 'value') else s.provider, is_system=False) for s in form_data['embedding_services']]
                + [EmbeddingServiceOptionSchema(service_id=s.service_id, name=s.name, provider=s.provider.value if hasattr(s.provider, 'value') else s.provider, is_system=True) for s in form_data.get('system_embedding_services', [])]
            )
            ai_services = [
                {"service_id": s.service_id, "name": s.name, "provider": s.provider}
                for s in form_data.get('ai_services', [])
            ]
            logger.info(f"Found {len(output_parsers)} parsers, {len(embedding_services)} embedding services, {len(ai_services)} AI services")
        except Exception as e:
            logger.error(f"Error getting form data: {str(e)}")
            output_parsers = []
            embedding_services = []
            ai_services = []
        
        # Get metadata definition fields if silo has one
        metadata_fields = None
        try:
            if silo.metadata_definition_id:
                logger.info(f"Getting metadata parser {silo.metadata_definition_id}")
                metadata_parser = SiloRepository.get_output_parser_by_id(silo.metadata_definition_id, db)
                if metadata_parser and metadata_parser.fields:
                    metadata_fields = [
                        {
                            "name": field.get("name", ""),
                            "type": field.get("type", "str"),
                            "description": field.get("description", "")
                        }
                        for field in metadata_parser.fields
                    ]
        except Exception as e:
            logger.error(f"Error getting metadata fields: {str(e)}")
            metadata_fields = None
        
        try:
            logger.info(f"Creating SiloDetailSchema for silo {silo_id}")
            vector_db_type = _resolve_vector_db_type(silo)
            vector_db_options = VectorStoreFactory.get_available_type_options()
            return SiloDetailSchema(
                silo_id=silo.silo_id,
                name=silo.name,
                description=silo.description,
                type=silo.silo_type if silo.silo_type else None,
                created_at=silo.create_date,
                docs_count=docs_count,
                vector_db_type=vector_db_type,
                # Current values for editing
                metadata_definition_id=silo.metadata_definition_id,
                embedding_service_id=silo.embedding_service_id,
                indexing_service_id=silo.indexing_service_id,
                extract_service_id=getattr(silo, 'extract_service_id', None) or silo.indexing_service_id,
                keywords_service_id=getattr(silo, 'keywords_service_id', None),
                vlm_service_id=getattr(silo, 'vlm_service_id', None),
                lightrag_vector_db_type=getattr(silo, 'lightrag_vector_db_type', None) or 'QDRANT',
                lightrag_chunk_strategy=silo.lightrag_chunk_strategy,
                lightrag_chunk_token_size=silo.lightrag_chunk_token_size,
                lightrag_chunk_overlap_token_size=silo.lightrag_chunk_overlap_token_size,
                lightrag_language=getattr(silo, 'lightrag_language', None),
                lightrag_entity_extract_max_gleaning=getattr(silo, 'lightrag_entity_extract_max_gleaning', None),
                lightrag_max_source_ids_per_entity=getattr(silo, 'lightrag_max_source_ids_per_entity', None),
                lightrag_max_source_ids_per_relation=getattr(silo, 'lightrag_max_source_ids_per_relation', None),
                lightrag_entity_types=getattr(silo, 'lightrag_entity_types', None),
                # Form data
                output_parsers=output_parsers,
                embedding_services=embedding_services,
                ai_services=ai_services,
                vector_db_options=vector_db_options,
                # Metadata definition fields for playground
                metadata_fields=metadata_fields
            )
        except Exception as e:
            logger.error(f"Error creating SiloDetailSchema: {str(e)}")
            raise
    
    @staticmethod
    def create_or_update_silo_router(
        app_id: int, 
        silo_id: int, 
        silo_data: CreateUpdateSiloSchema, 
        db: Session
    ) -> Silo:
        """
        Create or update silo using router data
        """
        # Prepare form data for the service
        # vector_db_type uses getattr so this method works with both CreateSiloSchema
        # (which has the field) and UpdateSiloSchema (which intentionally omits it).
        form_data = {
            'silo_id': silo_id,
            'name': silo_data.name,
            'description': silo_data.description,
            'app_id': app_id,
            'type': silo_data.type,
            'output_parser_id': silo_data.output_parser_id,
            'embedding_service_id': getattr(silo_data, 'embedding_service_id', None),
            'vector_db_type': getattr(silo_data, 'vector_db_type', None),
            'lightrag_vector_db_type': getattr(silo_data, 'lightrag_vector_db_type', None),
            'indexing_service_id': getattr(silo_data, 'indexing_service_id', None),
            'extract_service_id': getattr(silo_data, 'extract_service_id', None),
            'keywords_service_id': getattr(silo_data, 'keywords_service_id', None),
            'vlm_service_id': getattr(silo_data, 'vlm_service_id', None),
            'lightrag_chunk_strategy': getattr(silo_data, 'lightrag_chunk_strategy', None),
            'lightrag_chunk_token_size': getattr(silo_data, 'lightrag_chunk_token_size', None),
            'lightrag_chunk_overlap_token_size': getattr(silo_data, 'lightrag_chunk_overlap_token_size', None),
            'lightrag_language': getattr(silo_data, 'lightrag_language', None),
            'lightrag_entity_extract_max_gleaning': getattr(silo_data, 'lightrag_entity_extract_max_gleaning', None),
            'lightrag_max_source_ids_per_entity': getattr(silo_data, 'lightrag_max_source_ids_per_entity', None),
            'lightrag_max_source_ids_per_relation': getattr(silo_data, 'lightrag_max_source_ids_per_relation', None),
            'lightrag_entity_types': getattr(silo_data, 'lightrag_entity_types', None),
        }
        
        # Create or update using the existing service
        silo = SiloService.create_or_update_silo(form_data, db=db)
        return silo
    
    @staticmethod
    def delete_silo_router(silo_id: int, db: Session) -> bool:
        """
        Delete a silo and all its documents
        """
        return SiloRepository.delete(silo_id, db)
    
    @staticmethod
    async def _search_via_lightrag_retriever(
        silo,
        query: str,
        lightrag_query_mode: str,
        limit: Optional[int],
        filter_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async search using LightRAG retriever (no LLM generation).

        Uses aretrieve_graph_context so Neo4j runs in the correct event loop.
        Returns chunks from lightrag_raw_data and graph data separately.
        """
        collection_name = COLLECTION_PREFIX + str(silo.silo_id)
        results_limit = limit if limit and limit > 0 else DEFAULT_SEARCH_LIMIT
        if results_limit > MAX_SEARCH_LIMIT:
            results_limit = MAX_SEARCH_LIMIT

        store = _get_vector_store(silo)
        raw_data = await store.aretrieve_graph_context(
            collection_name, query, lightrag_query_mode, results_limit
        )

        lightrag_graph = raw_data if raw_data else None
        chunk_results = []
        chunks = (raw_data.get("data") or {}).get("chunks") or []
        for chunk in chunks:
            chunk_results.append({
                "page_content": chunk.get("content") or "",
                "metadata": {k: v for k, v in chunk.items() if k != "content"},
                "score": None,
            })

        return {
            "query": query,
            "results": chunk_results,
            "total_results": len(chunk_results),
            "filter_metadata": filter_metadata,
            "lightrag_graph": lightrag_graph,
        }

    @staticmethod
    def search_silo_documents_router(
        silo_id: int,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
        lightrag_query_mode: Optional[str] = None,
        db: Session = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Search for documents in a silo using semantic search with optional metadata filtering.
        """
        silo = SiloService.get_silo(silo_id, db)
        if not silo:
            return None

        if lightrag_query_mode is not None:
            return SiloService._search_via_lightrag_retriever(
                silo, query, lightrag_query_mode, limit, filter_metadata
            )

        results = SiloService.find_docs_in_collection(
            silo_id,
            query,
            filter_metadata=filter_metadata,
            limit=limit,
            search_type=search_type,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            db=db,
        )

        response_results = []
        for doc in results:
            score = doc.metadata.pop('_score', None) if '_score' in doc.metadata else None
            response_results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })

        return {
            "query": query,
            "results": response_results,
            "total_results": len(response_results),
            "filter_metadata": filter_metadata
        }

    @staticmethod
    def get_neighboring_chunks(
        silo_id: int,
        source_type: str,
        source_id: str,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all chunks from the same source document, ordered by position.

        source_type: "media" or "resource"
        source_id:   str value of media_id or resource_id (as stored in metadata)

        Raises ValueError for unsupported source_type.
        """
        if source_type not in ("media", "resource"):
            raise ValueError(f"Unsupported source_type '{source_type}'. Must be 'media' or 'resource'.")

        if source_type == "media":
            filter_metadata = {
                "media_id": {"$eq": source_id},
                "content_type": {"$eq": "media_chunk"},
            }
        else:
            filter_metadata = {"resource_id": {"$eq": source_id}}

        docs = SiloService.find_docs_in_collection(
            silo_id,
            "",
            filter_metadata=filter_metadata,
            limit=MAX_SEARCH_LIMIT,
            db=db,
        )

        # Sort by position in the source document
        if source_type == "media":
            docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        else:
            docs.sort(key=lambda d: d.metadata.get("page", d.metadata.get("chunk_index", 0)))

        return [
            {"page_content": doc.page_content, "metadata": doc.metadata, "score": None}
            for doc in docs
        ]

    @staticmethod
    def get_metadata_field_values(
        silo_id: int,
        field: str,
        prefix: Optional[str] = None,
        limit: int = 100,
        db: Session = None,
    ) -> List[str]:
        """
        Return distinct values for a metadata field in the silo's vector collection.
        Sorted alphabetically, filtered by optional case-insensitive prefix.
        limit is clamped to 1–500.
        Raises ValueError for invalid field names, NotFoundError for missing silo.
        """
        import re
        if not field or not re.match(r'^[\w.\-]+$', field):
            raise ValueError(f"Invalid metadata field name: '{field}'")

        limit = min(max(1, limit), 500)

        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            raise NotFoundError(f"Silo {silo_id} not found", "silo")

        collection_name = COLLECTION_PREFIX + str(silo_id)
        return _get_vector_store(silo).get_distinct_metadata_values(
            collection_name, field, prefix=prefix, limit=limit,
        )

    @staticmethod
    def estimate_indexing_cost(silo_id: int, documents: List[Dict[str, Any]], db: Session) -> dict:
        """Estimate the cost of indexing documents into a LightRAG silo.

        Returns a dict matching CostEstimationResponseSchema fields.
        Raises LookupError if silo not found, ValueError if not a LightRAG silo
        or missing required services.
        """
        silo = SiloRepository.get_by_id(silo_id, db)
        if not silo:
            raise LookupError(f"Silo {silo_id} not found")

        if getattr(silo, 'vector_db_type', None) != 'LIGHTRAG':
            raise ValueError("Cost estimation is only available for LightRAG silos")

        if not silo.indexing_service:
            raise ValueError("Silo has no indexing AI service configured")
        if not silo.embedding_service:
            raise ValueError("Silo has no embedding service configured")

        chunk_token_size = silo.lightrag_chunk_token_size or 1200
        chunk_overlap_token_size = silo.lightrag_chunk_overlap_token_size or 0

        num_chunks = 0
        total_content_tokens = 0
        for doc in documents:
            doc_tokens = _count_tokens(doc.get('content', ''))
            total_content_tokens += doc_tokens
            num_chunks += _chunks_from_tokens(doc_tokens, chunk_token_size, chunk_overlap_token_size)

        import config
        max_gleaning = config.ENTITY_EXTRACT_MAX_GLEANING

        # LightRAG extracts entities AND relationships in a SINGLE LLM call per
        # chunk (one JSON response with both), plus one extra call per gleaning
        # pass. So calls/chunk = 1 + max_gleaning (not 2 + max_gleaning).
        extraction_calls = num_chunks * (1 + max_gleaning)
        embedding_calls = num_chunks
        # Fixed prompt overhead per LightRAG extraction call (measured against
        # lightrag.prompt.PROMPTS): system 1,305 + few-shot examples 2,244 +
        # user wrapper 250 ≈ 3,800 tokens on top of the chunk content every call.
        _prompt_overhead_tokens = 3800
        # Entity+relationship JSON output is large and varies with entity density.
        _out_per_call_avg = 2000
        _out_per_call_min = 1000
        _out_per_call_max = 3000

        # Average real chunk size in tokens (last chunk of a doc is smaller than
        # chunk_token_size), used for the per-call content portion.
        avg_chunk_tokens = (total_content_tokens / num_chunks) if num_chunks else chunk_token_size

        estimated_llm_calls = extraction_calls
        estimated_input_tokens = int(extraction_calls * (avg_chunk_tokens + _prompt_overhead_tokens))
        estimated_output_tokens = int(extraction_calls * _out_per_call_avg)

        # Embedding tokens = chunk embeddings (≈ all content) + the entity and
        # relationship description embeddings LightRAG writes to the vector store.
        # Those descriptions ARE the extraction output, so the graph contribution
        # tracks output (entity density), not raw content. A flat content-based
        # factor underestimated dense documents by 2-3x.
        estimated_embedding_tokens = int(total_content_tokens + estimated_output_tokens)

        estimated_cost_min = None
        estimated_cost_max = None

        warnings: List[str] = []

        def _service_model_name(service):
            if service is None:
                return ''
            return (
                getattr(service, 'description', None)
                or getattr(service, 'model_name', None)
                or getattr(service, 'name', None)
                or ''
            )

        # EXTRACT model drives indexing cost — prefer the new role column,
        # fall back to legacy indexing_service for older silos.
        extract_service = getattr(silo, 'extract_service', None) or silo.indexing_service
        model_name = _service_model_name(extract_service)
        embedding_model_name = (
            getattr(silo.embedding_service, 'description', None)
            or getattr(silo.embedding_service, 'model_name', None)
            or getattr(silo.embedding_service, 'name', None)
            or None
        )

        # --- Pricing ---
        from services.pricing_service import PricingService
        _ensure_pricing_catalog(db)

        currency = "USD"

        # Self-hosted inference (Ollama, or a Custom OpenAI-compatible endpoint
        # such as vLLM) has no per-token price, so it counts as 0 and the other,
        # genuinely billed side still yields a number — previously a self-hosted
        # extraction model made the whole estimate null even when embeddings were
        # billed by OpenAI. When BOTH sides are self-hosted there is nothing to
        # add up, and the real cost (GPU time, power) is not something this
        # estimate can know: it is reported as unavailable rather than "0.00",
        # which would read as "free".
        llm_self_hosted = getattr(extract_service, 'provider', None) in _SELF_HOSTED_PROVIDERS
        emb_self_hosted = getattr(silo.embedding_service, 'provider', None) in _SELF_HOSTED_PROVIDERS

        llm_price = None if llm_self_hosted else PricingService.get_llm_pricing(db, model_name)
        emb_price = (
            None if emb_self_hosted
            else PricingService.get_embedding_pricing(db, embedding_model_name)
        )

        llm_cost_min = llm_cost_max = None
        if llm_self_hosted:
            llm_cost_min = llm_cost_max = 0.0
        elif llm_price is not None:
            in_price, out_price = llm_price
            llm_input_cost = estimated_input_tokens * in_price / 1_000_000
            llm_cost_min = (
                llm_input_cost
                + (extraction_calls * _out_per_call_min) * out_price / 1_000_000
            )
            llm_cost_max = (
                llm_input_cost
                + (extraction_calls * _out_per_call_max) * out_price / 1_000_000
            )
        else:
            warnings.append(
                f"No pricing found for extraction model '{model_name}', so the "
                f"cost estimate is unavailable."
            )

        emb_cost = None
        if emb_self_hosted:
            emb_cost = 0.0
        elif emb_price is not None:
            emb_cost = estimated_embedding_tokens * emb_price / 1_000_000
        else:
            warnings.append(
                f"No pricing found for embedding model '{embedding_model_name}', "
                f"so the cost estimate is unavailable."
            )

        if llm_self_hosted and emb_self_hosted:
            warnings.append(
                "Cost estimate unavailable: both the extraction model "
                f"('{model_name}') and the embedding model "
                f"('{embedding_model_name}') are self-hosted, so there is no "
                "per-token price to add up."
            )
        elif llm_cost_min is not None and emb_cost is not None:
            estimated_cost_min = round(llm_cost_min + emb_cost, 4)
            estimated_cost_max = round(llm_cost_max + emb_cost, 4)

        # --- Time Estimates (seconds) ---
        # Account for worker concurrency. When we already have successful
        # indexing runs for this silo, prefer an empirical throughput derived
        # from real durations over static heuristics.
        llm_workers = getattr(silo, 'llm_model_max_sync', 4) or 4
        embedding_workers = getattr(silo, 'embedding_model_max_sync', 8) or 8
        overhead_factor = 1.1

        successful_metrics = (
            db.query(IndexingMetric)
            .filter(
                IndexingMetric.silo_id == silo.silo_id,
                IndexingMetric.status == 'success',
                IndexingMetric.total_tokens > 0,
                IndexingMetric.duration_seconds > 0,
            )
            .order_by(IndexingMetric.created_at.desc())
            .limit(10)
            .all()
        )

        # Cold start: the first indexing into a silo also creates the Neo4j
        # database + full-text/B-tree indexes, the Qdrant collections and
        # initialises LightRAG storages — tens of seconds that a warm silo
        # (existing successful runs) no longer pays.
        fixed_overhead_s = 8 if successful_metrics else 30

        # Default envelope tuned from observed local runs; historical metrics
        # below can further refine the estimate for an already-used silo.
        llm_throughput_opt = 180.0
        llm_throughput_pess = 70.0
        embedding_throughput_opt = 1200.0
        embedding_throughput_pess = 350.0

        # Phase 1 — extraction (1 call/chunk + gleaning passes, parallelised across chunks)
        llm_tokens_per_chunk = (avg_chunk_tokens + _prompt_overhead_tokens) * (1 + max_gleaning)
        llm_s_per_chunk_opt = (llm_tokens_per_chunk / llm_throughput_opt) * overhead_factor
        llm_s_per_chunk_pess = (llm_tokens_per_chunk / llm_throughput_pess) * overhead_factor
        llm_wall_s_opt = math.ceil(num_chunks / llm_workers) * llm_s_per_chunk_opt
        llm_wall_s_pess = math.ceil(num_chunks / llm_workers) * llm_s_per_chunk_pess

        # Embedding (parallel to LLM, but we take the max wall time)
        emb_tokens_total = estimated_embedding_tokens
        emb_raw_s_opt = (emb_tokens_total / embedding_throughput_opt) * overhead_factor
        emb_raw_s_pess = (emb_tokens_total / embedding_throughput_pess) * overhead_factor
        emb_wall_s_opt = emb_raw_s_opt / max(1, embedding_workers)
        emb_wall_s_pess = emb_raw_s_pess / max(1, embedding_workers)

        estimated_indexing_time_min = fixed_overhead_s + max(llm_wall_s_opt, emb_wall_s_opt)
        estimated_indexing_time_max = fixed_overhead_s + max(llm_wall_s_pess, emb_wall_s_pess)
        if successful_metrics:
            observed_total_tokens = sum(metric.total_tokens or 0 for metric in successful_metrics)
            observed_total_duration = sum(metric.duration_seconds or 0 for metric in successful_metrics)
            if observed_total_tokens > 0 and observed_total_duration > 0:
                observed_tokens_per_second = observed_total_tokens / observed_total_duration
                estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
                observed_time = fixed_overhead_s + (estimated_total_tokens / observed_tokens_per_second)
                estimated_indexing_time_min = min(estimated_indexing_time_min, observed_time * 0.85)
                estimated_indexing_time_max = min(estimated_indexing_time_max, observed_time * 1.35)

        estimated_indexing_time_min = round(max(fixed_overhead_s, estimated_indexing_time_min), 1)
        estimated_indexing_time_max = round(max(estimated_indexing_time_min, estimated_indexing_time_max), 1)
        estimated_indexing_time_avg = round((estimated_indexing_time_min + estimated_indexing_time_max) / 2, 1)

        # Emit per-role warnings for every configured role service.
        role_services = {
            'extract':  extract_service,
            'keywords': getattr(silo, 'keywords_service', None),
            'vlm':      getattr(silo, 'vlm_service', None),
        }
        for role, service in role_services.items():
            role_model = _service_model_name(service)
            if not role_model:
                continue
            warning = _validate_model_for_role(role, role_model)
            if warning:
                warnings.append(warning)

        return {
            "total_chunks": num_chunks,
            "chunk_token_size": chunk_token_size,
            "estimated_llm_calls": estimated_llm_calls,
            "estimated_embedding_calls": embedding_calls,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_embedding_tokens": estimated_embedding_tokens,
            "estimated_cost_min": estimated_cost_min,
            "estimated_cost_max": estimated_cost_max,
            "currency": currency,
            "model_name": model_name or None,
            "embedding_model_name": embedding_model_name,
            "estimated_indexing_time_min": estimated_indexing_time_min,
            "estimated_indexing_time_max": estimated_indexing_time_max,
            "estimated_indexing_time_avg": estimated_indexing_time_avg,
            "warnings": warnings,
        }
