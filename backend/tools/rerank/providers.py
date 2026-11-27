import math
from dataclasses import dataclass, field
from typing import Any, Optional

from core.export_constants import PLACEHOLDER_API_KEY
from models.rerank_service import RerankProvider
from tools.rerank.endpoints import rerank_url
from tools.rerank.errors import InvalidRerankResponseError, RerankError, UnknownRerankProviderError
from tools.rerank.types import RerankResult, RerankTarget


@dataclass(frozen=True)
class PreparedRequest:
    url: str
    headers: dict = field(repr=False)  # may carry the Authorization key
    json: dict = field(repr=False)  # carries the query and documents


def _require_known_provider(provider: Optional[str]) -> None:
    if provider != RerankProvider.CohereCompatible.value:
        raise UnknownRerankProviderError("Unknown rerank provider")


def build_request(target: RerankTarget, query: str, documents: list[str], top_n: int) -> PreparedRequest:
    """Build the Cohere / Jina / Infinity / vLLM / LiteLLM `POST {endpoint}/rerank` request (AD-5).

    Raises:
        UnknownRerankProviderError: The target's provider is not the Cohere-compatible one.
        RerankError: `top_n` is not an integer >= 1.
    """
    _require_known_provider(target.provider)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise RerankError("top_n must be an integer >= 1", kind="config")
    headers = {"Content-Type": "application/json"}
    key = (target.api_key or "").strip()
    if key and key != PLACEHOLDER_API_KEY:
        headers["Authorization"] = f"Bearer {key}"
    return PreparedRequest(
        url=rerank_url(target.endpoint),
        headers=headers,
        json={"model": target.model, "query": query, "documents": list(documents),
              "top_n": min(top_n, len(documents))},
    )


def parse_response(status_code: int, body: Any, n_documents: int, top_n: int) -> list[RerankResult]:
    """Validate a Cohere-compatible response and return the top `top_n` results by score."""
    if not isinstance(body, dict):
        raise InvalidRerankResponseError("Response is not a JSON object")
    results = body.get("results")
    if not isinstance(results, list):
        raise InvalidRerankResponseError("Response has no 'results' list")
    if len(results) > n_documents:
        raise InvalidRerankResponseError("Response has more results than documents sent")
    seen: set[int] = set()
    parsed: list[RerankResult] = []
    for item in results:
        if not isinstance(item, dict):
            raise InvalidRerankResponseError("A result item is not an object")
        idx = item.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise InvalidRerankResponseError("A result is missing an integer 'index'")
        if idx < 0 or idx >= n_documents:
            raise InvalidRerankResponseError("A result index is out of range")
        if idx in seen:
            raise InvalidRerankResponseError("Duplicate result index")
        score = item.get("relevance_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidRerankResponseError("A result is missing a numeric 'relevance_score'")
        try:
            fscore = float(score)
        except OverflowError:
            raise InvalidRerankResponseError("A 'relevance_score' is out of numeric range") from None
        if not math.isfinite(fscore):
            raise InvalidRerankResponseError("A 'relevance_score' is not finite")
        seen.add(idx)
        parsed.append(RerankResult(index=idx, score=fscore))
    parsed.sort(key=lambda r: r.score, reverse=True)  # stable
    return parsed[:top_n]
