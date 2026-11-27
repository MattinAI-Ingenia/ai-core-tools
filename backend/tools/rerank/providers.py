import math
from abc import ABC, abstractmethod
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


class RerankAdapter(ABC):
    @abstractmethod
    def build_request(self, target: RerankTarget, query: str, documents: list[str], top_n: int) -> PreparedRequest:
        ...

    @abstractmethod
    def parse_response(self, status_code: int, body: Any, n_documents: int, top_n: int) -> list[RerankResult]:
        ...


class CohereCompatibleAdapter(RerankAdapter):
    """Cohere / Jina / Infinity / vLLM / LiteLLM `POST {endpoint}/rerank` contract (AD-5)."""

    def build_request(self, target, query, documents, top_n) -> PreparedRequest:
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise RerankError("top_n must be an integer >= 1", kind="config")
        headers = {"Content-Type": "application/json"}
        key = (target.api_key or "").strip()
        if key and key != PLACEHOLDER_API_KEY:
            headers["Authorization"] = f"Bearer {key}"
        return PreparedRequest(
            url=rerank_url(target.endpoint),
            headers=headers,
            json={"model": target.model, "query": query, "documents": list(documents), "top_n": top_n},
        )

    def parse_response(self, status_code, body, n_documents, top_n) -> list[RerankResult]:
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


_ADAPTERS: dict[str, RerankAdapter] = {
    RerankProvider.CohereCompatible.value: CohereCompatibleAdapter(),
}


def get_adapter(provider: Optional[str]) -> RerankAdapter:
    adapter = _ADAPTERS.get(provider or "")
    if adapter is None:
        raise UnknownRerankProviderError("Unknown rerank provider")
    return adapter
