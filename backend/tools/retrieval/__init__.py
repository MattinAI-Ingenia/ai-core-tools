"""
Retrieval strategy package.

This package contains implementations of the RetrievalTransformer contract,
which wrap one or more retrievers with additional retrieval logic.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .strategies.rerank_strategy import RerankStrategy

__all__ = ['RerankStrategy']
