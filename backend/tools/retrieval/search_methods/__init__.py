"""
Retrieval search methods package.

A *search method* produces a retriever directly from the corpus (dense vector
search, BM25 lexical search, sparse search, ...). Search methods implement
:class:`tools.retrieval.retrieval_component.SearchMethod`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bm25_search_method import BM25SearchMethod
    from .dense_search_method import DenseSearchMethod

__all__ = ["DenseSearchMethod", "BM25SearchMethod"]
