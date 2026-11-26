from types import SimpleNamespace

from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from tools.embeddingTools import get_embeddings_model


def _service(provider, endpoint, model="some-model", api_key="key"):
    return SimpleNamespace(provider=provider, description=model, api_key=api_key, endpoint=endpoint, api_version=None)


def test_custom_provider_is_a_langchain_embeddings_usable_by_rerank_filter():
    embeddings = get_embeddings_model(_service("Custom", "http://127.0.0.1:8765/embed"))

    assert isinstance(embeddings, Embeddings)
    # The embeddings-based rerank strategy wraps the silo embeddings in this filter; it rejected the adapter before.
    EmbeddingsFilter(embeddings=embeddings, k=3)


def test_openai_compatible_provider_targets_the_configured_server():
    embeddings = get_embeddings_model(_service("OpenAICompatible", "http://infinity:7998", model="BAAI/bge-m3", api_key=None))

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "BAAI/bge-m3"
    assert embeddings.openai_api_base == "http://infinity:7998"
    # Non-OpenAI servers expect raw text, not tiktoken ids.
    assert embeddings.check_embedding_ctx_length is False
    EmbeddingsFilter(embeddings=embeddings, k=3)
