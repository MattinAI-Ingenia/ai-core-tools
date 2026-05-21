from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from huggingface_hub import InferenceClient
from models.embedding_service import EmbeddingProvider
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingsAdapter:
    def __init__(self, client):
        self.client = client
        
    def embed_query(self, text):
        result = self.client.feature_extraction(text).tolist()
        return result[0] if isinstance(result[0], list) else result
        
    def embed_documents(self, documents):
        return [self.embed_query(doc) for doc in documents]

def get_embeddings_model(embedding_service):
    """Returns the appropriate embeddings model based on the service configuration.

    Field convention (must match AIService): ``description`` holds the model
    identifier expected by the upstream API (e.g. ``"text-embedding-3-large"``)
    while ``name`` is the human-readable label shown in the UI.
    """
    if embedding_service is None:
        raise ValueError("No embedding service provided")

    logger.info(f"Proveedor {embedding_service.provider}")

    model_id = embedding_service.description

    if embedding_service.provider == EmbeddingProvider.OpenAI.value:
        return OpenAIEmbeddings(
            model=model_id,
            api_key=embedding_service.api_key
        )

    elif embedding_service.provider == EmbeddingProvider.MistralAI.value:
        return MistralAIEmbeddings(
            model=model_id,
            api_key=embedding_service.api_key
        )

    elif embedding_service.provider == EmbeddingProvider.Custom.value:
        client = InferenceClient(
            model=embedding_service.endpoint,
            api_key=embedding_service.api_key
        )
        return HuggingFaceEmbeddingsAdapter(client)
    elif embedding_service.provider == EmbeddingProvider.Ollama.value:
        return OllamaEmbeddings(
            model=model_id,
            base_url=embedding_service.endpoint,
            client_kwargs={
                "verify": False,
                "headers": {
                    "Authorization": f"Bearer {embedding_service.api_key}"
                }
            }
        )
    elif embedding_service.provider == EmbeddingProvider.Azure.value:
        return AzureAIEmbeddingsModel(
            model_name=model_id,
            api_version=embedding_service.api_version or "2024-02-15-preview",
            credential=embedding_service.api_key,
            endpoint=embedding_service.endpoint
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_service.provider}")
