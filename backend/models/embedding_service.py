import enum
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from models.base_service import BaseService

class EmbeddingProvider(enum.Enum):
    OpenAI = "OpenAI"
    # Any self-hosted server speaking OpenAI's /embeddings protocol (Infinity,
    # TEI, vLLM, LiteLLM). Same builder as OpenAI — it only differs in carrying
    # an endpoint. A distinct value, not a reuse of "OpenAI", because the wizard
    # resolves its provider card by value and two cards sharing one value would
    # silently collapse onto the first (hiding the base-URL field).
    # "Custom" was not available: for embeddings it means HuggingFace's
    # Inference protocol, and "Ollama" means Ollama's own — neither of which
    # such a server speaks. The column is free-form TEXT, so no migration.
    OpenAICompatible = "OpenAICompatible"
    MistralAI = "MistralAI"
    Ollama = "Ollama"
    Custom = "Custom"
    Azure = "Azure"
    Google = "Google"
    GoogleCloud = "GoogleCloud"

class EmbeddingService(BaseService):
    __tablename__ = 'embedding_service'
    
    provider = Column(String(45), nullable=False)
    app_id = Column(Integer, ForeignKey('App.app_id'), nullable=True)  # NULL = system/platform service
    app = relationship('App', back_populates='embedding_services') 