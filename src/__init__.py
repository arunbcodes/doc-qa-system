"""
PDF Q&A System - Core Package
Provides document processing, embedding, and RAG capabilities.
"""

from .extract import PDFParser
from .chunk import TextChunker
from .embed import EmbeddingModel
from .vector_store import VectorStore
from .query import QueryInterface
from .llm_providers import (
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    HuggingFaceLLM,
    LocalServerLLM,
    MockLLM,
    get_available_llm
)
from .rag import RAGInterface

__version__ = "2.0.0"

__all__ = [
    "PDFParser",
    "TextChunker",
    "EmbeddingModel",
    "VectorStore",
    "QueryInterface",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "HuggingFaceLLM",
    "LocalServerLLM",
    "MockLLM",
    "get_available_llm",
    "RAGInterface",
]

