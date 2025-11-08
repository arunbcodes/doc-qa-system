"""Pytest configuration and fixtures for PDF Q&A System tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Returns the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text() -> str:
    """Returns sample text for testing."""
    return """
    PDF Q&A System Documentation

    This is a production-ready PDF question-answering system with semantic search.
    The system supports multiple LLM providers including OpenAI, Anthropic, and Ollama.

    Key Features:
    - Semantic search using sentence transformers
    - RAG (Retrieval Augmented Generation) pipeline
    - Model-agnostic LLM integration
    - Vector database for efficient retrieval

    Installation:
    pip install pdf-qa-system

    Usage:
    python main.py data/sample.pdf
    """


@pytest.fixture
def sample_chunks() -> list[str]:
    """Returns sample text chunks for testing."""
    return [
        "PDF Q&A System is a production-ready question-answering system.",
        "The system supports semantic search using sentence transformers.",
        "RAG pipeline enables LLM-powered answers from documents.",
        "Multiple LLM providers are supported: OpenAI, Anthropic, Ollama.",
        "Vector database enables efficient document retrieval.",
    ]


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Returns mock embeddings for testing (384 dimensions)."""
    import numpy as np

    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in range(5)]


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    env_vars_to_clean = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
    ]

    original_env = {}
    for var in env_vars_to_clean:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original environment
    for var, value in original_env.items():
        os.environ[var] = value
