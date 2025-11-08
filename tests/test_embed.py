"""Unit tests for embedding module."""

import pytest
import numpy as np

from src.embed import EmbeddingModel


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create a reusable embedder instance."""
        return EmbeddingModel()

    def test_embedder_initialization(self, embedder):
        """Test EmbeddingModel initialization."""
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
        assert embedder.dimension == 384

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedder.embed(text)

        assert isinstance(embedding, (list, np.ndarray))
        assert len(embedding) == 384
        assert all(isinstance(x, (float, np.floating)) for x in embedding)

    def test_embed_empty_text(self, embedder):
        """Test embedding empty text."""
        embedding = embedder.embed("")

        assert isinstance(embedding, (list, np.ndarray))
        assert len(embedding) == 384

    def test_embed_batch(self, embedder, sample_chunks):
        """Test embedding multiple texts in batch."""
        embeddings = embedder.embed_batch(sample_chunks)

        assert isinstance(embeddings, (list, np.ndarray))
        assert len(embeddings) == len(sample_chunks)
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_batch_empty_list(self, embedder):
        """Test embedding empty list."""
        embeddings = embedder.embed_batch([])

        assert isinstance(embeddings, (list, np.ndarray))
        assert len(embeddings) == 0

    def test_embed_consistency(self, embedder):
        """Test that same text produces consistent embeddings."""
        text = "Consistency test"
        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)

        embedding1_array = np.array(embedding1)
        embedding2_array = np.array(embedding2)

        # Embeddings should be very similar (cosine similarity near 1)
        cosine_sim = np.dot(embedding1_array, embedding2_array) / (
            np.linalg.norm(embedding1_array) * np.linalg.norm(embedding2_array)
        )
        assert cosine_sim > 0.99

    def test_embed_different_texts(self, embedder):
        """Test that different texts produce different embeddings."""
        text1 = "Machine learning is fascinating"
        text2 = "The weather is sunny today"

        embedding1 = embedder.embed(text1)
        embedding2 = embedder.embed(text2)

        embedding1_array = np.array(embedding1)
        embedding2_array = np.array(embedding2)

        # Different texts should have lower similarity
        cosine_sim = np.dot(embedding1_array, embedding2_array) / (
            np.linalg.norm(embedding1_array) * np.linalg.norm(embedding2_array)
        )
        assert cosine_sim < 0.95

    def test_embedding_normalized(self, embedder):
        """Test that embeddings are properly normalized."""
        text = "Normalization test"
        embedding = embedder.embed(text)
        embedding_array = np.array(embedding)

        norm = np.linalg.norm(embedding_array)
        # Sentence transformers typically normalize embeddings
        assert 0.9 < norm < 1.1

    @pytest.mark.slow
    def test_embed_large_batch(self, embedder):
        """Test embedding a large batch of texts."""
        texts = [f"Test sentence number {i}" for i in range(100)]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)
