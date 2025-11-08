"""Unit tests for vector store module."""

import pytest
import numpy as np

from src.vector_store import VectorStore


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def vector_store(self):
        """Create a fresh vector store instance for each test."""
        return VectorStore(collection_name="test_collection")

    def test_vector_store_initialization(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store.collection_name == "test_collection"
        assert vector_store.collection is not None

    def test_add_single_chunk(self, vector_store):
        """Test adding a single chunk with embedding."""
        chunk = "Test chunk"
        embedding = np.random.rand(384).tolist()

        vector_store.add_chunks([chunk], [embedding])

        # Verify chunk was added
        results = vector_store.query(embedding, n_results=1)
        assert len(results) > 0

    def test_add_multiple_chunks(self, vector_store, sample_chunks, mock_embeddings):
        """Test adding multiple chunks."""
        vector_store.add_chunks(sample_chunks, mock_embeddings)

        # Query should return results
        query_embedding = mock_embeddings[0]
        results = vector_store.query(query_embedding, n_results=3)

        assert len(results) <= 3
        assert len(results) > 0

    def test_add_chunks_with_metadata(self, vector_store):
        """Test adding chunks with metadata."""
        chunks = ["Chunk 1", "Chunk 2"]
        embeddings = [np.random.rand(384).tolist() for _ in range(2)]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        vector_store.add_chunks(chunks, embeddings, metadatas=metadatas)

        # Query and verify metadata
        results = vector_store.query(embeddings[0], n_results=2)
        assert len(results) > 0

    def test_query_returns_top_results(self, vector_store, sample_chunks, mock_embeddings):
        """Test that query returns top N results."""
        vector_store.add_chunks(sample_chunks, mock_embeddings)

        query_embedding = np.random.rand(384).tolist()
        results = vector_store.query(query_embedding, n_results=2)

        assert len(results) <= 2

    def test_query_empty_store(self, vector_store):
        """Test querying an empty vector store."""
        query_embedding = np.random.rand(384).tolist()
        results = vector_store.query(query_embedding, n_results=3)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_add_empty_chunks(self, vector_store):
        """Test adding empty chunks list."""
        vector_store.add_chunks([], [])

        # Should not raise error
        query_embedding = np.random.rand(384).tolist()
        results = vector_store.query(query_embedding, n_results=1)
        assert len(results) == 0

    def test_query_with_various_n_results(
        self, vector_store, sample_chunks, mock_embeddings
    ):
        """Test querying with different n_results values."""
        vector_store.add_chunks(sample_chunks, mock_embeddings)
        query_embedding = mock_embeddings[0]

        for n in [1, 3, 5, 10]:
            results = vector_store.query(query_embedding, n_results=n)
            assert len(results) <= min(n, len(sample_chunks))

    def test_collection_persistence(self, temp_dir):
        """Test that collection can be persisted."""
        persist_dir = str(temp_dir / "chroma_test")

        # Create store with persistence
        store1 = VectorStore(
            collection_name="persist_test", persist_directory=persist_dir
        )

        chunks = ["Test chunk"]
        embeddings = [np.random.rand(384).tolist()]
        store1.add_chunks(chunks, embeddings)

        # Create new store with same directory
        store2 = VectorStore(
            collection_name="persist_test", persist_directory=persist_dir
        )

        # Should be able to query data
        results = store2.query(embeddings[0], n_results=1)
        assert len(results) > 0

    @pytest.mark.parametrize("n_chunks", [1, 5, 10, 50])
    def test_add_various_chunk_counts(self, vector_store, n_chunks):
        """Test adding various numbers of chunks."""
        chunks = [f"Chunk {i}" for i in range(n_chunks)]
        embeddings = [np.random.rand(384).tolist() for _ in range(n_chunks)]

        vector_store.add_chunks(chunks, embeddings)

        results = vector_store.query(embeddings[0], n_results=min(n_chunks, 3))
        assert len(results) > 0
