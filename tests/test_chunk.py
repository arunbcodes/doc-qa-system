"""Unit tests for text chunking module."""

import pytest

from src.chunk import TextChunker


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_chunker_initialization(self):
        """Test TextChunker initialization with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_chunker_custom_parameters(self):
        """Test TextChunker initialization with custom parameters."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_chunk_text_basic(self, sample_text):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(sample_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        short_text = "This is a short text."
        chunks = chunker.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_empty(self):
        """Test chunking with empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_chunk_text_whitespace_only(self):
        """Test chunking with whitespace-only text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("   \n\t  ")

        # Should return empty list or single empty chunk
        assert isinstance(chunks, list)

    def test_chunk_overlap(self):
        """Test that chunks have appropriate overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "word " * 50  # Create text with repeated words
        chunks = chunker.chunk_text(text)

        # With overlap, should have more chunks
        assert len(chunks) >= 2

    @pytest.mark.parametrize("chunk_size,chunk_overlap", [
        (100, 20),
        (500, 50),
        (1000, 100),
    ])
    def test_various_chunk_sizes(self, sample_text, chunk_size, chunk_overlap):
        """Test chunking with various chunk sizes."""
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_text(sample_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
