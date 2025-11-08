"""
Tests for PDF Processor Module
Tests single and multiple PDF processing with proper isolation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.embed import EmbeddingModel


class TestPDFProcessor:
    """Test suite for PDFProcessor class."""

    @pytest.fixture
    def mock_pdf_parser(self):
        """Mock PDF parser."""
        with patch("src.pdf_processor.PDFParser") as mock:
            parser_instance = Mock()
            parser_instance.extract_text.return_value = (
                "This is test content. " * 100
            )  # ~2400 chars
            mock.return_value = parser_instance
            yield parser_instance

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        mock = Mock(spec=EmbeddingModel)
        # Return proper numpy arrays for embeddings
        mock.embed_batch.side_effect = lambda chunks, show_progress=True: [
            np.random.rand(384).tolist() for _ in chunks
        ]
        mock.get_embedding_dimension.return_value = 384
        return mock

    @pytest.fixture
    def processor(self, mock_embedding_model):
        """Create processor with mocked embedding model."""
        return PDFProcessor(embedding_model=mock_embedding_model, chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def temp_pdf_file(self, tmp_path):
        """Create a temporary fake PDF file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf content")
        return str(pdf_file)

    @pytest.fixture
    def multiple_temp_pdfs(self, tmp_path):
        """Create multiple temporary fake PDF files."""
        pdf_files = []
        for i in range(3):
            pdf_file = tmp_path / f"test{i+1}.pdf"
            pdf_file.write_text(f"fake pdf content {i+1}")
            pdf_files.append(str(pdf_file))
        return pdf_files

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        processor = PDFProcessor(chunk_size=1000, chunk_overlap=100)
        assert processor is not None
        assert processor.chunker.chunk_size == 1000
        assert processor.chunker.chunk_overlap == 100
        assert processor.embedding_model is not None

    def test_processor_with_custom_components(self, mock_embedding_model):
        """Test processor accepts custom components."""
        from src.chunk import TextChunker

        custom_chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        processor = PDFProcessor(embedding_model=mock_embedding_model, chunker=custom_chunker)

        assert processor.embedding_model == mock_embedding_model
        assert processor.chunker == custom_chunker

    def test_process_single_pdf_success(self, processor, temp_pdf_file):
        """Test successful processing of a single PDF."""
        # Mock the parser's extract_text method directly
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            result = processor.process_pdf(temp_pdf_file, show_progress=False)

        # Verify result structure
        assert "pdf_path" in result
        assert "pdf_name" in result
        assert "chunks" in result
        assert "embeddings" in result
        assert "metadatas" in result
        assert "stats" in result

        # Verify data
        assert result["pdf_name"] == "test.pdf"
        assert len(result["chunks"]) > 0
        assert len(result["embeddings"]) == len(result["chunks"])
        assert len(result["metadatas"]) == len(result["chunks"])

        # Verify metadata contains source info
        for metadata in result["metadatas"]:
            assert "source" in metadata
            assert "source_path" in metadata
            assert "chunk_index" in metadata
            assert metadata["source"] == "test.pdf"

    def test_process_pdf_file_not_found(self, processor):
        """Test processing non-existent PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            processor.process_pdf("/nonexistent/file.pdf")

    def test_process_pdf_empty_text(self, processor, temp_pdf_file):
        """Test processing PDF with no text raises ValueError."""
        with patch.object(processor.parser, "extract_text", return_value=""):
            with pytest.raises(ValueError, match="No text extracted"):
                processor.process_pdf(temp_pdf_file)

    def test_process_pdf_whitespace_only(self, processor, temp_pdf_file):
        """Test processing PDF with only whitespace raises ValueError."""
        with patch.object(processor.parser, "extract_text", return_value="   \n\n  \t  "):
            with pytest.raises(ValueError, match="No text extracted"):
                processor.process_pdf(temp_pdf_file)

    def test_process_multiple_pdfs_success(self, processor, multiple_temp_pdfs):
        """Test successful processing of multiple PDFs."""
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            chunks, embeddings, metadatas, stats = processor.process_multiple_pdfs(
                multiple_temp_pdfs, show_progress=False
            )

        # Verify all data returned
        assert len(chunks) > 0
        assert len(embeddings) == len(chunks)
        assert len(metadatas) == len(chunks)

        # Verify stats structure
        assert stats["total_pdfs"] == 3
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_embeddings"] == len(embeddings)
        assert len(stats["per_pdf_stats"]) == 3

        # Verify each PDF has its own stats
        for pdf_stat in stats["per_pdf_stats"]:
            assert "pdf_name" in pdf_stat
            assert "num_chunks" in pdf_stat
            assert "text_length" in pdf_stat

        # Verify metadata tracks different sources
        sources = {meta["source"] for meta in metadatas}
        assert len(sources) == 3
        assert "test1.pdf" in sources
        assert "test2.pdf" in sources
        assert "test3.pdf" in sources

    def test_process_multiple_pdfs_with_one_missing(self, processor, multiple_temp_pdfs):
        """Test processing multiple PDFs with one missing file."""
        # Put the missing file first to avoid processing valid PDFs before error
        pdf_paths = ["/nonexistent/missing.pdf"] + multiple_temp_pdfs

        with pytest.raises(FileNotFoundError, match="PDF not found"):
            processor.process_multiple_pdfs(pdf_paths, show_progress=False)

    def test_process_and_store_single_pdf(self, processor, temp_pdf_file, tmp_path):
        """Test process_and_store with single PDF."""
        vector_store = VectorStore(collection_name="test_single", persist_directory=str(tmp_path))
        vector_store.clear()

        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            stats = processor.process_and_store(temp_pdf_file, vector_store, show_progress=False)

        # Verify stats
        assert stats["total_pdfs"] == 1
        assert stats["total_chunks"] > 0
        assert stats["total_embeddings"] > 0

        # Verify data stored in vector store
        assert vector_store.get_count() == stats["total_chunks"]

    def test_process_and_store_single_pdf_as_list(self, processor, temp_pdf_file, tmp_path):
        """Test process_and_store with single PDF in a list."""
        vector_store = VectorStore(
            collection_name="test_single_list", persist_directory=str(tmp_path)
        )
        vector_store.clear()

        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            stats = processor.process_and_store([temp_pdf_file], vector_store, show_progress=False)

        # Verify single PDF handling
        assert stats["total_pdfs"] == 1
        assert vector_store.get_count() > 0

    def test_process_and_store_multiple_pdfs(self, processor, multiple_temp_pdfs, tmp_path):
        """Test process_and_store with multiple PDFs."""
        vector_store = VectorStore(collection_name="test_multiple", persist_directory=str(tmp_path))
        vector_store.clear()

        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            stats = processor.process_and_store(
                multiple_temp_pdfs, vector_store, show_progress=False
            )

        # Verify stats
        assert stats["total_pdfs"] == 3
        assert stats["total_chunks"] > 0
        assert len(stats["per_pdf_stats"]) == 3

        # Verify all chunks stored
        assert vector_store.get_count() == stats["total_chunks"]

        # Verify per-PDF stats sum to total
        total_chunks_per_pdf = sum(pdf["num_chunks"] for pdf in stats["per_pdf_stats"])
        assert total_chunks_per_pdf == stats["total_chunks"]

    def test_process_and_store_empty_list(self, processor, tmp_path):
        """Test process_and_store with empty PDF list raises ValueError."""
        vector_store = VectorStore(collection_name="test_empty", persist_directory=str(tmp_path))

        with pytest.raises(ValueError, match="No PDF paths provided"):
            processor.process_and_store([], vector_store)

    def test_process_and_store_none(self, processor, tmp_path):
        """Test process_and_store with None raises ValueError."""
        vector_store = VectorStore(collection_name="test_none", persist_directory=str(tmp_path))

        # None should be treated as empty and raise ValueError
        with pytest.raises((ValueError, TypeError)):
            processor.process_and_store(None, vector_store)

    def test_metadata_preservation_through_pipeline(self, processor, temp_pdf_file, tmp_path):
        """Test that metadata is preserved through entire pipeline."""
        vector_store = VectorStore(collection_name="test_metadata", persist_directory=str(tmp_path))
        vector_store.clear()

        # Process and store
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            processor.process_and_store(temp_pdf_file, vector_store, show_progress=False)

        # Search to retrieve stored chunks
        query_embedding = np.random.rand(384)
        results = vector_store.search(query_embedding, n_results=1)

        # Verify metadata exists in stored chunks
        assert len(results["metadatas"][0]) > 0
        metadata = results["metadatas"][0][0]
        assert "source" in metadata
        assert "chunk_index" in metadata

    def test_chunk_statistics_accuracy(self, processor, temp_pdf_file):
        """Test that chunk statistics are calculated correctly."""
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            result = processor.process_pdf(temp_pdf_file, show_progress=False)
            stats = result["stats"]

            # Verify stats match actual data
            assert stats["num_chunks"] == len(result["chunks"])
            assert stats["text_length"] > 0

            # Verify average calculation
            expected_avg = stats["text_length"] / stats["num_chunks"]
            assert abs(stats["avg_chunk_size"] - expected_avg) < 0.01

    def test_show_progress_parameter(self, processor, temp_pdf_file):
        """Test that show_progress parameter is passed correctly."""
        # Process with show_progress=True
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            result1 = processor.process_pdf(temp_pdf_file, show_progress=True)
            assert result1 is not None

            # Process with show_progress=False
            result2 = processor.process_pdf(temp_pdf_file, show_progress=False)
            assert result2 is not None

            # Both should produce same structure
            assert result1.keys() == result2.keys()

    def test_multiple_pdfs_different_sizes(self, processor, tmp_path):
        """Test processing PDFs of different sizes."""
        # Create PDFs that will result in different chunk counts
        pdf_files = []

        # Small PDF (short text)
        small_pdf = tmp_path / "small.pdf"
        small_pdf.write_text("small")
        pdf_files.append(str(small_pdf))
        small_text = "Small content. " * 10

        # Large PDF (long text)
        large_pdf = tmp_path / "large.pdf"
        large_pdf.write_text("large")
        pdf_files.append(str(large_pdf))
        large_text = "Large content. " * 500

        # Mock different text lengths
        with patch.object(processor.parser, "extract_text", side_effect=[small_text, large_text]):
            chunks, embeddings, metadatas, stats = processor.process_multiple_pdfs(
                pdf_files, show_progress=False
            )

        # Verify both PDFs processed
        assert stats["total_pdfs"] == 2
        assert len(stats["per_pdf_stats"]) == 2

        # Verify different chunk counts
        chunk_counts = [pdf["num_chunks"] for pdf in stats["per_pdf_stats"]]
        assert chunk_counts[0] < chunk_counts[1]  # Small PDF has fewer chunks

    def test_backward_compatibility_single_string_path(self, processor, temp_pdf_file, tmp_path):
        """Test backward compatibility: single PDF path as string still works."""
        vector_store = VectorStore(collection_name="test_compat", persist_directory=str(tmp_path))
        vector_store.clear()

        # Should accept string path (not just list)
        with patch.object(
            processor.parser, "extract_text", return_value="This is test content. " * 100
        ):
            stats = processor.process_and_store(temp_pdf_file, vector_store, show_progress=False)

            assert stats["total_pdfs"] == 1
            assert vector_store.get_count() > 0
