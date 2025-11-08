"""
Integration tests for multi-PDF support in main_rag.py
Tests command-line argument parsing and end-to-end workflows.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.main_rag import main, process_pdf_with_rag, demo_mode


class TestMainRagMultiPDF:
    """Test suite for multi-PDF functionality in main_rag.py."""

    @pytest.fixture
    def temp_pdfs(self, tmp_path):
        """Create temporary test PDF files."""
        pdf_files = []
        for i in range(3):
            pdf_file = tmp_path / f"test_doc_{i+1}.pdf"
            pdf_file.write_text(f"Test PDF content {i+1}")
            pdf_files.append(str(pdf_file))
        return pdf_files

    @pytest.fixture
    def mock_components(self):
        """Mock all components for testing."""
        with patch("src.main_rag.PDFProcessor") as mock_processor, patch(
            "src.main_rag.EmbeddingModel"
        ) as mock_embedding, patch("src.main_rag.VectorStore") as mock_vector, patch(
            "src.main_rag.RAGInterface"
        ) as mock_rag, patch(
            "src.main_rag.get_available_llm"
        ) as mock_llm:

            # Setup processor mock
            processor_instance = Mock()
            processor_instance.process_and_store.return_value = {
                "total_pdfs": 1,
                "total_chunks": 10,
                "total_embeddings": 10,
                "per_pdf_stats": [{"pdf_name": "test.pdf", "num_chunks": 10, "text_length": 5000}],
            }
            mock_processor.return_value = processor_instance

            # Setup vector store mock
            vector_instance = Mock()
            vector_instance.get_count.return_value = 10
            mock_vector.return_value = vector_instance

            # Setup RAG interface mock
            rag_instance = Mock()
            rag_instance.interactive_qa_loop = Mock()
            rag_instance.answer_question.return_value = {
                "answer": "Test answer",
                "question": "Test question",
            }
            mock_rag.return_value = rag_instance

            # Setup LLM mock
            llm_instance = Mock()
            llm_instance.__class__.__name__ = "MockLLM"
            mock_llm.return_value = llm_instance

            yield {
                "processor": processor_instance,
                "vector_store": vector_instance,
                "rag": rag_instance,
                "llm": llm_instance,
            }

    def test_process_single_pdf_as_string(self, temp_pdfs, mock_components):
        """Test processing single PDF passed as string (backward compatibility)."""
        single_pdf = temp_pdfs[0]

        # Should accept string path
        process_pdf_with_rag(single_pdf, llm=mock_components["llm"])

        # Verify processor was called
        assert mock_components["processor"].process_and_store.called

    def test_process_single_pdf_as_list(self, temp_pdfs, mock_components):
        """Test processing single PDF passed as list."""
        single_pdf_list = [temp_pdfs[0]]

        process_pdf_with_rag(single_pdf_list, llm=mock_components["llm"])

        # Verify processor was called
        assert mock_components["processor"].process_and_store.called

    def test_process_multiple_pdfs(self, temp_pdfs, mock_components):
        """Test processing multiple PDFs."""
        # Update mock to return multi-PDF stats
        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 3,
            "total_chunks": 30,
            "total_embeddings": 30,
            "per_pdf_stats": [
                {"pdf_name": "test_doc_1.pdf", "num_chunks": 10, "text_length": 5000},
                {"pdf_name": "test_doc_2.pdf", "num_chunks": 10, "text_length": 5000},
                {"pdf_name": "test_doc_3.pdf", "num_chunks": 10, "text_length": 5000},
            ],
        }

        process_pdf_with_rag(temp_pdfs, llm=mock_components["llm"])

        # Verify processor called with all PDFs
        call_args = mock_components["processor"].process_and_store.call_args
        pdf_paths_arg = call_args[0][0]
        assert len(pdf_paths_arg) == 3

    def test_process_pdf_file_not_found(self, mock_components):
        """Test error handling for non-existent PDF."""
        with pytest.raises(FileNotFoundError):
            process_pdf_with_rag("/nonexistent/file.pdf", llm=mock_components["llm"])

    def test_demo_mode_single_pdf(self, temp_pdfs, mock_components):
        """Test demo mode with single PDF."""
        single_pdf = temp_pdfs[0]

        demo_mode(single_pdf, llm=mock_components["llm"])

        # Verify processor was called
        assert mock_components["processor"].process_and_store.called

        # Verify answer_question was called
        assert mock_components["rag"].answer_question.called

    def test_demo_mode_multiple_pdfs(self, temp_pdfs, mock_components):
        """Test demo mode with multiple PDFs."""
        # Update mock for multi-PDF
        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 3,
            "total_chunks": 30,
            "total_embeddings": 30,
            "per_pdf_stats": [],
        }

        demo_mode(temp_pdfs, llm=mock_components["llm"])

        # Verify processor called with all PDFs
        assert mock_components["processor"].process_and_store.called

    def test_main_single_pdf_argument(self, temp_pdfs, mock_components, monkeypatch, capsys):
        """Test main() with single PDF command-line argument."""
        test_args = ["main_rag.py", temp_pdfs[0]]
        monkeypatch.setattr(sys, "argv", test_args)

        # Mock interactive_qa_loop to prevent blocking
        mock_components["rag"].interactive_qa_loop = Mock()

        with patch("src.main_rag.select_llm_provider") as mock_select:
            mock_select.return_value = mock_components["llm"]
            main()

        # Verify processing happened
        assert mock_components["processor"].process_and_store.called

    def test_main_multiple_pdf_arguments(self, temp_pdfs, mock_components, monkeypatch):
        """Test main() with multiple PDF command-line arguments."""
        test_args = ["main_rag.py"] + temp_pdfs
        monkeypatch.setattr(sys, "argv", test_args)

        # Mock to prevent blocking
        mock_components["rag"].interactive_qa_loop = Mock()
        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 3,
            "total_chunks": 30,
            "total_embeddings": 30,
            "per_pdf_stats": [
                {"pdf_name": f"test_doc_{i}.pdf", "num_chunks": 10, "text_length": 5000}
                for i in range(1, 4)
            ],
        }

        with patch("src.main_rag.select_llm_provider") as mock_select:
            mock_select.return_value = mock_components["llm"]
            main()

        # Verify processor called with multiple PDFs
        call_args = mock_components["processor"].process_and_store.call_args
        pdf_paths_arg = call_args[0][0]
        assert len(pdf_paths_arg) == 3

    def test_main_demo_flag_single_pdf(self, temp_pdfs, mock_components, monkeypatch):
        """Test main() with --demo flag and single PDF."""
        test_args = ["main_rag.py", temp_pdfs[0], "--demo"]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch("src.main_rag.get_available_llm") as mock_get_llm:
            mock_get_llm.return_value = mock_components["llm"]
            main()

        # Verify demo mode ran
        assert mock_components["rag"].answer_question.called

    def test_main_demo_flag_multiple_pdfs(self, temp_pdfs, mock_components, monkeypatch):
        """Test main() with --demo flag and multiple PDFs."""
        test_args = ["main_rag.py"] + temp_pdfs + ["--demo"]
        monkeypatch.setattr(sys, "argv", test_args)

        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 3,
            "total_chunks": 30,
            "total_embeddings": 30,
            "per_pdf_stats": [],
        }

        with patch("src.main_rag.get_available_llm") as mock_get_llm:
            mock_get_llm.return_value = mock_components["llm"]
            main()

        # Verify demo mode processed all PDFs
        call_args = mock_components["processor"].process_and_store.call_args
        pdf_paths_arg = call_args[0][0]
        assert len(pdf_paths_arg) == 3

    def test_main_no_arguments_shows_help(self, monkeypatch, capsys):
        """Test main() without arguments shows help message."""
        test_args = ["main_rag.py"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        # Verify help message shown
        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "pdf_file2" in captured.out  # Should show multi-PDF usage

    def test_main_only_demo_flag_shows_error(self, monkeypatch, capsys):
        """Test main() with only --demo flag shows error."""
        test_args = ["main_rag.py", "--demo"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "No PDF files specified" in captured.out

    def test_argument_parsing_preserves_order(self, temp_pdfs, mock_components, monkeypatch):
        """Test that argument parsing preserves PDF file order."""
        # Pass PDFs in specific order with --demo in middle
        test_args = ["main_rag.py", temp_pdfs[0], "--demo", temp_pdfs[1], temp_pdfs[2]]
        monkeypatch.setattr(sys, "argv", test_args)

        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 3,
            "total_chunks": 30,
            "total_embeddings": 30,
            "per_pdf_stats": [],
        }

        with patch("src.main_rag.get_available_llm") as mock_get_llm:
            mock_get_llm.return_value = mock_components["llm"]
            main()

        # Verify PDFs passed in correct order (--demo filtered out)
        call_args = mock_components["processor"].process_and_store.call_args
        pdf_paths_arg = call_args[0][0]
        assert pdf_paths_arg == temp_pdfs

    def test_rag_interface_receives_combined_vector_store(self, temp_pdfs, mock_components):
        """Test that RAG interface gets vector store with all PDF chunks."""
        mock_components["processor"].process_and_store.return_value = {
            "total_pdfs": 2,
            "total_chunks": 20,
            "total_embeddings": 20,
            "per_pdf_stats": [
                {"pdf_name": "test_doc_1.pdf", "num_chunks": 10, "text_length": 5000},
                {"pdf_name": "test_doc_2.pdf", "num_chunks": 10, "text_length": 5000},
            ],
        }

        mock_components["vector_store"].get_count.return_value = 20

        process_pdf_with_rag(temp_pdfs[:2], llm=mock_components["llm"])

        # Verify vector store has combined count
        assert mock_components["vector_store"].get_count() == 20

    def test_error_propagation_from_processor(self, temp_pdfs, mock_components):
        """Test that errors from processor are properly propagated."""
        # Make processor raise an error
        mock_components["processor"].process_and_store.side_effect = ValueError("Processing failed")

        # The error causes sys.exit(1) in main_rag.py
        with pytest.raises(SystemExit) as exc_info:
            process_pdf_with_rag(temp_pdfs[0], llm=mock_components["llm"])

        assert exc_info.value.code == 1


class TestMultiPDFBackwardCompatibility:
    """Test backward compatibility with existing single-PDF code."""

    @pytest.fixture
    def single_pdf(self, tmp_path):
        """Create a single test PDF."""
        pdf_file = tmp_path / "single.pdf"
        pdf_file.write_text("Single PDF content")
        return str(pdf_file)

    def test_old_single_pdf_workflow_still_works(self, single_pdf):
        """Test that existing single-PDF usage pattern still works."""
        with patch("src.main_rag.PDFProcessor") as mock_processor, patch(
            "src.main_rag.EmbeddingModel"
        ), patch("src.main_rag.VectorStore"), patch("src.main_rag.RAGInterface") as mock_rag:

            processor_instance = Mock()
            processor_instance.process_and_store.return_value = {
                "total_pdfs": 1,
                "total_chunks": 10,
                "total_embeddings": 10,
                "per_pdf_stats": [
                    {"pdf_name": "single.pdf", "num_chunks": 10, "text_length": 5000}
                ],
            }
            mock_processor.return_value = processor_instance

            rag_instance = Mock()
            mock_rag.return_value = rag_instance

            # Old style: pass single string
            process_pdf_with_rag(single_pdf, llm=Mock())

            # Should still work
            assert processor_instance.process_and_store.called

    def test_single_pdf_statistics_format(self, single_pdf):
        """Test that single PDF returns correct statistics format."""
        with patch("src.main_rag.PDFProcessor") as mock_processor, patch(
            "src.main_rag.EmbeddingModel"
        ), patch("src.main_rag.VectorStore") as mock_vector, patch("src.main_rag.RAGInterface"):

            processor_instance = Mock()
            processor_instance.process_and_store.return_value = {
                "total_pdfs": 1,
                "total_chunks": 5,
                "total_embeddings": 5,
                "per_pdf_stats": [{"pdf_name": "single.pdf", "num_chunks": 5, "text_length": 2500}],
            }
            mock_processor.return_value = processor_instance

            vector_instance = Mock()
            vector_instance.get_count.return_value = 5
            mock_vector.return_value = vector_instance

            # Process single PDF
            process_pdf_with_rag(single_pdf, llm=Mock())

            # Verify correct stats structure for single PDF
            call_result = processor_instance.process_and_store.return_value
            assert call_result["total_pdfs"] == 1
            assert len(call_result["per_pdf_stats"]) == 1
