"""
PDF Processor Module
Handles processing of single or multiple PDF files with metadata tracking.
Follows Single Responsibility Principle - separates PDF processing logic from CLI.
"""

from typing import List, Dict, Tuple
from pathlib import Path
from .extract import PDFParser
from .chunk import TextChunker
from .embed import EmbeddingModel
from .vector_store import VectorStore


class PDFProcessor:
    """
    Processes PDF files and stores them in a vector database.
    Supports both single and multiple PDF processing with source tracking.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        chunker: TextChunker = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize PDF processor.

        Args:
            embedding_model: Embedding model instance (creates new if None)
            chunker: Text chunker instance (creates new if None)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.parser = PDFParser()
        self.chunker = chunker or TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = embedding_model or EmbeddingModel()

    def process_pdf(self, pdf_path: str, show_progress: bool = True) -> Dict:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file
            show_progress: Whether to show progress during embedding

        Returns:
            Dictionary with processed data:
            {
                'pdf_path': str,
                'pdf_name': str,
                'chunks': List[str],
                'embeddings': List,
                'metadatas': List[Dict],
                'stats': Dict
            }

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF extraction fails
        """
        pdf_file = Path(pdf_path)

        # Validate file exists
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Extract text
        text = self.parser.extract_text(str(pdf_file))
        if not text or not text.strip():
            raise ValueError(f"No text extracted from PDF: {pdf_path}")

        # Create chunks with metadata
        chunk_data = self.chunker.chunk_with_metadata(
            text, source_metadata={"source": pdf_file.name, "source_path": str(pdf_file.absolute())}
        )

        # Extract chunks and metadatas
        chunks = [item["text"] for item in chunk_data]
        metadatas = [item["metadata"] for item in chunk_data]

        # Generate embeddings
        embeddings = self.embedding_model.embed_batch(chunks, show_progress=show_progress)

        return {
            "pdf_path": str(pdf_file.absolute()),
            "pdf_name": pdf_file.name,
            "chunks": chunks,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "stats": {
                "num_chunks": len(chunks),
                "text_length": len(text),
                "avg_chunk_size": len(text) / len(chunks) if chunks else 0,
            },
        }

    def process_multiple_pdfs(
        self, pdf_paths: List[str], show_progress: bool = True
    ) -> Tuple[List[str], List, List[Dict], Dict]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files
            show_progress: Whether to show progress

        Returns:
            Tuple of (all_chunks, all_embeddings, all_metadatas, summary_stats)

        Raises:
            FileNotFoundError: If any PDF file doesn't exist
            ValueError: If any PDF extraction fails
        """
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        pdf_stats = []

        for idx, pdf_path in enumerate(pdf_paths, 1):
            if show_progress:
                print(f"\n[{idx}/{len(pdf_paths)}] Processing: {Path(pdf_path).name}")

            result = self.process_pdf(pdf_path, show_progress=show_progress)

            all_chunks.extend(result["chunks"])
            all_embeddings.extend(result["embeddings"])
            all_metadatas.extend(result["metadatas"])
            pdf_stats.append(
                {
                    "pdf_name": result["pdf_name"],
                    "num_chunks": result["stats"]["num_chunks"],
                    "text_length": result["stats"]["text_length"],
                }
            )

        summary_stats = {
            "total_pdfs": len(pdf_paths),
            "total_chunks": len(all_chunks),
            "total_embeddings": len(all_embeddings),
            "per_pdf_stats": pdf_stats,
        }

        return all_chunks, all_embeddings, all_metadatas, summary_stats

    def process_and_store(
        self, pdf_paths: List[str], vector_store: VectorStore, show_progress: bool = True
    ) -> Dict:
        """
        Process PDF(s) and store in vector database.

        Args:
            pdf_paths: Single path or list of paths to PDF files
            vector_store: Vector store instance to store chunks
            show_progress: Whether to show progress

        Returns:
            Dictionary with processing statistics
        """
        # Handle single PDF path
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        # Validate input
        if not pdf_paths:
            raise ValueError("No PDF paths provided")

        # Process PDFs
        if len(pdf_paths) == 1:
            # Single PDF processing
            result = self.process_pdf(pdf_paths[0], show_progress=show_progress)
            chunks = result["chunks"]
            embeddings = result["embeddings"]
            metadatas = result["metadatas"]
            stats = {
                "total_pdfs": 1,
                "total_chunks": result["stats"]["num_chunks"],
                "total_embeddings": len(embeddings),
                "per_pdf_stats": [
                    {
                        "pdf_name": result["pdf_name"],
                        "num_chunks": result["stats"]["num_chunks"],
                        "text_length": result["stats"]["text_length"],
                    }
                ],
            }
        else:
            # Multiple PDF processing
            chunks, embeddings, metadatas, stats = self.process_multiple_pdfs(
                pdf_paths, show_progress=show_progress
            )

        # Store in vector database
        vector_store.add_chunks(chunks, embeddings, metadatas)

        return stats
