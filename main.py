"""
Main CLI for PDF Q&A System
Orchestrates PDF processing, embedding, and interactive querying.
"""

import sys
import os
from pathlib import Path

from pdf_parser import PDFParser
from text_chunker import TextChunker
from embeddings import EmbeddingModel
from vector_store import VectorStore
from query_interface import QueryInterface


def print_banner():
    """Print application banner."""
    print("\n" + "="*80)
    print("PDF Q&A System - Phase 1 MVP")
    print("="*80 + "\n")


def validate_pdf_path(pdf_path: str) -> Path:
    """
    Validate that the PDF file exists.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    return path


def process_pdf(pdf_path: str):
    """
    Main processing pipeline for PDF Q&A system.
    
    Args:
        pdf_path: Path to the PDF file
    """
    try:
        print_banner()
        
        # Validate PDF path
        pdf_file = validate_pdf_path(pdf_path)
        print(f"Processing PDF: {pdf_file.name}")
        print(f"Full path: {pdf_file.absolute()}\n")
        
        # Step 1: Initialize components
        print("Step 1/5: Initializing components...")
        parser = PDFParser()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        embedding_model = EmbeddingModel()
        vector_store = VectorStore()
        
        # Step 2: Extract text from PDF
        print(f"\nStep 2/5: Extracting text from PDF...")
        extracted_data = parser.extract_with_metadata(str(pdf_file))
        text = extracted_data['text']
        metadata = extracted_data['metadata']
        
        print(f"✓ Extracted {len(text)} characters")
        if metadata.get('num_pages'):
            print(f"✓ Document has {metadata['num_pages']} pages")
        
        # Step 3: Chunk the text
        print(f"\nStep 3/5: Chunking text...")
        chunks_with_meta = chunker.chunk_with_metadata(text, metadata)
        chunks = [c['text'] for c in chunks_with_meta]
        chunk_metadatas = [c['metadata'] for c in chunks_with_meta]
        
        stats = chunker.get_stats(text)
        print(f"✓ Created {stats['num_chunks']} chunks")
        print(f"✓ Average chunk size: {stats['avg_chunk_size']:.0f} characters")
        
        # Step 4: Generate embeddings
        print(f"\nStep 4/5: Generating embeddings...")
        embeddings = embedding_model.embed_batch(chunks, show_progress=True)
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"✓ Embedding dimension: {embedding_model.get_embedding_dimension()}")
        
        # Step 5: Store in vector database
        print(f"\nStep 5/5: Storing in vector database...")
        vector_store.add_chunks(chunks, embeddings, chunk_metadatas)
        print(f"✓ Stored {vector_store.get_count()} chunks in vector database")
        
        # Summary
        print("\n" + "="*80)
        print("PDF Processing Complete!")
        print("="*80)
        print(f"Document: {pdf_file.name}")
        print(f"Chunks: {len(chunks)}")
        print(f"Ready for querying")
        print("="*80 + "\n")
        
        # Step 6: Enter interactive query mode
        query_interface = QueryInterface(embedding_model, vector_store, n_results=3)
        query_interface.interactive_query_loop()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("PDF Q&A System - Phase 1 MVP")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_pdf>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    process_pdf(pdf_path)


if __name__ == "__main__":
    main()

