"""
Main RAG CLI - PDF Q&A with LLM-powered answers
This script works with ANY LLM provider!
"""

import sys
import os
from pathlib import Path

from .extract import PDFParser
from .chunk import TextChunker
from .embed import EmbeddingModel
from .vector_store import VectorStore
from .rag import RAGInterface
from .pdf_processor import PDFProcessor
from .llm_providers import (
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    HuggingFaceLLM,
    LocalServerLLM,
    MockLLM,
    get_available_llm,
)


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 80)
    print("PDF Q&A System with RAG - Phase 2")
    print("=" * 80 + "\n")


def select_llm_provider():
    """
    Let user select which LLM to use.
    Returns configured LLM instance.
    """
    print("\n📋 Select LLM Provider:")
    print("=" * 80)
    print("1. Auto-detect (recommended)")
    print("2. Ollama (free, local - requires Ollama running)")
    print("3. OpenAI (requires API key)")
    print("4. Anthropic Claude (requires API key)")
    print("5. HuggingFace (free, local - downloads model)")
    print("6. Local Server (vLLM, text-gen-webui, etc.)")
    print("7. Mock (no real LLM, for testing)")
    print("=" * 80)

    choice = input("\nChoice (1-7, default=1): ").strip() or "1"

    if choice == "1":
        print("\n🔍 Auto-detecting available LLM...")
        return get_available_llm()

    elif choice == "2":
        model = input("Ollama model (default=llama3.2): ").strip() or "llama3.2"
        llm = OllamaLLM(model=model)
        if not llm.is_available():
            print("⚠️  Ollama not running. Install from https://ollama.ai")
            print("   Then run: ollama run llama3.2")
            return MockLLM()
        return llm

    elif choice == "3":
        api_key = os.getenv("OPENAI_API_KEY") or input("OpenAI API key: ").strip()
        model = input("Model (default=gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
        return OpenAILLM(model=model, api_key=api_key)

    elif choice == "4":
        api_key = os.getenv("ANTHROPIC_API_KEY") or input("Anthropic API key: ").strip()
        model = input("Model (default=claude-3-sonnet): ").strip() or "claude-3-sonnet-20240229"
        return AnthropicLLM(model=model, api_key=api_key)

    elif choice == "5":
        model = input("HF model (default=Phi-3): ").strip() or "microsoft/Phi-3-mini-4k-instruct"
        return HuggingFaceLLM(model=model)

    elif choice == "6":
        base_url = (
            input("Server URL (default=http://localhost:5000/v1): ").strip()
            or "http://localhost:5000/v1"
        )
        model = input("Model name (default=local-model): ").strip() or "local-model"
        llm = LocalServerLLM(base_url=base_url, model=model)
        if not llm.is_available():
            print(f"⚠️  Cannot connect to {base_url}")
            print("    Make sure your local server is running")
            return MockLLM()
        return llm

    else:
        return MockLLM()


def process_pdf_with_rag(pdf_paths, llm=None):
    """
    Process single or multiple PDFs and start RAG Q&A session.

    Args:
        pdf_paths: Single PDF path (str) or list of PDF paths
        llm: LLM instance (auto-detects if None)
    """
    try:
        print_banner()

        # Convert single path to list for uniform processing
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        # Validate PDFs exist
        for pdf_path in pdf_paths:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Display what we're processing
        if len(pdf_paths) == 1:
            print(f"📄 Processing: {Path(pdf_paths[0]).name}\n")
        else:
            print(f"📄 Processing {len(pdf_paths)} PDF files:\n")
            for i, path in enumerate(pdf_paths, 1):
                print(f"  {i}. {Path(path).name}")
            print()

        # Step 1: Load embedding model
        print("Step 1/3: Loading embedding model...")
        embedding_model = EmbeddingModel()
        print("✓ Embedding model ready\n")

        # Step 2: Initialize vector store
        print("Step 2/3: Initializing vector database...")
        vector_store = VectorStore()
        print("✓ Vector database ready\n")

        # Step 3: Process PDF(s)
        print("Step 3/3: Processing PDF(s)...")
        processor = PDFProcessor(embedding_model=embedding_model, chunk_size=500, chunk_overlap=50)
        stats = processor.process_and_store(pdf_paths, vector_store, show_progress=True)
        print(f"\n✓ Processed {stats['total_pdfs']} PDF(s)\n")

        # Summary
        print("=" * 80)
        print("✅ PDF Processing Complete!")
        print("=" * 80)
        if len(pdf_paths) == 1:
            print(f"Document: {Path(pdf_paths[0]).name}")
        else:
            print(f"Documents: {len(pdf_paths)} PDFs")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Vector DB: {vector_store.get_count()} embeddings stored")
        if len(pdf_paths) > 1:
            print("\nPer-PDF Statistics:")
            for pdf_stat in stats["per_pdf_stats"]:
                print(f"  • {pdf_stat['pdf_name']}: {pdf_stat['num_chunks']} chunks")
        print("=" * 80)

        # Select LLM if not provided
        if llm is None:
            llm = select_llm_provider()

        # Start RAG interface
        print("\n🚀 Starting RAG Q&A System...")
        rag = RAGInterface(
            embedding_model=embedding_model, vector_store=vector_store, llm=llm, n_results=3
        )

        rag.interactive_qa_loop()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def demo_mode(pdf_paths, llm=None):
    """
    Demo mode: Shows one example question/answer without interactive loop.
    Useful for testing!

    Args:
        pdf_paths: Single PDF path (str) or list of PDF paths
        llm: LLM instance (auto-detects if None)
    """
    print_banner()
    print("🎬 DEMO MODE - Testing RAG System\n")

    # Convert single path to list
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    # Process PDF(s)
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    processor = PDFProcessor(embedding_model=embedding_model)

    print(f"Processing {len(pdf_paths)} PDF(s)...")
    stats = processor.process_and_store(pdf_paths, vector_store, show_progress=False)

    print(f"✓ Processed {stats['total_chunks']} chunks from {stats['total_pdfs']} PDF(s)\n")

    # Get LLM
    if llm is None:
        llm = get_available_llm()

    # Create RAG interface
    rag = RAGInterface(embedding_model, vector_store, llm, n_results=3)

    # Ask demo question
    demo_question = "What is this document about?"
    print(f"Demo Question: {demo_question}\n")

    result = rag.answer_question(demo_question, show_context=True)

    print("=" * 80)
    print("💡 ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)

    print("\n📚 CONTEXT USED:")
    print("-" * 80)
    for chunk in result["context"]:
        print(f"\n[Chunk {chunk['rank']}]:")
        preview = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        print(preview)
    print("=" * 80)

    print("\n✅ Demo complete! Run without --demo for interactive mode.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("PDF Q&A System with RAG - Phase 2")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <pdf_file> [pdf_file2 ...] [--demo]")
        print("\nExamples:")
        print(f"  python {sys.argv[0]} document.pdf")
        print(f"  python {sys.argv[0]} doc1.pdf doc2.pdf doc3.pdf")
        print(f"  python {sys.argv[0]} document.pdf --demo")
        print(f"  python {sys.argv[0]} doc1.pdf doc2.pdf --demo")
        print("\nEnvironment Variables:")
        print("  OPENAI_API_KEY     - For OpenAI models")
        print("  ANTHROPIC_API_KEY  - For Anthropic Claude")
        print("\nLocal Options:")
        print("  - Install Ollama: https://ollama.ai")
        print("  - Or use HuggingFace models (auto-downloads)")
        sys.exit(1)

    # Parse arguments
    demo = "--demo" in sys.argv
    pdf_paths = [arg for arg in sys.argv[1:] if arg != "--demo"]

    if not pdf_paths:
        print("Error: No PDF files specified")
        sys.exit(1)

    # Handle single path vs multiple paths
    if len(pdf_paths) == 1:
        pdf_paths = pdf_paths[0]

    if demo:
        demo_mode(pdf_paths)
    else:
        process_pdf_with_rag(pdf_paths)


if __name__ == "__main__":
    main()
