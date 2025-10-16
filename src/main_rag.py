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
from .llm_providers import (
    OpenAILLM, AnthropicLLM, OllamaLLM, 
    HuggingFaceLLM, LocalServerLLM, MockLLM, get_available_llm
)


def print_banner():
    """Print application banner."""
    print("\n" + "="*80)
    print("PDF Q&A System with RAG - Phase 2")
    print("="*80 + "\n")


def select_llm_provider():
    """
    Let user select which LLM to use.
    Returns configured LLM instance.
    """
    print("\n📋 Select LLM Provider:")
    print("="*80)
    print("1. Auto-detect (recommended)")
    print("2. Ollama (free, local - requires Ollama running)")
    print("3. OpenAI (requires API key)")
    print("4. Anthropic Claude (requires API key)")
    print("5. HuggingFace (free, local - downloads model)")
    print("6. Local Server (vLLM, text-gen-webui, etc.)")
    print("7. Mock (no real LLM, for testing)")
    print("="*80)
    
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
        base_url = input("Server URL (default=http://localhost:5000/v1): ").strip() or "http://localhost:5000/v1"
        model = input("Model name (default=local-model): ").strip() or "local-model"
        llm = LocalServerLLM(base_url=base_url, model=model)
        if not llm.is_available():
            print(f"⚠️  Cannot connect to {base_url}")
            print("    Make sure your local server is running")
            return MockLLM()
        return llm
    
    else:
        return MockLLM()


def process_pdf_with_rag(pdf_path: str, llm=None):
    """
    Process PDF and start RAG Q&A session.
    
    Args:
        pdf_path: Path to PDF file
        llm: LLM instance (auto-detects if None)
    """
    try:
        print_banner()
        
        # Validate PDF
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"📄 Processing: {pdf_file.name}\n")
        
        # Step 1: Parse PDF
        print("Step 1/5: Extracting text from PDF...")
        parser = PDFParser()
        text = parser.extract_text(str(pdf_file))
        print(f"✓ Extracted {len(text)} characters\n")
        
        # Step 2: Chunk text
        print("Step 2/5: Chunking text...")
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text(text)
        print(f"✓ Created {len(chunks)} chunks\n")
        
        # Step 3: Load embedding model
        print("Step 3/5: Loading embedding model...")
        embedding_model = EmbeddingModel()
        print("✓ Embedding model ready\n")
        
        # Step 4: Generate embeddings
        print("Step 4/5: Generating embeddings...")
        embeddings = embedding_model.embed_batch(chunks, show_progress=True)
        print(f"✓ Generated {len(embeddings)} embeddings\n")
        
        # Step 5: Store in vector DB
        print("Step 5/5: Storing in vector database...")
        vector_store = VectorStore()
        vector_store.add_chunks(chunks, embeddings)
        print(f"✓ Stored {vector_store.get_count()} chunks\n")
        
        # Summary
        print("="*80)
        print("✅ PDF Processing Complete!")
        print("="*80)
        print(f"Document: {pdf_file.name}")
        print(f"Chunks: {len(chunks)}")
        print(f"Vector DB: {vector_store.get_count()} embeddings stored")
        print("="*80)
        
        # Select LLM if not provided
        if llm is None:
            llm = select_llm_provider()
        
        # Start RAG interface
        print("\n🚀 Starting RAG Q&A System...")
        rag = RAGInterface(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm=llm,
            n_results=3
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


def demo_mode(pdf_path: str, llm=None):
    """
    Demo mode: Shows one example question/answer without interactive loop.
    Useful for testing!
    """
    print_banner()
    print("🎬 DEMO MODE - Testing RAG System\n")
    
    # Process PDF
    pdf_file = Path(pdf_path)
    parser = PDFParser()
    chunker = TextChunker()
    embedding_model = EmbeddingModel()
    
    print("Processing PDF...")
    text = parser.extract_text(str(pdf_file))
    chunks = chunker.chunk_text(text)
    embeddings = embedding_model.embed_batch(chunks, show_progress=False)
    
    vector_store = VectorStore()
    vector_store.add_chunks(chunks, embeddings)
    
    print(f"✓ Processed {len(chunks)} chunks\n")
    
    # Get LLM
    if llm is None:
        llm = get_available_llm()
    
    # Create RAG interface
    rag = RAGInterface(embedding_model, vector_store, llm, n_results=3)
    
    # Ask demo question
    demo_question = "What is this document about?"
    print(f"Demo Question: {demo_question}\n")
    
    result = rag.answer_question(demo_question, show_context=True)
    
    print("="*80)
    print("💡 ANSWER:")
    print("="*80)
    print(result['answer'])
    print("="*80)
    
    print("\n📚 CONTEXT USED:")
    print("-"*80)
    for chunk in result['context']:
        print(f"\n[Chunk {chunk['rank']}]:")
        preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        print(preview)
    print("="*80)
    
    print("\n✅ Demo complete! Run without --demo for interactive mode.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("PDF Q&A System with RAG - Phase 2")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <pdf_file> [--demo]")
        print("\nExamples:")
        print(f"  python {sys.argv[0]} document.pdf")
        print(f"  python {sys.argv[0]} document.pdf --demo")
        print("\nEnvironment Variables:")
        print("  OPENAI_API_KEY     - For OpenAI models")
        print("  ANTHROPIC_API_KEY  - For Anthropic Claude")
        print("\nLocal Options:")
        print("  - Install Ollama: https://ollama.ai")
        print("  - Or use HuggingFace models (auto-downloads)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    demo = "--demo" in sys.argv
    
    if demo:
        demo_mode(pdf_path)
    else:
        process_pdf_with_rag(pdf_path)


if __name__ == "__main__":
    main()

