"""
Test Query Script - Non-interactive demonstration
Run specific queries against a PDF without interactive mode.
"""

import sys
from .extract import PDFParser
from .chunk import TextChunker
from .embed import EmbeddingModel
from .vector_store import VectorStore
from .query import QueryInterface


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Step 1: Load PDF
    print(f"\n{'='*80}")
    print(f"Loading PDF: {pdf_path}")
    print(f"{'='*80}\n")
    
    parser = PDFParser()
    document_text = parser.extract_text(pdf_path)
    
    if not document_text:
        print("Error: No text extracted from PDF")
        sys.exit(1)
    
    print(f"✓ Extracted {len(document_text)} characters from PDF\n")
    
    # Step 2: Chunk text
    print("Chunking text...")
    chunker = TextChunker()
    chunks = chunker.chunk_text(document_text)
    print(f"✓ Created {len(chunks)} chunks\n")
    
    # Step 3: Initialize embedding model
    print("Loading embedding model...")
    embedding_model = EmbeddingModel()
    print("✓ Embedding model ready\n")
    
    # Step 4: Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_model.embed_batch(chunks, show_progress=True)
    print(f"✓ Generated {len(embeddings)} embeddings\n")
    
    # Step 5: Store in vector database
    print("Storing in vector database...")
    vector_store = VectorStore()
    vector_store.add_chunks(chunks, embeddings)
    print(f"✓ Stored {vector_store.get_count()} chunks in vector database\n")
    
    # Step 6: Initialize query interface
    query_interface = QueryInterface(embedding_model, vector_store, n_results=3)
    
    # Step 7: Run test queries
    test_queries = [
        "What is this document about?",
        "What are the main benefits?",
        "What are the coverage details?"
    ]
    
    print(f"\n{'='*80}")
    print("RUNNING TEST QUERIES")
    print(f"{'='*80}\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        results = query_interface.query(query)
        
        if results:
            for result in results:
                rank = result['rank']
                text = result['text']
                distance = result.get('distance', 0)
                similarity = 1 - distance if distance is not None else 0
                
                print(f"\nResult #{rank} (Similarity: {similarity:.4f}):")
                # Show first 200 characters of result
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"{preview}")
        else:
            print("No results found")
        
        print("\n" + "=" * 80)
    
    print("\n✓ Test completed successfully!")
    print("\nTo run your own queries, use:")
    print(f"  python main.py \"{pdf_path}\"")
    print("Then enter your questions interactively.\n")


if __name__ == "__main__":
    main()

