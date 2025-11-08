# VectorStore API

Manage Chroma vector database for storing and retrieving document chunks.

## Class: VectorStore

::: src.vector_store.VectorStore

### Overview

The `VectorStore` class provides a wrapper around ChromaDB for storing text chunks with their embeddings and performing similarity-based search. It supports both in-memory and persistent storage.

## Basic Usage

```python
from src import VectorStore

# In-memory store (temporary)
store = VectorStore(collection_name="docs")

# Persistent store (saves to disk)
store = VectorStore(
    collection_name="docs",
    persist_directory="./chroma_db"
)
```

## Methods

### `__init__(collection_name: str = "pdf_chunks", persist_directory: Optional[str] = None)`

Initialize the vector store.

**Parameters:**

- `collection_name` (str): Name of the collection (default: "pdf_chunks")
- `persist_directory` (Optional[str]): Directory for persistent storage (None for in-memory)

**Returns:** `VectorStore` instance

**Example:**

```python
# In-memory (ephemeral)
store = VectorStore(collection_name="temp_docs")

# Persistent (survives restarts)
store = VectorStore(
    collection_name="my_documents",
    persist_directory="./my_db"
)
```

### `add_chunks(chunks: List[str], embeddings: List, metadatas: Optional[List[Dict]] = None)`

Add text chunks with embeddings to the store.

**Parameters:**

- `chunks` (List[str]): List of text chunks
- `embeddings` (List): List of embedding vectors
- `metadatas` (Optional[List[Dict]]): Optional metadata for each chunk

**Example:**

```python
chunks = ["First chunk", "Second chunk", "Third chunk"]
embeddings = embedder.embed_batch(chunks)

# Without metadata
store.add_chunks(chunks, embeddings)

# With metadata
metadatas = [
    {"source": "doc1.pdf", "page": 1},
    {"source": "doc1.pdf", "page": 2},
    {"source": "doc2.pdf", "page": 1}
]
store.add_chunks(chunks, embeddings, metadatas)
```

### `search(query_embedding, n_results: int = 5) -> Dict`

Search for similar chunks using query embedding.

**Parameters:**

- `query_embedding`: Query embedding vector
- `n_results` (int): Number of results to return (default: 5)

**Returns:**

- `Dict`: Dictionary with search results containing:
  - `documents`: List of matching text chunks
  - `metadatas`: List of metadata for each result
  - `distances`: List of distances (lower is more similar)
  - `ids`: List of document IDs

**Example:**

```python
# Generate query embedding
question = "What is machine learning?"
query_embedding = embedder.embed_text(question)

# Search
results = store.search(query_embedding, n_results=3)

# Access results
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}:")
    print(f"  Text: {doc}")
    print(f"  Distance: {results['distances'][0][i]}")
    print(f"  Metadata: {results['metadatas'][0][i]}")
```

### `get_count() -> int`

Get the number of chunks in the store.

**Returns:**

- `int`: Number of stored chunks

**Example:**

```python
count = store.get_count()
print(f"Vector store contains {count} chunks")
```

### `clear()`

Clear all data from the collection.

**Example:**

```python
# Remove all chunks
store.clear()
print(f"Chunks after clear: {store.get_count()}")  # 0
```

### `get_stats() -> Dict`

Get statistics about the vector store.

**Returns:**

- `Dict`: Dictionary with statistics:
  - `collection_name`: Name of the collection
  - `num_chunks`: Number of chunks
  - `persist_directory`: Storage location

**Example:**

```python
stats = store.get_stats()
print(f"Collection: {stats['collection_name']}")
print(f"Chunks: {stats['num_chunks']}")
print(f"Storage: {stats['persist_directory']}")
```

## Complete Example

```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore

# Extract and chunk
parser = PDFParser()
result = parser.extract_with_metadata("document.pdf")

chunker = TextChunker(chunk_size=500)
chunks = chunker.chunk_with_metadata(
    result["text"],
    source_metadata=result["metadata"]
)

# Generate embeddings
embedder = EmbeddingModel()
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_batch(texts)

# Create persistent store
store = VectorStore(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)

# Add chunks
texts = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
store.add_chunks(texts, embeddings, metadatas)

print(f"✓ Stored {store.get_count()} chunks")

# Search
question = "What is the main topic?"
query_emb = embedder.embed_text(question)
results = store.search(query_emb, n_results=3)

print(f"\nFound {len(results['documents'][0])} relevant chunks")
```

## Storage Modes

### In-Memory (Ephemeral)

```python
# Fast, but data lost on exit
store = VectorStore(collection_name="temp")

# Good for:
# - Testing
# - Temporary analysis
# - Single-session work
```

### Persistent (Disk)

```python
# Data persists across sessions
store = VectorStore(
    collection_name="docs",
    persist_directory="./chroma_db"
)

# Good for:
# - Production use
# - Reusable indexes
# - Large document collections
```

### Reloading Persistent Store

```python
# First run: create and populate
store = VectorStore(
    collection_name="docs",
    persist_directory="./chroma_db"
)
store.add_chunks(chunks, embeddings)

# Later: reload existing data
store = VectorStore(
    collection_name="docs",  # Same collection name
    persist_directory="./chroma_db"  # Same directory
)
print(f"Loaded {store.get_count()} existing chunks")
```

## Search Strategies

### Basic Search

```python
query_emb = embedder.embed_text("machine learning")
results = store.search(query_emb, n_results=5)
```

### Filtered Search with Metadata

```python
# Add chunks with metadata
store.add_chunks(
    chunks=["chunk1", "chunk2", "chunk3"],
    embeddings=embeddings,
    metadatas=[
        {"source": "doc1.pdf", "category": "AI"},
        {"source": "doc2.pdf", "category": "ML"},
        {"source": "doc3.pdf", "category": "AI"}
    ]
)

# Filter by metadata (ChromaDB supports where clause)
# Note: requires direct ChromaDB API access
results = store.collection.query(
    query_embeddings=[query_emb.tolist()],
    n_results=5,
    where={"category": "AI"}  # Only AI category
)
```

### Multi-Query Search

```python
# Search with multiple queries
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "What are neural networks?"
]

all_results = []
for query in queries:
    query_emb = embedder.embed_text(query)
    results = store.search(query_emb, n_results=2)
    all_results.append(results)

print(f"Found {len(all_results)} result sets")
```

## Performance Tips

### Batch Indexing

```python
# Efficient: Batch add
chunks = ["chunk" + str(i) for i in range(1000)]
embeddings = embedder.embed_batch(chunks)
store.add_chunks(chunks, embeddings)

# Inefficient: One at a time
# for chunk, emb in zip(chunks, embeddings):
#     store.add_chunks([chunk], [emb])  # Slow!
```

### Index Size Management

```python
# Check size
stats = store.get_stats()
print(f"Current size: {stats['num_chunks']} chunks")

# Clear old data
if stats['num_chunks'] > 10000:
    print("Index too large, clearing...")
    store.clear()
```

### Memory Usage

```python
# Large collections: use persistent storage
store = VectorStore(
    collection_name="large_docs",
    persist_directory="./chroma_db"
)

# Process in batches
batch_size = 1000
for i in range(0, len(all_chunks), batch_size):
    batch_chunks = all_chunks[i:i+batch_size]
    batch_embeddings = embedder.embed_batch(batch_chunks)
    store.add_chunks(batch_chunks, batch_embeddings)
    print(f"Processed {i+batch_size}/{len(all_chunks)}")
```

## Multiple Collections

```python
# Separate collections for different document types
papers_store = VectorStore(
    collection_name="research_papers",
    persist_directory="./chroma_db"
)

books_store = VectorStore(
    collection_name="textbooks",
    persist_directory="./chroma_db"  # Same directory, different collection
)

# Each maintains independent data
print(f"Papers: {papers_store.get_count()}")
print(f"Books: {books_store.get_count()}")
```

## Error Handling

```python
# Empty search
try:
    results = store.search(query_embedding, n_results=5)
    if not results['documents'][0]:
        print("No results found")
except Exception as e:
    print(f"Search error: {e}")

# Check before adding
if chunks and embeddings:
    if len(chunks) == len(embeddings):
        store.add_chunks(chunks, embeddings)
    else:
        print("Error: Chunks and embeddings length mismatch")
```

## Backup and Restore

```bash
# Backup (copy persist directory)
cp -r ./chroma_db ./chroma_db_backup

# Restore
cp -r ./chroma_db_backup ./chroma_db
```

## Dependencies

- `chromadb>=0.4.0`: Vector database

## See Also

- [EmbeddingModel API](embed.md) - Generate embeddings
- [RAGInterface API](rag.md) - Use with RAG system
- [Configuration Guide](../user-guide/configuration.md) - Storage configuration
