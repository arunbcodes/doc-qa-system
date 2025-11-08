# EmbeddingModel API

Generate vector embeddings for text using sentence-transformers.

## Class: EmbeddingModel

::: src.embed.EmbeddingModel

### Overview

The `EmbeddingModel` class provides a wrapper around sentence-transformers models for generating dense vector embeddings from text. These embeddings capture semantic meaning and enable similarity-based search.

## Basic Usage

```python
from src import EmbeddingModel

# Initialize with default model
embedder = EmbeddingModel()

# Embed single text
text = "Machine learning is fascinating"
embedding = embedder.embed_text(text)

print(f"Embedding shape: {embedding.shape}")  # (384,)
print(f"Dimension: {embedder.get_embedding_dimension()}")  # 384
```

## Methods

### `__init__(model_name: str = "all-MiniLM-L6-v2")`

Initialize the embedding model.

**Parameters:**

- `model_name` (str): Name of the sentence-transformers model (default: "all-MiniLM-L6-v2")

**Returns:** `EmbeddingModel` instance

**Available Models:**

| Model | Dimensions | Speed | Quality | Size |
|-------|------------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | 80MB |
| all-mpnet-base-v2 | 768 | Medium | Excellent | 420MB |
| all-MiniLM-L12-v2 | 384 | Medium | Better | 120MB |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Fast | Good | 420MB |

**Example:**

```python
# Default (recommended for most cases)
embedder = EmbeddingModel()

# Higher quality
embedder = EmbeddingModel(model_name="all-mpnet-base-v2")

# Multilingual
embedder = EmbeddingModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

### `embed_text(text: str) -> np.ndarray`

Generate embedding for a single text.

**Parameters:**

- `text` (str): Input text to embed

**Returns:**

- `np.ndarray`: Embedding vector

**Raises:**

- `ValueError`: If text is empty

**Example:**

```python
text = "This is a test sentence."
embedding = embedder.embed_text(text)

print(f"Shape: {embedding.shape}")  # (384,)
print(f"Type: {type(embedding)}")   # <class 'numpy.ndarray'>
print(f"First 5 values: {embedding[:5]}")
```

### `embed_batch(texts: List[str], show_progress: bool = True) -> np.ndarray`

Generate embeddings for multiple texts efficiently.

**Parameters:**

- `texts` (List[str]): List of texts to embed
- `show_progress` (bool): Show progress bar (default: True)

**Returns:**

- `np.ndarray`: Array of embedding vectors, shape (n_texts, embedding_dim)

**Example:**

```python
texts = [
    "Machine learning is fascinating.",
    "Deep learning uses neural networks.",
    "Python is a programming language."
]

# Batch embedding (efficient)
embeddings = embedder.embed_batch(texts, show_progress=True)

print(f"Shape: {embeddings.shape}")  # (3, 384)
print(f"First embedding: {embeddings[0]}")
```

### `get_embedding_dimension() -> int`

Get the dimension of the embedding vectors.

**Returns:**

- `int`: Embedding dimension

**Example:**

```python
dim = embedder.get_embedding_dimension()
print(f"Embedding dimension: {dim}")  # 384 for all-MiniLM-L6-v2
```

### `compute_similarity(text1: str, text2: str) -> float`

Compute cosine similarity between two texts.

**Parameters:**

- `text1` (str): First text
- `text2` (str): Second text

**Returns:**

- `float`: Similarity score between -1 and 1 (higher is more similar)

**Example:**

```python
text1 = "Machine learning is fascinating."
text2 = "Deep learning uses neural networks."
text3 = "What's the weather like today?"

# Related texts
sim = embedder.compute_similarity(text1, text2)
print(f"Similarity 1-2: {sim:.4f}")  # ~0.7-0.8

# Unrelated texts
sim = embedder.compute_similarity(text1, text3)
print(f"Similarity 1-3: {sim:.4f}")  # ~0.1-0.3
```

## Complete Example

```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore

# 1. Extract text
parser = PDFParser()
text = parser.extract_text("document.pdf")

# 2. Chunk text
chunker = TextChunker(chunk_size=500)
chunks = chunker.chunk_text(text)

# 3. Initialize embedder
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
print(f"Model: {embedder.model_name}")
print(f"Dimension: {embedder.get_embedding_dimension()}")

# 4. Generate embeddings
print(f"Embedding {len(chunks)} chunks...")
embeddings = embedder.embed_batch(chunks, show_progress=True)

# 5. Store in vector database
store = VectorStore(collection_name="docs")
store.add_chunks(chunks, embeddings)

print(f"✓ Indexed {len(chunks)} chunks")
```

## Performance Considerations

### GPU Acceleration

```python
# The model automatically uses GPU if available
embedder = EmbeddingModel()

# Check device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Batch vs Single

```python
import time

texts = ["Text " + str(i) for i in range(100)]

# Slow: One at a time
start = time.time()
embeddings = [embedder.embed_text(t) for t in texts]
print(f"Single: {time.time() - start:.2f}s")

# Fast: Batch processing
start = time.time()
embeddings = embedder.embed_batch(texts)
print(f"Batch: {time.time() - start:.2f}s")

# Batch is 5-10x faster!
```

### Memory Management

```python
# For very large datasets, process in batches
def embed_large_dataset(texts, batch_size=100):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedder.embed_batch(batch, show_progress=False)
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)

# Process 10,000 texts in batches
large_texts = ["Text " + str(i) for i in range(10000)]
embeddings = embed_large_dataset(large_texts)
```

## Model Selection Guide

### For Speed (Local/Production)

```python
# Fastest, smallest model
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
# 384 dims, 80MB, ~5000 sentences/sec on CPU
```

### For Quality (Research/Analysis)

```python
# Best quality
embedder = EmbeddingModel(model_name="all-mpnet-base-v2")
# 768 dims, 420MB, ~2500 sentences/sec on CPU
```

### For Multilingual

```python
# Supports 50+ languages
embedder = EmbeddingModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
# 384 dims, supports English, Spanish, French, German, Chinese, etc.
```

## Similarity Search Example

```python
# Create embeddings for documents
documents = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "JavaScript is used for web development"
]

doc_embeddings = embedder.embed_batch(documents)

# Query
query = "What is machine learning?"
query_embedding = embedder.embed_text(query)

# Compute similarities
similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    # Cosine similarity
    sim = np.dot(query_embedding, doc_emb) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
    )
    similarities.append((i, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Show results
print("Most similar documents:")
for idx, sim in similarities[:3]:
    print(f"{sim:.4f}: {documents[idx]}")
```

## Error Handling

```python
# Empty text
try:
    embedding = embedder.embed_text("")
except ValueError as e:
    print(f"Error: {e}")  # "Cannot embed empty text"

# Empty batch
embeddings = embedder.embed_batch([])
print(embeddings.shape)  # (0,)

# Very long text (automatically handled)
long_text = "word " * 10000  # Model has max length (usually 256-512 tokens)
embedding = embedder.embed_text(long_text)  # Works, truncates automatically
```

## Caching Embeddings

```python
import pickle
from pathlib import Path

# Generate and cache
texts = ["text1", "text2", "text3"]
embeddings = embedder.embed_batch(texts)

# Save
cache_file = Path("embeddings_cache.pkl")
with open(cache_file, 'wb') as f:
    pickle.dump(embeddings, f)

# Load
with open(cache_file, 'rb') as f:
    cached_embeddings = pickle.load(f)

print(f"Loaded {len(cached_embeddings)} embeddings from cache")
```

## Dependencies

- `sentence-transformers>=2.2.0`: Embedding model library
- `torch>=2.0.0`: PyTorch backend
- `numpy>=1.24.0`: Array operations

## See Also

- [VectorStore API](vector-store.md) - Store and search embeddings
- [TextChunker API](chunk.md) - Prepare text for embedding
- [Configuration Guide](../user-guide/configuration.md) - Model selection
