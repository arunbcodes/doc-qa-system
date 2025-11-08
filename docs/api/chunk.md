# TextChunker API

Split text into manageable chunks with overlap for better context preservation.

## Class: TextChunker

::: src.chunk.TextChunker

### Overview

The `TextChunker` class splits large text documents into smaller, overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. This is essential for semantic search and LLM processing, as it maintains context while keeping chunks within size limits.

## Basic Usage

```python
from src import TextChunker

# Initialize chunker
chunker = TextChunker(chunk_size=500, chunk_overlap=50)

# Split text
text = "Your long document text here..."
chunks = chunker.chunk_text(text)

print(f"Created {len(chunks)} chunks")
```

## Methods

### `__init__(chunk_size: int = 500, chunk_overlap: int = 50)`

Initialize the text chunker.

**Parameters:**

- `chunk_size` (int): Maximum size of each chunk in characters (default: 500)
- `chunk_overlap` (int): Number of overlapping characters between chunks (default: 50)

**Returns:** `TextChunker` instance

**Example:**

```python
# Default settings (balanced)
chunker = TextChunker()

# Small chunks (precise search)
chunker = TextChunker(chunk_size=300, chunk_overlap=30)

# Large chunks (more context)
chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
```

### `chunk_text(text: str) -> List[str]`

Split text into chunks.

**Parameters:**

- `text` (str): Input text to split

**Returns:**

- `List[str]`: List of text chunks

**Example:**

```python
text = "This is a long document. It has multiple paragraphs..."
chunks = chunker.chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
```

### `chunk_with_metadata(text: str, source_metadata: Dict = None) -> List[Dict]`

Split text into chunks with metadata.

**Parameters:**

- `text` (str): Input text to split
- `source_metadata` (Dict, optional): Metadata to attach to each chunk

**Returns:**

- `List[Dict]`: List of dictionaries containing:
  - `text` (str): Chunk text
  - `metadata` (dict): Chunk metadata including:
    - `chunk_index` (int): Index of chunk
    - `chunk_size` (int): Size in characters
    - Additional fields from `source_metadata`

**Example:**

```python
text = "Document content..."
metadata = {"source": "document.pdf", "author": "John Doe"}

chunks = chunker.chunk_with_metadata(text, metadata)

for chunk_data in chunks:
    print(f"Chunk {chunk_data['metadata']['chunk_index']}")
    print(f"Size: {chunk_data['metadata']['chunk_size']}")
    print(f"Source: {chunk_data['metadata']['source']}")
    print(f"Text: {chunk_data['text'][:100]}...")
```

### `get_stats(text: str) -> Dict`

Get statistics about how text will be chunked.

**Parameters:**

- `text` (str): Input text

**Returns:**

- `Dict`: Dictionary with statistics:
  - `num_chunks` (int): Number of chunks
  - `avg_chunk_size` (float): Average chunk size
  - `min_chunk_size` (int): Smallest chunk size
  - `max_chunk_size` (int): Largest chunk size
  - `total_characters` (int): Total characters in input

**Example:**

```python
text = "Your document text..."
stats = chunker.get_stats(text)

print(f"Will create {stats['num_chunks']} chunks")
print(f"Average size: {stats['avg_chunk_size']:.0f} chars")
print(f"Range: {stats['min_chunk_size']}-{stats['max_chunk_size']} chars")
```

## Complete Example

```python
from src import PDFParser, TextChunker

# Extract text
parser = PDFParser()
result = parser.extract_with_metadata("document.pdf")

# Initialize chunker
chunker = TextChunker(chunk_size=500, chunk_overlap=50)

# Get stats first
stats = chunker.get_stats(result["text"])
print(f"Document will be split into {stats['num_chunks']} chunks")
print(f"Average chunk size: {stats['avg_chunk_size']:.0f} characters")

# Create chunks with metadata
chunks = chunker.chunk_with_metadata(
    result["text"],
    source_metadata=result["metadata"]
)

# Process chunks
for chunk_data in chunks:
    print(f"\nChunk {chunk_data['metadata']['chunk_index']}:")
    print(f"  Size: {chunk_data['metadata']['chunk_size']} chars")
    print(f"  Text preview: {chunk_data['text'][:100]}...")
```

## Chunking Strategy

The chunker uses a recursive strategy with the following separators in order:

1. Double newline (`\n\n`) - Paragraph breaks
2. Single newline (`\n`) - Line breaks
3. Period and space (`. `) - Sentence breaks
4. Space (` `) - Word breaks
5. Empty string (`""`) - Character level (fallback)

This ensures chunks break at natural boundaries when possible.

## Choosing Parameters

### Chunk Size

| Size | Use Case | Pros | Cons |
|------|----------|------|------|
| 200-400 | Precise search | Exact matches | Less context |
| 500-800 | Balanced (default) | Good balance | General purpose |
| 1000+ | LLM context | More context | Less granular |

### Chunk Overlap

- **Low overlap (20-50)**: Faster processing, less redundancy
- **Medium overlap (50-100)**: Balanced, recommended
- **High overlap (100-200)**: Better context preservation, more chunks

**Rule of thumb:** Overlap should be 10-20% of chunk size

```python
# Examples
chunker = TextChunker(chunk_size=500, chunk_overlap=50)   # 10% overlap
chunker = TextChunker(chunk_size=1000, chunk_overlap=100) # 10% overlap
chunker = TextChunker(chunk_size=300, chunk_overlap=60)   # 20% overlap
```

## Advanced Usage

### Custom Separators

For specialized text (e.g., code, structured data):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Code-specific separators
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
)
```

### Adaptive Chunking

Adjust chunk size based on document length:

```python
def get_optimal_chunk_size(text_length):
    """Choose chunk size based on document length."""
    if text_length < 5000:
        return 300  # Small docs: precise
    elif text_length < 50000:
        return 500  # Medium docs: balanced
    else:
        return 1000  # Large docs: efficiency

text_length = len(text)
chunk_size = get_optimal_chunk_size(text_length)
chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_size // 10)
```

### Batch Processing

```python
documents = [
    {"id": 1, "text": "Document 1 text..."},
    {"id": 2, "text": "Document 2 text..."},
]

chunker = TextChunker()
all_chunks = []

for doc in documents:
    chunks = chunker.chunk_with_metadata(
        doc["text"],
        source_metadata={"doc_id": doc["id"]}
    )
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")
```

## Integration with Pipeline

### With Embeddings

```python
from src import TextChunker, EmbeddingModel

# Chunk text
chunker = TextChunker(chunk_size=500)
chunks = chunker.chunk_text(text)

# Generate embeddings
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

print(f"Created {len(embeddings)} embeddings for {len(chunks)} chunks")
```

### Complete RAG Pipeline

```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore

# 1. Extract
parser = PDFParser()
result = parser.extract_with_metadata("document.pdf")

# 2. Chunk
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_with_metadata(
    result["text"],
    source_metadata=result["metadata"]
)

# 3. Embed
embedder = EmbeddingModel()
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_batch(texts)

# 4. Store
store = VectorStore(collection_name="docs")
for chunk, embedding in zip(chunks, embeddings):
    store.add_chunks(
        [chunk["text"]],
        [embedding],
        [chunk["metadata"]]
    )

print(f"Indexed {len(chunks)} chunks from {result['metadata']['source']}")
```

## Performance Tips

### Memory Optimization

```python
# Process in batches for large documents
def chunk_large_document(text, batch_size=10000):
    chunker = TextChunker(chunk_size=500)
    all_chunks = []

    # Process in segments
    for i in range(0, len(text), batch_size):
        segment = text[i:i+batch_size]
        chunks = chunker.chunk_text(segment)
        all_chunks.extend(chunks)

    return all_chunks
```

### Empty Text Handling

```python
text = ""
chunks = chunker.chunk_text(text)
print(chunks)  # Returns: []

# Always check before processing
if text and text.strip():
    chunks = chunker.chunk_text(text)
else:
    print("No text to chunk")
```

## Dependencies

- `langchain-text-splitters>=0.2.0`: Text splitting functionality

## See Also

- [PDFParser API](extract.md) - Extract text from PDFs
- [EmbeddingModel API](embed.md) - Generate embeddings
- [Configuration Guide](../user-guide/configuration.md) - Tune chunking parameters
