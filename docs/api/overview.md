# API Reference Overview

Complete API documentation for the PDF Q&A System.

## Core Modules

### [PDFParser](extract.md)
Extract text from PDF files using Docling.

```python
from src import PDFParser

parser = PDFParser()
text = parser.extract_text("document.pdf")
```

### [TextChunker](chunk.md)
Split text into manageable chunks with overlap.

```python
from src import TextChunker

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_text(text)
```

### [EmbeddingModel](embed.md)
Generate semantic embeddings using sentence transformers.

```python
from src import EmbeddingModel

embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
embeddings = embedder.embed_batch(chunks)
```

### [VectorStore](vector-store.md)
Store and query embeddings using ChromaDB.

```python
from src import VectorStore

store = VectorStore(collection_name="docs")
store.add_chunks(chunks, embeddings)
results = store.query(query_embedding, n_results=3)
```

### [LLM Providers](llm-providers.md)
Integrate with multiple LLM providers.

```python
from src import get_available_llm, OpenAILLM, OllamaLLM

# Auto-detect available LLM
llm = get_available_llm()

# Or use specific provider
llm = OpenAILLM(model_name="gpt-3.5-turbo")
llm = OllamaLLM(model_name="llama3.2")
```

### [RAGInterface](rag.md)
Complete RAG pipeline for question answering.

```python
from src import RAGInterface

rag = RAGInterface(embedder, store, llm)
result = rag.answer_question("What is this about?")
```

## Quick Reference

### Initialization

```python
from src import (
    PDFParser,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    RAGInterface,
    get_available_llm
)

# Initialize components
parser = PDFParser()
chunker = TextChunker()
embedder = EmbeddingModel()
store = VectorStore()
llm = get_available_llm()
rag = RAGInterface(embedder, store, llm)
```

### Processing Pipeline

```python
# 1. Extract
text = parser.extract_text("document.pdf")

# 2. Chunk
chunks = chunker.chunk_text(text)

# 3. Embed
embeddings = embedder.embed_batch(chunks)

# 4. Store
store.add_chunks(chunks, embeddings)

# 5. Query
result = rag.answer_question("Your question?")
```

## Type Hints

All modules include comprehensive type hints:

```python
def extract_text(self, pdf_path: str) -> str: ...
def chunk_text(self, text: str) -> list[str]: ...
def embed(self, text: str) -> list[float]: ...
def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
def add_chunks(
    self,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None
) -> None: ...
```

## Error Handling

```python
from src.exceptions import (
    PDFParsingError,
    EmbeddingError,
    VectorStoreError,
    LLMError
)

try:
    text = parser.extract_text("file.pdf")
except PDFParsingError as e:
    print(f"Failed to parse PDF: {e}")

try:
    embeddings = embedder.embed_batch(chunks)
except EmbeddingError as e:
    print(f"Failed to generate embeddings: {e}")
```

## Configuration

Most classes accept configuration parameters:

```python
# Text chunking
chunker = TextChunker(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=50       # Overlap between chunks
)

# Embeddings
embedder = EmbeddingModel(
    model_name="all-MiniLM-L6-v2",  # Model name
    device="cpu"                      # "cpu" or "cuda"
)

# Vector store
store = VectorStore(
    collection_name="my_docs",
    persist_directory="./db"
)

# RAG
rag = RAGInterface(
    embedder=embedder,
    vector_store=store,
    llm=llm,
    n_results=3           # Results to retrieve
)
```

## Detailed Documentation

- [PDFParser](extract.md) - PDF text extraction
- [TextChunker](chunk.md) - Text chunking strategies
- [EmbeddingModel](embed.md) - Semantic embeddings
- [VectorStore](vector-store.md) - Vector database operations
- [LLM Providers](llm-providers.md) - LLM integrations
- [RAGInterface](rag.md) - RAG pipeline
