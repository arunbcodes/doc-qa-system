# Basic Usage

Complete guide to using the PDF Q&A System.

## Command-Line Interface

### Phase 1: Semantic Search

```bash
python main.py <pdf_file>
```

**Features:**
- No LLM required
- Fast retrieval
- Privacy-preserving
- Returns text chunks with similarity scores

**Example:**
```bash
python main.py data/research-paper.pdf
```

### Phase 2: RAG with LLM

**Single PDF:**
```bash
python main_rag.py <pdf_file>
```

**Multiple PDFs:**
```bash
python main_rag.py <pdf_file1> <pdf_file2> [pdf_file3 ...]
```

**Features:**
- Natural language answers
- Multiple LLM support
- Context-aware responses
- Source attribution
- **Multi-PDF support**: Query across multiple documents simultaneously
- **Persistent database**: PDFs saved to disk, survive restarts

**Examples:**
```bash
# Single PDF with environment variable
export OPENAI_API_KEY="sk-..."
python main_rag.py data/research-paper.pdf

# Multiple PDFs
python main_rag.py data/report1.pdf data/report2.pdf data/report3.pdf

# Multiple PDFs with inline API key
OPENAI_API_KEY="sk-..." python main_rag.py data/doc1.pdf data/doc2.pdf
```

## Interactive Mode

Both scripts run in interactive mode by default:

```
❓ Enter your question (or 'quit' to exit): What is the methodology?

💡 Answer:
The methodology involves...

❓ Enter your question (or 'quit' to exit): quit
👋 Goodbye!
```

**Commands:**
- Type your question and press Enter
- Type `quit` or `exit` to stop
- Press Ctrl+C to interrupt

## Python API

### Basic Example

```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore

# Process PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# Chunk and embed
chunker = TextChunker()
chunks = chunker.chunk_text(text)

embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

# Store
store = VectorStore()
store.add_chunks(chunks, embeddings)

# Query
query_embedding = embedder.embed("What is this about?")
results = store.query(query_embedding, n_results=3)

for i, result in enumerate(results, 1):
    print(f"[{i}] {result}")
```

### RAG Example

```python
from src import RAGInterface, get_available_llm

# Initialize RAG
llm = get_available_llm()
rag = RAGInterface(embedder, store, llm)

# Ask question
result = rag.answer_question("What are the key findings?")

print(f"Answer: {result['answer']}")
print(f"Context used: {len(result['context'])} chunks")
```

### Custom Configuration

```python
# Custom chunk size
chunker = TextChunker(chunk_size=1000, chunk_overlap=100)

# Custom embedding model
embedder = EmbeddingModel(model_name="all-mpnet-base-v2")

# Persistent vector store
store = VectorStore(
    collection_name="my_collection",
    persist_directory="./my_db"
)

# Custom number of results
results = store.query(query_embedding, n_results=5)
```

## Working with Multiple Documents

### Using PDFProcessor (Recommended)

The `PDFProcessor` class provides a streamlined way to process multiple PDFs:

```python
from src import PDFProcessor, EmbeddingModel, VectorStore, RAGInterface, get_available_llm

# Initialize components
embedding_model = EmbeddingModel()
vector_store = VectorStore(collection_name="all_docs")
processor = PDFProcessor(embedding_model=embedding_model)

# Process multiple PDFs
pdf_paths = ["data/report1.pdf", "data/report2.pdf", "data/report3.pdf"]
stats = processor.process_and_store(pdf_paths, vector_store, show_progress=True)

print(f"Processed {stats['total_pdfs']} PDFs")
print(f"Total chunks: {stats['total_chunks']}")

# Query across all documents
rag = RAGInterface(embedding_model, vector_store, llm=get_available_llm())
result = rag.answer_question("What are the common themes across all reports?")
print(result['answer'])

# View source information
result_with_context = rag.answer_question("Summarize key points", show_context=True)
for chunk in result_with_context['context']:
    source = chunk['metadata']['source']
    text_preview = chunk['text'][:100]
    print(f"From {source}: {text_preview}...")
```

### Manual Processing (Advanced)

For fine-grained control, process PDFs manually:

```python
from pathlib import Path

pdf_dir = Path("data/")
store = VectorStore(collection_name="all_docs")

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")

    text = parser.extract_text(str(pdf_file))
    chunks = chunker.chunk_text(text)
    embeddings = embedder.embed_batch(chunks)

    # Add metadata
    metadatas = [{"source": pdf_file.name} for _ in chunks]
    store.add_chunks(chunks, embeddings, metadatas=metadatas)

print("All documents processed!")
```

### Viewing Source Metadata

All chunks include source tracking metadata:

```python
# Get results with context
result = rag.answer_question("What is the policy?", show_context=True)

# Check which PDFs the answer came from
for chunk in result['context']:
    print(f"Source: {chunk['metadata']['source']}")
    print(f"Text: {chunk['text'][:200]}...")
```

## Database Persistence

### How Persistence Works

The vector database is automatically saved to `./chroma_db`:

```bash
# First run - processes PDFs
python main_rag.py doc1.pdf doc2.pdf
# Creates: ./chroma_db/ directory

# Second run - skips already processed PDFs
python main_rag.py doc1.pdf doc2.pdf doc3.pdf
# Only processes doc3.pdf (doc1 and doc2 already in database)

# Third run - query without re-uploading
python main_rag.py doc1.pdf
# Uses existing database, no processing needed
```

### Managing the Database

**Clear the database:**
```bash
# Remove the database directory to start fresh
rm -rf ./chroma_db
```

**Check database contents:**
```python
from src import VectorStore

store = VectorStore(collection_name="pdf_documents", persist_directory="./chroma_db")
print(f"Total chunks: {store.get_count()}")

# Get sample to see sources
results = store.collection.get(limit=10)
sources = {meta.get('source') for meta in results['metadatas'] if meta}
print(f"PDFs in database: {sources}")
```

**Force re-processing:**
```bash
# Delete database and run again
rm -rf ./chroma_db
python main_rag.py doc1.pdf doc2.pdf
```

## Advanced Usage

### Custom LLM Configuration

```python
from src.llm_providers import OpenAILLM, OllamaLLM

# OpenAI with custom model
llm = OpenAILLM(
    model_name="gpt-4",
    api_key="sk-...",
    temperature=0.7
)

# Ollama with custom URL
llm = OllamaLLM(
    model_name="mistral",
    base_url="http://localhost:11434"
)

# Use in RAG
rag = RAGInterface(embedder, store, llm)
```

### Batch Processing

```python
questions = [
    "What is the main topic?",
    "Who are the authors?",
    "What are the conclusions?"
]

for question in questions:
    result = rag.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

### Error Handling

```python
try:
    text = parser.extract_text("document.pdf")
except FileNotFoundError:
    print("PDF file not found")
except Exception as e:
    print(f"Error processing PDF: {e}")

try:
    result = rag.answer_question("What is this?")
except Exception as e:
    print(f"Error generating answer: {e}")
```

## Performance Tips

### 1. Adjust Chunk Size

```python
# Smaller chunks (more granular)
chunker = TextChunker(chunk_size=300, chunk_overlap=30)

# Larger chunks (more context)
chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
```

### 2. Persistent Storage

```python
# Save vector store to disk
store = VectorStore(
    collection_name="docs",
    persist_directory="./chroma_db"
)

# Reuse in next run (no re-embedding needed)
store = VectorStore(
    collection_name="docs",
    persist_directory="./chroma_db"
)
```

### 3. Batch Embeddings

```python
# Process all chunks at once
embeddings = embedder.embed_batch(chunks)

# Instead of one-by-one (slower)
# embeddings = [embedder.embed(chunk) for chunk in chunks]
```

## Next Steps

- [Configuration Guide](../user-guide/configuration.md)
- [LLM Providers](../user-guide/llm-providers.md)
- [API Reference](../api/overview.md)
