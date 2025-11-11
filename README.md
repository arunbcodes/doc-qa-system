# PDF Q&A System with RAG

[![CI](https://github.com/arunbcodes/doc-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/arunbcodes/doc-qa-system/actions/workflows/ci.yml)
[![Docker Build](https://github.com/arunbcodes/doc-qa-system/actions/workflows/docker.yml/badge.svg)](https://github.com/arunbcodes/doc-qa-system/actions/workflows/docker.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready PDF question-answering system with semantic search and LLM-powered answers. Works with any LLM provider (OpenAI, Ollama, etc.) or no LLM at all.

## Features

- **Multi-PDF Support** - Process and query multiple PDF documents simultaneously
- **Semantic Search** - Find relevant content by meaning, not keywords
- **Model-Agnostic RAG** - Works with 6+ LLM providers (OpenAI, Ollama, Claude, etc.)
- **Local-First** - Run completely offline with local models
- **Clean Architecture** - Modular, testable, production-ready code

## Installation

### Option 1: Install as Package (Recommended)

```bash
# Install from source
pip install -e .

# Or with all dependencies (LLM providers + dev tools)
pip install -e ".[all]"

# Or install only what you need
pip install -e ".[llm]"  # LLM providers only
pip install -e ".[dev]"  # Development tools only
```

### Option 2: Install Dependencies Only

```bash
pip install -r requirements.txt
```

### Option 3: Docker

See [Docker Deployment](#docker-deployment) section below.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t pdf-qa-system .

# Run Phase 1 (Semantic Search)
docker run -it --rm -v $(pwd)/data:/app/data pdf-qa-system python main.py /app/data/sample.pdf

# Run Phase 2 (RAG with LLM) - with API key
docker run -it --rm -e OPENAI_API_KEY=sk-... -v $(pwd)/data:/app/data pdf-qa-system python main_rag.py /app/data/sample.pdf

# Or use docker-compose
docker-compose --profile search run --rm pdf-qa-search
```

### Option 2: Local Python Environment

#### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Run Phase 1: Semantic Search (No LLM)

```bash
python main.py data/sample.pdf
```

Returns relevant text chunks for your questions.

#### 3. Run Phase 2: RAG with LLM (Natural Language Answers)

**Single PDF:**
```bash
python main_rag.py data/sample.pdf
```

**Multiple PDFs:**
```bash
python main_rag.py data/doc1.pdf data/doc2.pdf data/doc3.pdf
```

Generates natural language answers using an LLM. When multiple PDFs are provided, the system combines all documents into a unified knowledge base for querying.

## LLM Options

### Local Models (Recommended)

**Ollama** - Easiest local setup:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
python main_rag.py data/sample.pdf
```

**OpenAI gpt-oss-20b** - Latest open-source model:
```bash
pip install transformers accelerate
python main_rag.py data/sample.pdf
# Select HuggingFace → openai/gpt-oss-20b
```

### Cloud APIs

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."

python main_rag.py data/sample.pdf
```

## Project Structure

```
pdf-qa-system/
├── main.py              # Semantic search CLI
├── main_rag.py          # RAG with LLM CLI
├── test.py              # Quick test
├── requirements.txt     # Dependencies
│
├── data/                # Your PDF files
│   └── sample.pdf
│
├── src/                 # Core modules
│   ├── extract.py       # PDF → text
│   ├── chunk.py         # Text → chunks
│   ├── embed.py         # Chunks → vectors
│   ├── vector_store.py  # Vector database
│   ├── query.py         # Search interface
│   ├── llm_providers.py # LLM integrations
│   └── rag.py           # RAG pipeline
│
└── docs/                # Documentation
    └── ARCHITECTURE.md  # Technical details
```

## Usage Examples

### Semantic Search
```bash
$ python main.py data/sample.pdf
> What are the benefits?
[Shows 3 most relevant text chunks]
```

### RAG with LLM
```bash
$ python main_rag.py data/sample.pdf
> What are the benefits?
💡 Based on the document, the main benefits include:
1. Wellness app with health tracking
2. Coverage up to Rs. 10 Lakhs
3. Accidental death coverage
...
```

## Supported LLM Providers

| Provider | Cost | Privacy | Setup |
|----------|------|---------|-------|
| Ollama | Free | 100% Local | `ollama pull llama3.2` |
| gpt-oss-20b | Free | 100% Local | Auto-downloads |
| OpenAI | Paid | Cloud | Set `OPENAI_API_KEY` |
| Anthropic | Paid | Cloud | Set `ANTHROPIC_API_KEY` |
| HuggingFace | Free | 100% Local | Auto-downloads |
| Local Server | Free | 100% Local | Start vLLM/text-gen-webui |

## Multi-PDF Usage

Process and query multiple PDF documents simultaneously:

```bash
# Process multiple PDFs
python main_rag.py report1.pdf report2.pdf report3.pdf

# With demo mode
python main_rag.py doc1.pdf doc2.pdf --demo

# Interactive mode - queries all documents
❓ Question: What are the common themes across all documents?
💡 ANSWER: [Analyzes content from all PDFs]
```

The system automatically:
- Processes all PDFs with source tracking metadata
- Combines chunks into a unified vector database
- Persists data to `./chroma_db` directory
- Skips already-processed PDFs on subsequent runs
- Attributes answers to specific source documents
- Maintains separate statistics per PDF

**Persistence:**
The vector database is saved to disk, so you can:
- Stop and restart the program without losing data
- Add new PDFs without re-processing old ones
- Query previously uploaded PDFs immediately

## Using as a Library

**Single PDF:**
```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore, RAGInterface

# Process PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# Create embeddings
chunker = TextChunker()
chunks = chunker.chunk_text(text)
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

# Store in vector DB
store = VectorStore()
store.add_chunks(chunks, embeddings)

# Query
from src import get_available_llm
rag = RAGInterface(embedder, store, llm=get_available_llm())
result = rag.answer_question("What is this about?")
print(result['answer'])
```

**Multiple PDFs:**
```python
from src import PDFProcessor, EmbeddingModel, VectorStore, RAGInterface, get_available_llm

# Initialize components
embedding_model = EmbeddingModel()
vector_store = VectorStore()
processor = PDFProcessor(embedding_model=embedding_model)

# Process multiple PDFs
pdf_paths = ["report1.pdf", "report2.pdf", "report3.pdf"]
stats = processor.process_and_store(pdf_paths, vector_store, show_progress=True)

print(f"Processed {stats['total_pdfs']} PDFs")
print(f"Total chunks: {stats['total_chunks']}")

# Query across all documents
rag = RAGInterface(embedding_model, vector_store, llm=get_available_llm())
result = rag.answer_question("What are the common themes?")
print(result['answer'])

# View source information in retrieved context
result_with_context = rag.answer_question("Summarize key points", show_context=True)
for chunk in result_with_context['context']:
    print(f"From {chunk['metadata']['source']}: {chunk['text'][:100]}...")
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Key configuration options:
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `CHUNK_SIZE` - Text chunk size (default: 500)
- `CHUNK_OVERLAP` - Chunk overlap (default: 50)

### Code Configuration

Edit settings in the respective modules:

- **Chunk size**: `src/chunk.py` → `TextChunker(chunk_size=500)`
- **Number of results**: `src/query.py` → `QueryInterface(n_results=3)`
- **Embedding model**: `src/embed.py` → `EmbeddingModel(model_name="...")`

## Testing

Run the test suite with pytest:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/test_chunk.py
pytest tests/test_embed.py

# Run only fast tests (skip slow tests)
pytest -m "not slow"

# Run only unit tests
pytest -m unit
```

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB+ recommended for large models)
- 10GB disk space (for models)

## Docker Deployment

### Building and Running

```bash
# Build the image
docker build -t pdf-qa-system:latest .

# Run with your PDF files
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main.py /app/data/your-file.pdf
```

### Using Docker Compose

```bash
# Phase 1 (Semantic Search)
docker-compose --profile search run --rm pdf-qa-search

# Phase 2 (RAG with LLM)
docker-compose --profile rag run --rm pdf-qa-rag

# With Ollama (local LLM)
docker-compose --profile ollama up -d ollama
docker-compose --profile rag run --rm pdf-qa-rag
```

### Environment Variables

Pass API keys and configuration via environment variables:

```bash
docker run -it --rm \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v $(pwd)/data:/app/data \
  pdf-qa-system:latest \
  python main_rag.py /app/data/sample.pdf
```

### Persistent Storage

Models and embeddings are cached in Docker volumes for faster subsequent runs:

```bash
# View volumes
docker volume ls | grep pdf-qa

# Clean up volumes
docker volume rm pdf-qa-cache pdf-qa-models
```

## Architecture

### Retrieval Pipeline (Phase 1)
```
PDF → Extract → Chunk → Embed → Vector Store → Query → Results
```

### RAG Pipeline (Phase 2)
```
PDF → Extract → Chunk → Embed → Vector Store
                                    ↓
Question → Embed → Search → Top Chunks → Prompt → LLM → Answer
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical details and design decisions

## License

MIT License

## Acknowledgments

- Docling - PDF parsing
- Sentence Transformers - Embeddings
- Chroma - Vector database
- LangChain - Text splitting
- Ollama - Local LLM runtime
