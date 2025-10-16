# PDF Q&A System with RAG

A production-ready PDF question-answering system with semantic search and LLM-powered answers. Works with any LLM provider (OpenAI, Ollama, etc.) or no LLM at all.

## Features

- **Semantic Search** - Find relevant content by meaning, not keywords
- **Model-Agnostic RAG** - Works with 6+ LLM providers (OpenAI, Ollama, Claude, etc.)
- **Local-First** - Run completely offline with local models
- **Clean Architecture** - Modular, testable, production-ready code

## Quick Start

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Phase 1: Semantic Search (No LLM)

```bash
python main.py data/sample.pdf
```

Returns relevant text chunks for your questions.

### 3. Run Phase 2: RAG with LLM (Natural Language Answers)

```bash
python main_rag.py data/sample.pdf
```

Generates natural language answers using an LLM.

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

## Using as a Library

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

## Configuration

Edit settings in the respective modules:

- **Chunk size**: `src/chunk.py` → `TextChunker(chunk_size=500)`
- **Number of results**: `src/query.py` → `QueryInterface(n_results=3)`
- **Embedding model**: `src/embed.py` → `EmbeddingModel(model_name="...")`

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB+ recommended for large models)
- 10GB disk space (for models)

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
