# PDF Q&A System Documentation

[![CI](https://github.com/arunbcodes/doc-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/arunbcodes/doc-qa-system/actions/workflows/ci.yml)
[![Docker Build](https://github.com/arunbcodes/doc-qa-system/actions/workflows/docker.yml/badge.svg)](https://github.com/arunbcodes/doc-qa-system/actions/workflows/docker.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the **PDF Q&A System** documentation! This system provides production-ready PDF question-answering capabilities with semantic search and LLM-powered Retrieval Augmented Generation (RAG).

## Overview

The PDF Q&A System allows you to:

- 📚 **Process multiple PDFs** simultaneously with source tracking
- 📄 **Extract text from PDFs** with high accuracy
- 🔍 **Search semantically** using meaning, not just keywords
- 🤖 **Generate answers** using multiple LLM providers
- 🏠 **Run locally** with complete privacy
- 🐳 **Deploy easily** with Docker
- 🔌 **Integrate simply** as a Python library

## Key Features

### Semantic Search
Find relevant content by meaning using state-of-the-art sentence transformers. No keyword matching required.

### Model-Agnostic RAG
Works with 6+ LLM providers out of the box:

- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Ollama (Local LLMs)
- HuggingFace Models
- Local Servers (vLLM, text-generation-webui)
- Mock LLM (for testing)

### Local-First
Run completely offline with local models. Your data never leaves your machine.

### Production-Ready
- Comprehensive test suite (47+ tests)
- Docker containerization
- CI/CD automation
- Security scanning
- Type hints and documentation

## Quick Example

```python
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore, RAGInterface, get_available_llm

# Process PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# Create embeddings
chunker = TextChunker()
chunks = chunker.chunk_text(text)
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

# Store and query
store = VectorStore()
store.add_chunks(chunks, embeddings)

# Get LLM-powered answer
llm = get_available_llm()
rag = RAGInterface(embedder, store, llm)
result = rag.answer_question("What is this document about?")
print(result['answer'])
```

## Architecture

The system follows a clean, modular architecture:

```
PDF → Extract → Chunk → Embed → Vector Store
                                      ↓
                Question → Embed → Search → Top Chunks → Prompt → LLM → Answer
```

### Phase 1: Semantic Search
Pure retrieval without LLM generation. Fast and privacy-preserving.

### Phase 2: RAG Pipeline
Combines retrieval with LLM generation for natural language answers.

## Getting Started

Choose your preferred method:

=== "Package Installation"
    ```bash
    pip install -e .
    python main_rag.py data/sample.pdf
    ```

=== "Docker"
    ```bash
    docker build -t pdf-qa-system .
    docker run -it --rm -v $(pwd)/data:/app/data pdf-qa-system python main.py /app/data/sample.pdf
    ```

=== "From Source"
    ```bash
    git clone https://github.com/arunbcodes/doc-qa-system.git
    cd doc-qa-system
    pip install -r requirements.txt
    python main.py data/sample.pdf
    ```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and quick start guides
- **[User Guide](user-guide/configuration.md)**: Configuration and usage instructions
- **[API Reference](api/overview.md)**: Detailed API documentation
- **[Deployment](deployment/docker.md)**: Production deployment guides
- **[Development](development/contributing.md)**: Contributing and development setup

## Support

- 📖 [Read the docs](https://arunbcodes.github.io/doc-qa-system/)
- 🐛 [Report issues](https://github.com/arunbcodes/doc-qa-system/issues)
- 💬 [Discussions](https://github.com/arunbcodes/doc-qa-system/discussions)
- ⭐ [Star on GitHub](https://github.com/arunbcodes/doc-qa-system)

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/arunbcodes/doc-qa-system/blob/main/LICENSE) file for details.
