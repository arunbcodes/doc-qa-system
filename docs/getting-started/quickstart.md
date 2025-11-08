# Quick Start

Get up and running with the PDF Q&A System in 5 minutes.

## Phase 1: Semantic Search (No LLM)

The simplest way to get started - no API keys required!

### 1. Install

```bash
pip install -e .
```

### 2. Prepare a PDF

Place your PDF file in the `data/` directory:

```bash
mkdir -p data
cp your-document.pdf data/
```

### 3. Run

```bash
python main.py data/your-document.pdf
```

### 4. Query

```
📄 Processing PDF: data/your-document.pdf
✓ Extracted 1234 characters
✓ Created 5 chunks
✓ Generated embeddings
✓ Stored in vector database

❓ Enter your question (or 'quit' to exit): What are the key benefits?

🔍 Top 3 relevant chunks:

[1] Score: 0.87
The key benefits include improved efficiency, cost reduction,
and enhanced user experience...

[2] Score: 0.82
Additional benefits are scalability, maintainability,
and comprehensive documentation...

[3] Score: 0.79
The system provides real-time processing, multi-format support,
and seamless integration...
```

## Phase 2: RAG with LLM (Natural Answers)

Get natural language answers powered by LLMs.

### 1. Choose Your LLM

=== "Ollama (Free, Local)"
    ```bash
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh

    # Pull a model
    ollama pull llama3.2

    # Run with single PDF
    python main_rag.py data/your-document.pdf

    # Run with multiple PDFs
    python main_rag.py data/doc1.pdf data/doc2.pdf data/doc3.pdf
    ```

=== "OpenAI"
    ```bash
    # Set API key
    export OPENAI_API_KEY="sk-..."

    # Run with single PDF
    python main_rag.py data/your-document.pdf

    # Run with multiple PDFs
    python main_rag.py data/report1.pdf data/report2.pdf
    ```

=== "Anthropic (Claude)"
    ```bash
    # Set API key
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Run with Claude
    python main_rag.py data/your-document.pdf
    ```

### 2. Get Answers

```
📄 Processing PDF: data/your-document.pdf
✓ Extracted 1234 characters
✓ Created 5 chunks
✓ Generated embeddings
✓ Stored in vector database
🤖 LLM: Ollama (llama3.2)

❓ Enter your question (or 'quit' to exit): What are the key benefits?

💡 Answer:
Based on the document, the key benefits include:

1. **Improved Efficiency**: The system processes documents 10x faster
2. **Cost Reduction**: Reduces operational costs by 40%
3. **Enhanced UX**: User satisfaction increased by 95%

These benefits are achieved through semantic search and intelligent
document processing capabilities.

⏱️  Time: 2.3s
```

## Docker Quick Start

Run with Docker (no local Python setup needed):

```bash
# Build once
docker build -t pdf-qa-system .

# Run Phase 1
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  pdf-qa-system python main.py /app/data/sample.pdf

# Run Phase 2 with OpenAI
docker run -it --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  pdf-qa-system python main_rag.py /app/data/sample.pdf
```

## Library Usage

Use as a Python library in your code:

```python
from src import (
    PDFParser,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    RAGInterface,
    get_available_llm
)

# 1. Extract text from PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# 2. Chunk the text
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_text(text)

# 3. Generate embeddings
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

# 4. Store in vector database
store = VectorStore(collection_name="my_docs")
store.add_chunks(chunks, embeddings)

# 5. Query with RAG
llm = get_available_llm()
rag = RAGInterface(embedder, store, llm)
result = rag.answer_question("What is the main topic?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Common Commands

### View Help

```bash
python main.py --help
python main_rag.py --help
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific test file
pytest tests/test_chunk.py
```

### Check Code Quality

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy src
```

## What's Next?

- [Basic Usage Guide](basic-usage.md) - Learn all features
- [Configuration](../user-guide/configuration.md) - Customize settings
- [LLM Providers](../user-guide/llm-providers.md) - Configure different LLMs
- [API Reference](../api/overview.md) - Detailed API docs
