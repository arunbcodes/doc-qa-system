# Configuration

Complete configuration guide for the PDF Q&A System.

## Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### LLM API Keys

#### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Get your key: [OpenAI Platform](https://platform.openai.com/api-keys)

#### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Get your key: [Anthropic Console](https://console.anthropic.com/)

#### HuggingFace

```bash
export HF_TOKEN="hf_..."
```

Get your token: [HuggingFace Settings](https://huggingface.co/settings/tokens)

### Ollama Configuration

```bash
# Ollama server URL (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# Model name (default: llama3.2)
export OLLAMA_MODEL="llama3.2"
```

### Model Configuration

```bash
# Embedding model (default: all-MiniLM-L6-v2)
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Text chunking
export CHUNK_SIZE=500
export CHUNK_OVERLAP=50

# Number of results to retrieve
export N_RESULTS=3
```

### Storage Configuration

```bash
# ChromaDB persistence directory
export CHROMA_PERSIST_DIR="./chroma_db"

# HuggingFace model cache
export HF_HOME="./models"
export TRANSFORMERS_CACHE="./models"
```

### Application Settings

```bash
# Python buffering
export PYTHONUNBUFFERED=1

# Logging level
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Code Configuration

### Text Chunking

Adjust chunk size and overlap:

```python
from src import TextChunker

# Smaller chunks (more granular, slower)
chunker = TextChunker(chunk_size=300, chunk_overlap=30)

# Larger chunks (more context, faster)
chunker = TextChunker(chunk_size=1000, chunk_overlap=100)

# Recommended for most cases
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
```

**Guidelines:**
- Smaller chunks (200-400): Better for precise search
- Medium chunks (500-800): Balanced performance
- Larger chunks (1000+): More context for LLM

### Embedding Model

Choose different embedding models:

```python
from src import EmbeddingModel

# Default (fast, 384 dimensions)
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Better quality (slower, 768 dimensions)
embedder = EmbeddingModel(model_name="all-mpnet-base-v2")

# Multilingual
embedder = EmbeddingModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

Available models: [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)

### Vector Store

Configure the vector database:

```python
from src import VectorStore

# In-memory (ephemeral)
store = VectorStore(collection_name="docs")

# Persistent storage
store = VectorStore(
    collection_name="docs",
    persist_directory="./my_db"
)

# Custom number of results
results = store.query(embedding, n_results=5)
```

### LLM Configuration

Configure specific LLM providers:

```python
from src.llm_providers import OpenAILLM, AnthropicLLM, OllamaLLM

# OpenAI
llm = OpenAILLM(
    model_name="gpt-4",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=500
)

# Anthropic
llm = AnthropicLLM(
    model_name="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    max_tokens=1000
)

# Ollama
llm = OllamaLLM(
    model_name="mistral",
    base_url="http://localhost:11434",
    temperature=0.5
)
```

### RAG Configuration

Configure the RAG pipeline:

```python
from src import RAGInterface

rag = RAGInterface(
    embedder=embedder,
    vector_store=store,
    llm=llm,
    n_results=3,  # Number of chunks to retrieve
)

# Custom prompt template (advanced)
rag.prompt_template = """Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
```

## Configuration Files

### pyproject.toml

Project-level configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Docker Configuration

Configure via `docker-compose.yml`:

```yaml
services:
  pdf-qa:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHUNK_SIZE=1000
      - N_RESULTS=5
    volumes:
      - ./data:/app/data
      - models:/app/models
```

## Performance Tuning

### Memory Usage

Reduce memory for large documents:

```python
# Process in batches
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embeddings = embedder.embed_batch(batch)
    store.add_chunks(batch, embeddings)
```

### Speed Optimization

```python
# Use GPU if available
embedder = EmbeddingModel(device="cuda")

# Reduce chunk size for faster processing
chunker = TextChunker(chunk_size=300)

# Use faster embedding model
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
```

### Quality vs Speed

```python
# Maximum quality (slow)
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
embedder = EmbeddingModel(model_name="all-mpnet-base-v2")
rag = RAGInterface(embedder, store, llm, n_results=5)

# Balanced (recommended)
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
rag = RAGInterface(embedder, store, llm, n_results=3)

# Maximum speed (lower quality)
chunker = TextChunker(chunk_size=300, chunk_overlap=30)
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
rag = RAGInterface(embedder, store, llm, n_results=2)
```

## Troubleshooting

### Environment Variables Not Loading

```python
# Load .env file manually
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("OPENAI_API_KEY")
```

### Model Download Issues

```python
# Specify cache directory
import os
os.environ["HF_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"
```

### Memory Errors

```python
# Reduce batch size
embeddings = []
for chunk in chunks:
    emb = embedder.embed(chunk)
    embeddings.append(emb)
```

## Next Steps

- [LLM Providers Guide](llm-providers.md)
- [Docker Configuration](docker.md)
- [Troubleshooting](troubleshooting.md)
