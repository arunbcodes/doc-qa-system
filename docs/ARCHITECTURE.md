# Architecture Guide

## System Overview

The PDF Q&A System is a modular pipeline that processes PDFs, creates semantic embeddings, and enables intelligent question-answering with optional LLM integration.

## Core Components

### 1. PDF Extraction (`src/extract.py`)

**Purpose**: Convert PDF to clean text

**Technology**: Docling - handles complex PDF layouts, tables, and formatting

**Key Method**:
```python
parser = PDFParser()
text = parser.extract_text("document.pdf")
```

**Output**: Clean, structured text with metadata (page numbers, sections, etc.)

---

### 2. Text Chunking (`src/chunk.py`)

**Purpose**: Split text into semantic units

**Technology**: LangChain's RecursiveCharacterTextSplitter

**Configuration**:
- `chunk_size`: 500 characters (configurable)
- `chunk_overlap`: 50 characters (preserves context at boundaries)

**Why Chunking?**
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap prevents losing context at splits

**Key Method**:
```python
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_text(text)
```

---

### 3. Embeddings (`src/embed.py`)

**Purpose**: Convert text to numerical vectors that capture semantic meaning

**Model**: all-MiniLM-L6-v2 (sentence-transformers)
- Dimensions: 384
- Fast inference
- Good quality/performance balance

**How It Works**:
```python
embedder = EmbeddingModel()
vectors = embedder.embed_batch(chunks)  # Returns list of 384-dim vectors
```

**Why Embeddings?**
- Finds meaning, not keywords
- "benefits" matches "advantages", "features", etc.
- Enables semantic similarity search

---

### 4. Vector Store (`src/vector_store.py`)

**Purpose**: Store and search embeddings efficiently

**Technology**: Chroma - lightweight vector database

**Operations**:
```python
store = VectorStore()
store.add_chunks(chunks, embeddings)           # Store
results = store.search(query_embedding, n=3)  # Search
```

**How Search Works**:
1. Query converted to embedding
2. Compute cosine similarity with all stored vectors
3. Return top N most similar chunks

---

### 5. Query Interface (`src/query.py`)

**Purpose**: Phase 1 retrieval system (no LLM)

**Flow**:
1. User asks question
2. Question → embedding
3. Search vector store
4. Return top chunks with similarity scores

**Use Case**: Quick search, testing, when no LLM needed

---

### 6. LLM Providers (`src/llm_providers.py`)

**Purpose**: Abstract interface for multiple LLM providers

**Supported Providers**:

| Provider | Class | Use Case |
|----------|-------|----------|
| Ollama | `OllamaLLM` | Local, free, fast |
| OpenAI | `OpenAILLM` | Cloud, highest quality |
| Anthropic | `AnthropicLLM` | Cloud, high quality |
| HuggingFace | `HuggingFaceLLM` | Local, customizable |
| Local Server | `LocalServerLLM` | vLLM, text-gen-webui |
| Mock | `MockLLM` | Testing |

**Design Pattern**: Strategy pattern with `BaseLLM` abstract class

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass
```

**Why This Design?**
- Easy to swap providers
- No vendor lock-in
- Same code works with any LLM

---

### 7. RAG Pipeline (`src/rag.py`)

**Purpose**: Phase 2 - combines retrieval with LLM for natural language answers

**Key Features**:
- **Model-agnostic prompts**: Same prompt template works with all LLMs
- **Context injection**: Retrieved chunks inserted into prompt
- **Grounded responses**: LLM answers based only on provided context

**Prompt Structure**:
```
CONTEXT FROM DOCUMENT:
[Retrieved chunk 1]
[Retrieved chunk 2]
[Retrieved chunk 3]

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the context above
- Be specific and cite details
- Say if information is insufficient

ANSWER:
```

**Why This Works Universally**:
- All LLMs understand natural language instructions
- Clear structure prevents hallucination
- Context grounding improves accuracy

**Flow**:
```python
rag = RAGInterface(embedder, vector_store, llm)
result = rag.answer_question("What is this about?")
# Returns: {"answer": "...", "context": [...]}
```

---

## Data Flow

### Phase 1: Retrieval Only

```
┌─────────┐
│   PDF   │
└────┬────┘
     │ extract.py
┌────▼────┐
│  Text   │
└────┬────┘
     │ chunk.py
┌────▼────┐
│ Chunks  │
└────┬────┘
     │ embed.py
┌────▼────┐
│ Vectors │
└────┬────┘
     │ vector_store.py
┌────▼────────┐
│ Vector DB   │
└────┬────────┘
     │
     │ User Question
     │
┌────▼────────┐
│   Search    │
└────┬────────┘
     │
┌────▼────────┐
│ Top Chunks  │ → Display
└─────────────┘
```

### Phase 2: RAG with LLM

```
┌─────────┐
│   PDF   │
└────┬────┘
     │ extract.py → chunk.py → embed.py
┌────▼────────┐
│ Vector DB   │
└────┬────────┘
     │
     │ User Question
     │
┌────▼────────┐
│   Search    │
└────┬────────┘
     │
┌────▼────────┐
│ Top Chunks  │
└────┬────────┘
     │ rag.py
┌────▼────────┐
│   Prompt    │ = Question + Context + Instructions
└────┬────────┘
     │ llm_providers.py
┌────▼────────┐
│     LLM     │
└────┬────────┘
     │
┌────▼────────┐
│   Answer    │ → Display
└─────────────┘
```

---

## Design Decisions

### Why Sentence Transformers?
- Lightweight (384 dimensions)
- Fast inference (CPU-friendly)
- Good multilingual support
- Open-source, no API needed

### Why Chroma?
- Simple API
- In-memory or persistent
- Good performance for < 1M vectors
- Easy to integrate

### Why Chunking at 500 Characters?
- Balance between context and precision
- Most sentences/paragraphs fit
- Works well with 384-dim embeddings
- Fast retrieval

### Why Model-Agnostic Design?
- Future-proof (new models constantly released)
- User choice (privacy vs quality vs cost)
- No vendor lock-in
- Easy testing (mock mode)

---

## Performance Considerations

### Memory Usage

| Component | RAM Required |
|-----------|--------------|
| Embedding Model | ~500 MB |
| Vector Store (1000 chunks) | ~5 MB |
| LLM (Ollama, 7B) | ~4-8 GB |
| LLM (gpt-oss-20b) | ~16 GB |

**Recommendations**:
- 8GB RAM: Use Ollama with quantized models
- 16GB RAM: Can run gpt-oss-20b
- 32GB RAM: Can run larger models comfortably

### Speed

**Embedding Generation**: ~100-200 chunks/second (CPU)

**Vector Search**: <100ms for 1000 chunks

**LLM Response**:
- Ollama (local): 20-50 tokens/second
- OpenAI (cloud): 100+ tokens/second
- gpt-oss-20b (local): 30-60 tokens/second

**Optimization Tips**:
- Reduce chunk overlap for faster processing
- Use GPU for embedding if available
- Use quantized models for faster inference
- Cache embeddings to avoid reprocessing

---

## Extending the System

### Adding a New LLM Provider

1. Create class extending `BaseLLM`:
```python
class CustomLLM(BaseLLM):
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Your implementation
        pass
    
    def is_available(self) -> bool:
        # Check if provider is configured
        pass
```

2. Add to `llm_providers.py`

3. Use it:
```python
rag = RAGInterface(embedder, store, llm=CustomLLM())
```

### Customizing Prompts

Edit `src/rag.py`:
```python
def build_prompt(self, question, context_chunks):
    # Customize prompt structure here
    pass
```

### Using Different Embedding Models

Edit `src/embed.py`:
```python
model = EmbeddingModel(model_name="all-mpnet-base-v2")  # Higher quality
```

### Persistent Storage

Enable persistent vector store:
```python
store = VectorStore(persist_directory="./embeddings_data")
```

---

## Testing

### Unit Tests

Test individual components:
```python
# Test extraction
parser = PDFParser()
text = parser.extract_text("test.pdf")
assert len(text) > 0

# Test chunking
chunker = TextChunker()
chunks = chunker.chunk_text(text)
assert len(chunks) > 0

# Test embeddings
embedder = EmbeddingModel()
vectors = embedder.embed_batch(chunks[:5])
assert vectors.shape == (5, 384)
```

### Integration Tests

Test full pipeline:
```bash
python test.py data/sample.pdf
```

### Manual Testing

```bash
# Phase 1: Retrieval
python main.py data/sample.pdf

# Phase 2: RAG
python main_rag.py data/sample.pdf
```

---

## Security Considerations

### Local Processing
- When using Ollama/HuggingFace: All data stays on your machine
- No external API calls
- Complete privacy

### Cloud APIs
- Data sent to OpenAI/Anthropic servers
- Review provider's privacy policy
- Consider data sensitivity

### Best Practices
- Use local models for sensitive documents
- Sanitize PDFs before processing
- Don't store API keys in code
- Use environment variables for credentials

---

## Troubleshooting

### Out of Memory
- Reduce chunk size
- Use smaller LLM
- Process fewer chunks
- Use quantized models

### Slow Performance
- Use GPU for embeddings
- Reduce chunk overlap
- Use smaller embedding model
- Use cloud APIs for faster LLM

### Poor Results
- Increase number of retrieved chunks (n_results)
- Use better embedding model (all-mpnet-base-v2)
- Use better LLM (GPT-4, Claude)
- Adjust chunk size for your documents

---

## Future Enhancements

### Potential Improvements
- Multi-PDF support (knowledge base)
- Conversation history (chat mode)
- Hybrid search (keyword + semantic)
- Re-ranking of results
- Fine-tuned embeddings for domain
- Web interface (Streamlit/Gradio)
- Document metadata filtering
- Citation extraction

### Scaling Considerations
- For > 10K documents: Use Pinecone/Weaviate
- For high traffic: Add caching layer
- For production: Add monitoring/logging
- For teams: Add authentication/authorization

---

## References

- **Sentence Transformers**: [sbert.net](https://sbert.net)
- **Chroma**: [docs.trychroma.com](https://docs.trychroma.com)
- **Docling**: PDF parsing library
- **LangChain**: Text splitting utilities
- **RAG Pattern**: Retrieval-Augmented Generation concept

