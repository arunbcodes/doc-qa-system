# Troubleshooting

Common issues and solutions for the PDF Q&A System.

## Installation Issues

### Package Installation Fails

**Problem:** `pip install` fails with dependency conflicts

```bash
ERROR: Cannot install pdf-qa-system because these package versions have conflicting dependencies.
```

**Solutions:**

```bash
# Option 1: Use fresh virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .

# Option 2: Install with --no-deps and resolve manually
pip install -e . --no-deps
pip install -r requirements.txt

# Option 3: Use specific Python version
python3.11 -m venv .venv
```

### PyTorch Installation Issues

**Problem:** PyTorch fails to install or wrong CUDA version

**Solutions:**

```bash
# CPU only (smaller, faster download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# macOS (MPS acceleration)
pip install torch
```

### ChromaDB Installation Fails

**Problem:** ChromaDB compilation errors on Windows/macOS

**Solutions:**

```bash
# Install build tools first
# Windows: Install Visual Studio Build Tools
# macOS: xcode-select --install

# Use pre-built wheels
pip install chromadb --prefer-binary

# Or use alternative backend
pip install chromadb-client
```

## PDF Processing Issues

### Cannot Parse PDF

**Problem:** Error when loading PDF file

```
Error: Failed to process PDF: [Errno 2] No such file or directory
```

**Solutions:**

```bash
# Check file exists
ls -la data/document.pdf

# Use absolute path
python main.py /full/path/to/document.pdf

# Check file permissions
chmod 644 data/document.pdf

# Verify PDF is valid
pdfinfo data/document.pdf  # Install: apt-get install poppler-utils
```

### Corrupted or Encrypted PDFs

**Problem:** PDF cannot be extracted

```
Error: PDF is encrypted or corrupted
```

**Solutions:**

```bash
# Remove password (if you have it)
qpdf --decrypt --password=PASSWORD input.pdf output.pdf

# Repair corrupted PDF
gs -o repaired.pdf -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress input.pdf

# Convert to text first
pdftotext input.pdf output.txt
```

### Unicode/Encoding Errors

**Problem:** Errors with special characters

```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**

```python
# In src/extract.py, add encoding fallback
def parse_pdf(self, pdf_path: str) -> str:
    try:
        # existing code
    except UnicodeDecodeError:
        # Retry with different encoding
        return text.encode('utf-8', errors='ignore').decode('utf-8')
```

## LLM Provider Issues

### OpenAI API Errors

**Problem:** OpenAI API key invalid or rate limited

```
Error: Incorrect API key provided
Error: Rate limit exceeded
```

**Solutions:**

```bash
# Verify API key
echo $OPENAI_API_KEY

# Check key validity
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Add retry logic for rate limits
pip install tenacity

# In code:
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_with_retry(prompt):
    return llm.generate(prompt)
```

### Anthropic Connection Errors

**Problem:** Cannot connect to Anthropic API

```
Error: Connection timeout
```

**Solutions:**

```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test connection
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-sonnet-20240229","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'

# Check proxy settings
unset HTTP_PROXY HTTPS_PROXY
```

### Ollama Not Running

**Problem:** Cannot connect to Ollama

```
Error: Connection refused at localhost:11434
```

**Solutions:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Or use systemd
sudo systemctl start ollama

# Check logs
journalctl -u ollama -f

# Verify model is pulled
ollama list
ollama pull llama3.2
```

### HuggingFace Model Download Fails

**Problem:** Cannot download models

```
Error: Connection timeout when downloading model
```

**Solutions:**

```bash
# Set cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Use mirror (China)
export HF_ENDPOINT=https://hf-mirror.com

# Download manually
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

# Or use local path
embedder = EmbeddingModel(model_name="/path/to/local/model")
```

## Memory Issues

### Out of Memory (OOM)

**Problem:** Process killed due to insufficient memory

```
Killed
Error: CUDA out of memory
```

**Solutions:**

```python
# Reduce batch size
chunker = TextChunker(chunk_size=300)  # Instead of 1000

# Process in batches
for i in range(0, len(chunks), 50):
    batch = chunks[i:i+50]
    embeddings = embedder.embed_batch(batch)

# Use CPU instead of GPU
embedder = EmbeddingModel(device="cpu")

# Clear cache
import torch
torch.cuda.empty_cache()

# Use smaller embedding model
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")  # 384 dims
# Instead of all-mpnet-base-v2 (768 dims)
```

### Docker Memory Limits

**Problem:** Container runs out of memory

**Solutions:**

```bash
# Increase memory limit
docker run --memory="8g" pdf-qa-system

# In docker-compose.yml
services:
  pdf-qa:
    mem_limit: 8g
    memswap_limit: 8g
```

## Performance Issues

### Slow Embedding Generation

**Problem:** Taking too long to embed documents

**Solutions:**

```python
# Use GPU if available
embedder = EmbeddingModel(device="cuda")

# Use faster model
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Reduce chunk size
chunker = TextChunker(chunk_size=300)

# Batch processing
embeddings = embedder.embed_batch(chunks, batch_size=32)
```

### Slow Vector Search

**Problem:** Queries take too long

**Solutions:**

```python
# Reduce number of results
rag = RAGInterface(embedder, store, llm, n_results=2)  # Instead of 5

# Use approximate search (ChromaDB default)
# Already optimized with HNSW index

# Index optimization
store = VectorStore(
    collection_name="docs",
    hnsw_space="cosine",
    hnsw_m=16,  # Reduce for speed
    hnsw_ef_construction=100
)
```

### Slow LLM Response

**Problem:** LLM takes too long to respond

**Solutions:**

```python
# Use faster model
llm = OpenAILLM(model_name="gpt-3.5-turbo")  # Instead of gpt-4

# Reduce max tokens
llm = OpenAILLM(max_tokens=200)

# Use local LLM
llm = OllamaLLM(model_name="llama3.2")

# Stream responses
for chunk in llm.stream(prompt):
    print(chunk, end='', flush=True)
```

## ChromaDB Issues

### Database Locked

**Problem:** ChromaDB database is locked

```
Error: database is locked
```

**Solutions:**

```bash
# Close all connections
# Kill processes using the DB
lsof | grep chroma_db
kill -9 <PID>

# Remove lock file
rm -f chroma_db/*.lock

# Use different collection
store = VectorStore(collection_name="docs_v2")
```

### Collection Not Found

**Problem:** Cannot find collection

```
Error: Collection 'docs' does not exist
```

**Solutions:**

```python
# Create collection if not exists
store = VectorStore(collection_name="docs")
# Already handles creation automatically

# List existing collections
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
print(client.list_collections())

# Reset and recreate
store.reset()  # If method exists
# Or delete directory
rm -rf chroma_db/
```

### Dimension Mismatch

**Problem:** Embedding dimensions don't match

```
Error: Dimension mismatch: expected 384, got 768
```

**Solutions:**

```python
# Use same embedding model as before
embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")  # 384 dims

# Or create new collection
store = VectorStore(collection_name="docs_v2")

# Or clear existing data
rm -rf chroma_db/
```

## Environment Issues

### Environment Variables Not Loading

**Problem:** API keys not found

```
Error: OpenAI API key not found
```

**Solutions:**

```bash
# Load .env file
source .env  # Won't work!

# Use python-dotenv
pip install python-dotenv

# In code:
from dotenv import load_dotenv
load_dotenv()

# Or export manually
export OPENAI_API_KEY="sk-..."

# Verify
echo $OPENAI_API_KEY
env | grep OPENAI
```

### PATH Issues

**Problem:** Command not found

```
bash: pdf-qa: command not found
```

**Solutions:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Or use python -m
python -m src.main_rag data/doc.pdf

# Or add to PATH
export PATH="$PATH:$HOME/.local/bin"
```

## Docker Issues

### Docker Build Fails

**Problem:** Cannot build Docker image

**Solutions:**

```bash
# Clear build cache
docker builder prune -af

# Build without cache
docker build --no-cache -t pdf-qa-system .

# Check Dockerfile syntax
docker build --check .

# Increase memory for build
docker build --memory=8g -t pdf-qa-system .
```

### Volume Permission Issues

**Problem:** Cannot write to mounted volumes

**Solutions:**

```bash
# Fix ownership
docker run --rm -v chroma_data:/data alpine chown -R 1000:1000 /data

# Run as specific user
docker run --user $(id -u):$(id -g) pdf-qa-system

# In docker-compose.yml
services:
  pdf-qa:
    user: "${UID}:${GID}"
```

## Testing Issues

### Tests Failing

**Problem:** Pytest tests fail

**Solutions:**

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_chunk.py::test_basic_chunking -v

# Clear cache
pytest --cache-clear

# Disable warnings
pytest -p no:warnings
```

### Import Errors in Tests

**Problem:** Cannot import src modules

**Solutions:**

```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in pytest.ini / pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

## Getting Help

If you're still stuck:

1. **Check Logs:**
   ```bash
   # Python logs
   python main.py data/doc.pdf 2>&1 | tee debug.log

   # Docker logs
   docker-compose logs -f
   ```

2. **Enable Debug Mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Check System Resources:**
   ```bash
   # Memory
   free -h

   # Disk space
   df -h

   # CPU
   top
   ```

4. **Open an Issue:**
   - Repository: https://github.com/arunbcodes/doc-qa-system/issues
   - Include: Python version, OS, error message, logs
   - Provide: Minimal reproducible example

## Next Steps

- [Configuration Guide](configuration.md)
- [LLM Providers](llm-providers.md)
- [Docker Guide](docker.md)
