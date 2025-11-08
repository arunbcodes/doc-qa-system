# LLM Providers

Guide to configuring and using different LLM providers.

## Supported Providers

The system supports 6 LLM providers out of the box:

1. **OpenAI** (GPT-3.5, GPT-4)
2. **Anthropic** (Claude)
3. **Ollama** (Local LLMs)
4. **HuggingFace** (Open models)
5. **Local Server** (vLLM, text-generation-webui)
6. **Mock LLM** (Testing)

## Auto-Detection

The system automatically selects the best available LLM:

```python
from src import get_available_llm

# Automatically detects in this order:
# 1. Ollama (if running)
# 2. OpenAI (if API key set)
# 3. Anthropic (if API key set)
# 4. HuggingFace (fallback)
# 5. Mock LLM (testing)
llm = get_available_llm()
```

## OpenAI

### Setup

```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```

### Usage

```python
from src.llm_providers import OpenAILLM

# Default (GPT-3.5 Turbo)
llm = OpenAILLM()

# GPT-4
llm = OpenAILLM(model_name="gpt-4")

# Custom configuration
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=500
)
```

### Available Models

- `gpt-3.5-turbo` - Fast, cost-effective
- `gpt-4` - Higher quality, slower
- `gpt-4-turbo` - Balance of speed and quality

**Pricing:** https://openai.com/pricing

## Anthropic (Claude)

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

### Usage

```python
from src.llm_providers import AnthropicLLM

# Default (Claude 3 Sonnet)
llm = AnthropicLLM()

# Claude 3 Opus (highest quality)
llm = AnthropicLLM(model_name="claude-3-opus-20240229")

# Custom configuration
llm = AnthropicLLM(
    model_name="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    max_tokens=1000
)
```

### Available Models

- `claude-3-opus-20240229` - Highest quality
- `claude-3-sonnet-20240229` - Balanced (default)
- `claude-3-haiku-20240307` - Fastest

**Pricing:** https://www.anthropic.com/pricing

## Ollama (Local)

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
```

### Usage

```python
from src.llm_providers import OllamaLLM

# Default (llama3.2)
llm = OllamaLLM()

# Custom model
llm = OllamaLLM(model_name="mistral")

# Custom server
llm = OllamaLLM(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.5
)
```

### Available Models

- `llama3.2` - Meta's latest (default)
- `mistral` - Fast, high quality
- `codellama` - Code-focused
- `phi3` - Microsoft, lightweight
- `gemma2` - Google

List all: `ollama list`

**Benefits:**
- 100% local and private
- No API costs
- Works offline
- No rate limits

## HuggingFace

### Setup

```bash
export HF_TOKEN="hf_..."  # Optional
pip install transformers accelerate
```

### Usage

```python
from src.llm_providers import HuggingFaceLLM

# Default model
llm = HuggingFaceLLM()

# Custom model
llm = HuggingFaceLLM(
    model_name="microsoft/phi-2",
    device="cuda"  # or "cpu"
)
```

### Popular Models

- `microsoft/phi-2` - Lightweight, fast
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `tiiuae/falcon-7b-instruct`

**Note:** First run downloads the model (~3-14GB)

## Local Server

For vLLM, text-generation-webui, or custom servers:

```python
from src.llm_providers import LocalServerLLM

llm = LocalServerLLM(
    base_url="http://localhost:5000",
    model_name="custom-model"
)
```

### Compatible Servers

- **vLLM**: High-throughput inference
- **text-generation-webui**: Gradio interface
- **llama.cpp server**: C++ implementation
- **Custom FastAPI**: Your own server

## Mock LLM (Testing)

For testing without real LLM:

```python
from src.llm_providers import MockLLM

llm = MockLLM()
response = llm.generate("Test prompt")
# Returns: "This is a mock response for testing purposes."
```

## Comparison

| Provider | Cost | Privacy | Speed | Quality | Setup |
|----------|------|---------|-------|---------|-------|
| OpenAI | $$$ | Cloud | Fast | Excellent | Easy |
| Anthropic | $$$ | Cloud | Fast | Excellent | Easy |
| Ollama | Free | 100% Local | Medium | Good | Medium |
| HuggingFace | Free | 100% Local | Slow | Varies | Easy |
| Local Server | Free | 100% Local | Varies | Varies | Hard |
| Mock | Free | Local | Instant | None | Instant |

## Switching Providers

### Via Environment Variables

```bash
# Use OpenAI
export OPENAI_API_KEY="sk-..."
python main_rag.py data/doc.pdf

# Use Anthropic
unset OPENAI_API_KEY
export ANTHROPIC_API_KEY="sk-ant-..."
python main_rag.py data/doc.pdf

# Use Ollama (no keys needed)
unset OPENAI_API_KEY ANTHROPIC_API_KEY
python main_rag.py data/doc.pdf
```

### Via Code

```python
from src import RAGInterface, get_available_llm
from src.llm_providers import OpenAILLM, OllamaLLM

# Explicitly choose provider
llm = OpenAILLM(model_name="gpt-4")
# or
llm = OllamaLLM(model_name="mistral")

# Use in RAG
rag = RAGInterface(embedder, store, llm)
```

## Best Practices

### For Development

```python
# Use Mock LLM for fast iteration
from src.llm_providers import MockLLM
llm = MockLLM()
```

### For Production (Cloud)

```python
# OpenAI for reliability
llm = OpenAILLM(
    model_name="gpt-3.5-turbo",
    temperature=0.3  # More deterministic
)
```

### For Production (Self-Hosted)

```python
# Ollama for privacy and cost
llm = OllamaLLM(
    model_name="llama3.2",
    base_url="http://llm-server:11434"
)
```

### For High Volume

```python
# Use local server with vLLM
llm = LocalServerLLM(
    base_url="http://vllm-server:8000",
    model_name="mistral-7b"
)
```

## Troubleshooting

### OpenAI Rate Limits

```python
# Add retry logic
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def generate_with_retry(prompt):
    return llm.generate(prompt)
```

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Check logs
journalctl -u ollama -f
```

### Out of Memory (Local Models)

```python
# Use smaller model
llm = HuggingFaceLLM(model_name="microsoft/phi-2")

# Or use 8-bit quantization
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    load_in_8bit=True
)
```

## Next Steps

- [Configuration Guide](configuration.md)
- [Docker Setup](docker.md)
- [API Reference](../api/llm-providers.md)
