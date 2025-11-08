# LLM Providers API

Pluggable interface for different LLM providers.

## Overview

The LLM providers module offers a unified interface for working with multiple language model providers including OpenAI, Anthropic, Ollama, HuggingFace, and custom servers.

## Base Class

### BaseLLM

Abstract base class that all LLM providers must implement.

```python
from src.llm_providers import BaseLLM

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available/configured."""
        pass
```

## Providers

### OpenAILLM

OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)

**Initialization:**

```python
from src.llm_providers import OpenAILLM

# Default (GPT-3.5 Turbo)
llm = OpenAILLM()

# Specific model
llm = OpenAILLM(model="gpt-4")

# With explicit API key
llm = OpenAILLM(model="gpt-3.5-turbo", api_key="sk-...")
```

**Methods:**

- `generate(prompt, max_tokens=500, temperature=0.7) -> str`
- `is_available() -> bool`

**Example:**

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

llm = OpenAILLM(model="gpt-3.5-turbo")

if llm.is_available():
    response = llm.generate(
        prompt="What is machine learning?",
        max_tokens=200,
        temperature=0.7
    )
    print(response)
```

**Models:**

- `gpt-4` - Highest quality
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4-turbo` - Balance of speed and quality

### AnthropicLLM

Anthropic Claude models

**Initialization:**

```python
from src.llm_providers import AnthropicLLM

# Default (Claude 3 Sonnet)
llm = AnthropicLLM()

# Specific model
llm = AnthropicLLM(model="claude-3-opus-20240229")

# With explicit API key
llm = AnthropicLLM(api_key="sk-ant-...")
```

**Example:**

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

llm = AnthropicLLM(model="claude-3-sonnet-20240229")

response = llm.generate(
    prompt="Explain neural networks",
    max_tokens=300
)
print(response)
```

**Models:**

- `claude-3-opus-20240229` - Highest quality
- `claude-3-sonnet-20240229` - Balanced (default)
- `claude-3-haiku-20240307` - Fastest

### OllamaLLM

Local Ollama models (Llama, Mistral, etc.)

**Initialization:**

```python
from src.llm_providers import OllamaLLM

# Default (llama3.2)
llm = OllamaLLM()

# Specific model
llm = OllamaLLM(model="mistral")

# Custom server
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434"
)
```

**Example:**

```python
# Ensure Ollama is running: ollama serve
llm = OllamaLLM(model="llama3.2")

if llm.is_available():
    response = llm.generate(
        prompt="What is Python?",
        max_tokens=200,
        temperature=0.5
    )
    print(response)
```

**Popular Models:**

- `llama3.2` - Meta's latest
- `mistral` - Fast, high quality
- `phi3` - Microsoft, lightweight
- `codellama` - Code-focused

### HuggingFaceLLM

Open source models from HuggingFace

**Initialization:**

```python
from src.llm_providers import HuggingFaceLLM

# Default model
llm = HuggingFaceLLM()

# Specific model
llm = HuggingFaceLLM(
    model="microsoft/phi-2",
    device="cuda"  # or "cpu"
)
```

**Example:**

```python
llm = HuggingFaceLLM(
    model="microsoft/phi-2",
    device="cpu"
)

response = llm.generate(
    prompt="What is AI?",
    max_tokens=150
)
print(response)
```

**Popular Models:**

- `microsoft/phi-2` - Lightweight, fast
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `tiiuae/falcon-7b-instruct`

### LocalServerLLM

Custom local servers (vLLM, text-generation-webui, etc.)

**Initialization:**

```python
from src.llm_providers import LocalServerLLM

llm = LocalServerLLM(
    base_url="http://localhost:5000",
    model_name="custom-model"
)
```

**Example:**

```python
# vLLM server running on localhost:8000
llm = LocalServerLLM(
    base_url="http://localhost:8000",
    model_name="mistral-7b"
)

response = llm.generate("Explain recursion")
print(response)
```

### MockLLM

Mock LLM for testing

**Initialization:**

```python
from src.llm_providers import MockLLM

llm = MockLLM()
```

**Example:**

```python
llm = MockLLM()

response = llm.generate("Test prompt")
print(response)  # "This is a mock response for testing purposes."
```

## Auto-Detection

The `get_available_llm()` function automatically selects the best available LLM:

```python
from src.llm_providers import get_available_llm

# Auto-detect (checks in order: Ollama, OpenAI, Anthropic, HuggingFace, Mock)
llm = get_available_llm()

print(f"Using: {llm.__class__.__name__}")

response = llm.generate("What is Python?")
print(response)
```

**Detection Order:**

1. Ollama (if running locally)
2. OpenAI (if API key set)
3. Anthropic (if API key set)
4. HuggingFace (fallback)
5. Mock LLM (testing)

## Usage in RAG

```python
from src import RAGInterface, EmbeddingModel, VectorStore
from src.llm_providers import OpenAILLM, OllamaLLM, get_available_llm

embedder = EmbeddingModel()
store = VectorStore(collection_name="docs")

# Option 1: Explicit provider
llm = OpenAILLM(model="gpt-4")
rag = RAGInterface(embedder, store, llm)

# Option 2: Auto-detect
llm = get_available_llm()
rag = RAGInterface(embedder, store, llm)

# Option 3: Let RAG auto-detect
rag = RAGInterface(embedder, store)  # Detects automatically
```

## Comparing Providers

```python
from src.llm_providers import OpenAILLM, AnthropicLLM, OllamaLLM

prompt = "What is machine learning?"

# OpenAI
openai_llm = OpenAILLM()
if openai_llm.is_available():
    print("OpenAI:", openai_llm.generate(prompt, max_tokens=100))

# Anthropic
anthropic_llm = AnthropicLLM()
if anthropic_llm.is_available():
    print("Anthropic:", anthropic_llm.generate(prompt, max_tokens=100))

# Ollama
ollama_llm = OllamaLLM()
if ollama_llm.is_available():
    print("Ollama:", ollama_llm.generate(prompt, max_tokens=100))
```

## Error Handling

```python
from src.llm_providers import OpenAILLM

llm = OpenAILLM()

# Check availability
if not llm.is_available():
    print("OpenAI API key not set")
    print("Set with: export OPENAI_API_KEY='sk-...'")
    exit(1)

# Handle generation errors
try:
    response = llm.generate("What is AI?")
    print(response)
except RuntimeError as e:
    print(f"Generation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Custom Provider

Create your own LLM provider:

```python
from src.llm_providers import BaseLLM

class CustomLLM(BaseLLM):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        # Your custom implementation
        import requests
        response = requests.post(
            self.api_url,
            json={"prompt": prompt, "max_tokens": max_tokens}
        )
        return response.json()["text"]

    def is_available(self) -> bool:
        # Check if your API is available
        try:
            import requests
            response = requests.get(f"{self.api_url}/health")
            return response.status_code == 200
        except:
            return False

# Use it
llm = CustomLLM(api_url="https://my-api.com")
response = llm.generate("Hello!")
```

## Dependencies

- `openai>=1.0.0` - OpenAI provider
- `anthropic>=0.18.0` - Anthropic provider
- `transformers>=4.30.0` - HuggingFace provider
- `requests>=2.31.0` - Ollama and custom servers

## See Also

- [RAGInterface API](rag.md) - Use LLMs in RAG
- [LLM Providers Guide](../user-guide/llm-providers.md) - Detailed setup
- [Configuration](../user-guide/configuration.md) - Environment setup
