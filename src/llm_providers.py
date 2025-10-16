"""
LLM Interface Module
Provides a pluggable interface for different LLM providers.
Supports: OpenAI, Anthropic, Ollama, and other local/cloud models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available/configured."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI LLM.
        
        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.is_available():
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check API key and installation.")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        return self.api_key is not None


class AnthropicLLM(BaseLLM):
    """Anthropic Claude models"""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        """
        Initialize Anthropic LLM.
        
        Args:
            model: Model name (claude-3-opus, claude-3-sonnet, etc.)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.is_available():
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Check API key and installation.")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def is_available(self) -> bool:
        return self.api_key is not None


class OllamaLLM(BaseLLM):
    """Local Ollama models (Llama, Mistral, etc.)"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (llama3.2, mistral, phi3, etc.)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.client = None
        
        if self.is_available():
            try:
                import requests
                self.requests = requests
            except ImportError:
                print("Warning: requests package not installed. Run: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.requests:
            raise RuntimeError("Requests library not available.")
        
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False


class HuggingFaceLLM(BaseLLM):
    """Local HuggingFace models (requires transformers and torch)"""
    
    def __init__(self, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model: HuggingFace model name
        """
        self.model_name = model
        self.model = None
        self.tokenizer = None
        
        if self.is_available():
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                print(f"Loading {model}... (this may take a moment)")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                
                # Detect device and dtype
                if torch.cuda.is_available():
                    device_map = "auto"
                    torch_dtype = torch.float16
                    print("Using GPU acceleration")
                elif torch.backends.mps.is_available():
                    device_map = "mps"
                    torch_dtype = torch.float16
                    print("Using Apple Metal acceleration")
                else:
                    device_map = "cpu"
                    torch_dtype = torch.float32
                    print("Using CPU (this will be slower)")
                
                # Load model with optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True  # Memory optimization
                )
                print(f"✓ Model loaded successfully on {device_map}!")
            except ImportError:
                print("Warning: transformers/torch not installed properly")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        return response[len(prompt):].strip()
    
    def is_available(self) -> bool:
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False


class LocalServerLLM(BaseLLM):
    """
    Local server with OpenAI-compatible API.
    Works with: vLLM, text-generation-webui, FastChat, LM Studio, etc.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000/v1", model: str = "local-model"):
        """
        Initialize local server LLM.
        
        Args:
            base_url: Base URL of your local API server
            model: Model name (can be anything for most local servers)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.requests = None
        
        if self.is_available():
            try:
                import requests
                self.requests = requests
            except ImportError:
                print("Warning: requests package not installed. Run: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.requests:
            raise RuntimeError("Requests library not available.")
        
        try:
            # OpenAI-compatible API format
            response = self.requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            # Try simpler completion endpoint if chat doesn't work
            try:
                response = self.requests.post(
                    f"{self.base_url}/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["choices"][0]["text"]
            except Exception as e2:
                raise RuntimeError(f"Local server generation failed: {e}, {e2}")
    
    def is_available(self) -> bool:
        """Check if local server is running."""
        try:
            import requests
            # Try to ping the server
            response = requests.get(f"{self.base_url.split('/v1')[0]}/health", timeout=2)
            return True
        except:
            # If health endpoint doesn't exist, try models endpoint
            try:
                import requests
                response = requests.get(f"{self.base_url}/models", timeout=2)
                return response.status_code == 200
            except:
                return False


class MockLLM(BaseLLM):
    """Mock LLM for testing without a real model."""
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        return """Based on the provided context, I can answer your question. 
        
        [This is a mock response - replace with a real LLM to get actual answers]
        
        The document contains relevant information about your query. 
        Please configure a real LLM provider (OpenAI, Anthropic, Ollama, etc.) to get detailed answers."""
    
    def is_available(self) -> bool:
        return True


def get_available_llm() -> BaseLLM:
    """
    Auto-detect and return the first available LLM.
    Priority: Ollama (local) > OpenAI > Anthropic > Mock
    Note: Skips HuggingFace in auto-detect to avoid large downloads
    """
    # Try Ollama first (free, local, no downloads)
    ollama = OllamaLLM()
    if ollama.is_available():
        print("✓ Using Ollama (local)")
        return ollama
    
    # Try OpenAI
    openai = OpenAILLM()
    if openai.is_available():
        print("✓ Using OpenAI")
        return openai
    
    # Try Anthropic
    anthropic = AnthropicLLM()
    if anthropic.is_available():
        print("✓ Using Anthropic Claude")
        return anthropic
    
    # Skip HuggingFace in auto-detect (large download)
    # Users can explicitly select it if they want
    
    # Fallback to mock
    print("⚠ No LLM configured. Using mock responses.")
    print("To use a real LLM:")
    print("  - Install Ollama: https://ollama.ai (recommended, free)")
    print("  - Or set OPENAI_API_KEY environment variable")
    print("  - Or set ANTHROPIC_API_KEY environment variable")
    print("  - Or explicitly select HuggingFace (requires large download)")
    return MockLLM()


if __name__ == "__main__":
    # Test which LLMs are available
    print("Checking available LLMs...\n")
    
    llms = [
        ("Ollama (local)", OllamaLLM()),
        ("OpenAI", OpenAILLM()),
        ("Anthropic", AnthropicLLM()),
        ("HuggingFace", HuggingFaceLLM()),
        ("Local Server (port 5000)", LocalServerLLM()),
        ("Mock", MockLLM())
    ]
    
    for name, llm in llms:
        status = "✓ Available" if llm.is_available() else "✗ Not available"
        print(f"{name}: {status}")

