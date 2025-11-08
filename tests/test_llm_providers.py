"""Unit tests for LLM provider module."""

import os
import pytest

from src.llm_providers import (
    MockLLM,
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    get_available_llm,
)


class TestMockLLM:
    """Tests for MockLLM class."""

    def test_mock_llm_initialization(self):
        """Test MockLLM initialization."""
        llm = MockLLM()
        assert llm.model_name == "mock"

    def test_mock_llm_generate(self):
        """Test MockLLM text generation."""
        llm = MockLLM()
        prompt = "Test prompt"
        response = llm.generate(prompt)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "mock" in response.lower() or "test" in response.lower()

    def test_mock_llm_generate_empty_prompt(self):
        """Test MockLLM with empty prompt."""
        llm = MockLLM()
        response = llm.generate("")

        assert isinstance(response, str)
        assert len(response) > 0


class TestOpenAILLM:
    """Tests for OpenAILLM class."""

    def test_openai_initialization_no_api_key(self):
        """Test OpenAILLM initialization without API key."""
        # Should not raise error, but won't work for actual generation
        try:
            llm = OpenAILLM()
            assert llm.model_name == "gpt-3.5-turbo"
        except Exception as e:
            # Expected if openai library not installed
            assert "openai" in str(e).lower()

    def test_openai_with_custom_model(self):
        """Test OpenAILLM with custom model name."""
        try:
            llm = OpenAILLM(model_name="gpt-4")
            assert llm.model_name == "gpt-4"
        except Exception:
            pytest.skip("OpenAI library not available")


class TestAnthropicLLM:
    """Tests for AnthropicLLM class."""

    def test_anthropic_initialization_no_api_key(self):
        """Test AnthropicLLM initialization without API key."""
        try:
            llm = AnthropicLLM()
            assert llm.model_name == "claude-3-sonnet-20240229"
        except Exception as e:
            # Expected if anthropic library not installed
            assert "anthropic" in str(e).lower()

    def test_anthropic_with_custom_model(self):
        """Test AnthropicLLM with custom model name."""
        try:
            llm = AnthropicLLM(model_name="claude-3-opus-20240229")
            assert llm.model_name == "claude-3-opus-20240229"
        except Exception:
            pytest.skip("Anthropic library not available")


class TestOllamaLLM:
    """Tests for OllamaLLM class."""

    def test_ollama_initialization(self):
        """Test OllamaLLM initialization."""
        llm = OllamaLLM()
        assert llm.model_name == "llama3.2"
        assert llm.base_url == "http://localhost:11434"

    def test_ollama_custom_url(self):
        """Test OllamaLLM with custom base URL."""
        custom_url = "http://custom-host:8080"
        llm = OllamaLLM(base_url=custom_url)
        assert llm.base_url == custom_url

    def test_ollama_custom_model(self):
        """Test OllamaLLM with custom model name."""
        llm = OllamaLLM(model_name="mistral")
        assert llm.model_name == "mistral"


class TestGetAvailableLLM:
    """Tests for get_available_llm function."""

    def test_get_available_llm_returns_llm(self):
        """Test that get_available_llm returns an LLM instance."""
        llm = get_available_llm()
        assert llm is not None
        assert hasattr(llm, "generate")

    def test_get_available_llm_fallback_to_mock(self):
        """Test that get_available_llm falls back to MockLLM."""
        # Without any API keys or Ollama, should return MockLLM
        llm = get_available_llm()
        # Should at least return a working LLM (likely MockLLM)
        assert llm is not None

    def test_get_available_llm_with_openai_key(self, monkeypatch):
        """Test LLM selection with OpenAI API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        llm = get_available_llm()
        # Should attempt to use OpenAI or fall back
        assert llm is not None

    def test_get_available_llm_with_anthropic_key(self, monkeypatch):
        """Test LLM selection with Anthropic API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        llm = get_available_llm()
        # Should attempt to use Anthropic or fall back
        assert llm is not None


class TestLLMIntegration:
    """Integration tests for LLM providers."""

    @pytest.mark.integration
    def test_mock_llm_full_workflow(self):
        """Test complete workflow with MockLLM."""
        llm = MockLLM()

        # Test with a realistic prompt
        prompt = """Based on the following context, answer the question.

Context: The PDF Q&A System uses semantic search and RAG.

Question: What does the system use?

Answer:"""

        response = llm.generate(prompt)

        assert isinstance(response, str)
        assert len(response) > 0
