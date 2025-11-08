# Testing Guide

Comprehensive guide to testing the PDF Q&A System.

## Test Structure

```
tests/
├── conftest.py           # Pytest fixtures
├── test_extract.py       # PDF extraction tests
├── test_chunk.py         # Text chunking tests
├── test_embed.py         # Embedding tests
├── test_vector_store.py  # Vector store tests
├── test_llm_providers.py # LLM provider tests
├── test_rag.py           # RAG integration tests
└── integration/          # Integration tests
    └── test_e2e.py       # End-to-end tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_chunk.py

# Run specific test
pytest tests/test_chunk.py::test_basic_chunking

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Selection

```bash
# Run by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run by pattern
pytest -k "chunk"       # Tests with "chunk" in name
pytest -k "not test_slow"  # Exclude tests with "slow" in name
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto          # Auto-detect CPU cores
pytest -n 4             # Use 4 processes
```

## Writing Tests

### Test Structure (AAA Pattern)

```python
def test_feature():
    """Test description."""
    # Arrange - Set up test data
    input_data = "test input"
    expected_output = "expected result"

    # Act - Execute the code
    result = function_under_test(input_data)

    # Assert - Verify results
    assert result == expected_output
```

### Using Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def sample_text():
    """Provide sample text for tests."""
    return "This is a test document. It has multiple sentences."

@pytest.fixture
def embedder():
    """Provide embedding model."""
    from src import EmbeddingModel
    return EmbeddingModel()

# test_file.py
def test_with_fixture(sample_text, embedder):
    """Test using fixtures."""
    embedding = embedder.embed_text(sample_text)
    assert embedding.shape == (384,)
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("chunk_size,expected_chunks", [
    (100, 5),
    (200, 3),
    (500, 2),
])
def test_chunk_sizes(chunk_size, expected_chunks):
    """Test different chunk sizes."""
    chunker = TextChunker(chunk_size=chunk_size)
    chunks = chunker.chunk_text(sample_long_text)
    assert len(chunks) == expected_chunks
```

## Unit Tests

### Testing Text Chunking

```python
# tests/test_chunk.py
import pytest
from src import TextChunker

class TestTextChunker:
    """Tests for TextChunker class."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")

        assert chunks == []

    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        chunker = TextChunker(chunk_size=100)
        text = "Test text"
        metadata = {"source": "test.pdf"}

        result = chunker.chunk_with_metadata(text, metadata)

        assert len(result) > 0
        assert result[0]["metadata"]["source"] == "test.pdf"
        assert "chunk_index" in result[0]["metadata"]

    @pytest.mark.parametrize("chunk_size", [100, 300, 500, 1000])
    def test_various_chunk_sizes(self, chunk_size):
        """Test various chunk sizes."""
        chunker = TextChunker(chunk_size=chunk_size)
        text = "word " * 1000

        chunks = chunker.chunk_text(text)
        stats = chunker.get_stats(text)

        assert stats["num_chunks"] > 0
        assert stats["max_chunk_size"] <= chunk_size * 1.1  # Allow 10% overflow
```

### Testing Embeddings

```python
# tests/test_embed.py
import pytest
import numpy as np
from src import EmbeddingModel

class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return EmbeddingModel()

    def test_single_embedding(self, embedder):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_batch_embedding(self, embedder):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed_batch(texts, show_progress=False)

        assert embeddings.shape == (3, 384)

    def test_empty_text(self, embedder):
        """Test embedding empty text."""
        with pytest.raises(ValueError):
            embedder.embed_text("")

    def test_similarity(self, embedder):
        """Test similarity computation."""
        text1 = "Machine learning is fascinating"
        text2 = "Deep learning uses neural networks"
        text3 = "The weather is nice today"

        sim_12 = embedder.compute_similarity(text1, text2)
        sim_13 = embedder.compute_similarity(text1, text3)

        # Related texts should be more similar
        assert sim_12 > sim_13
        assert -1 <= sim_12 <= 1
        assert -1 <= sim_13 <= 1
```

### Testing Vector Store

```python
# tests/test_vector_store.py
import pytest
import numpy as np
from src import VectorStore

class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def store(self):
        """Create in-memory store."""
        return VectorStore(collection_name="test_collection")

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        embeddings = [np.random.rand(384) for _ in range(3)]
        metadatas = [{"index": i} for i in range(3)]
        return chunks, embeddings, metadatas

    def test_add_chunks(self, store, sample_data):
        """Test adding chunks."""
        chunks, embeddings, metadatas = sample_data
        store.add_chunks(chunks, embeddings, metadatas)

        assert store.get_count() == 3

    def test_search(self, store, sample_data):
        """Test searching."""
        chunks, embeddings, metadatas = sample_data
        store.add_chunks(chunks, embeddings, metadatas)

        query_emb = np.random.rand(384)
        results = store.search(query_emb, n_results=2)

        assert len(results['documents'][0]) == 2
        assert len(results['distances'][0]) == 2

    def test_clear(self, store, sample_data):
        """Test clearing store."""
        chunks, embeddings, metadatas = sample_data
        store.add_chunks(chunks, embeddings, metadatas)

        assert store.get_count() == 3

        store.clear()
        assert store.get_count() == 0
```

## Integration Tests

### End-to-End Pipeline

```python
# tests/integration/test_e2e.py
import pytest
from pathlib import Path
from src import PDFParser, TextChunker, EmbeddingModel, VectorStore, RAGInterface
from src.llm_providers import MockLLM

@pytest.mark.integration
class TestE2EWorkflow:
    """End-to-end integration tests."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a sample PDF for testing."""
        # You would create or copy a test PDF here
        pdf_path = tmp_path / "test.pdf"
        # Create PDF or copy from test fixtures
        return pdf_path

    def test_complete_pipeline(self, sample_pdf):
        """Test complete RAG pipeline."""
        # 1. Extract
        parser = PDFParser()
        text = parser.extract_text(str(sample_pdf))
        assert len(text) > 0

        # 2. Chunk
        chunker = TextChunker(chunk_size=500)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0

        # 3. Embed
        embedder = EmbeddingModel()
        embeddings = embedder.embed_batch(chunks, show_progress=False)
        assert embeddings.shape[0] == len(chunks)

        # 4. Store
        store = VectorStore(collection_name="test_e2e")
        store.add_chunks(chunks, embeddings)
        assert store.get_count() == len(chunks)

        # 5. RAG
        llm = MockLLM()
        rag = RAGInterface(embedder, store, llm, n_results=3)

        result = rag.answer_question("What is this document about?")
        assert "answer" in result
        assert len(result["answer"]) > 0
```

## Mocking

### Mocking LLM Responses

```python
from unittest.mock import Mock, patch

def test_with_mocked_llm():
    """Test with mocked LLM."""
    mock_llm = Mock()
    mock_llm.generate.return_value = "Mocked response"
    mock_llm.is_available.return_value = True

    rag = RAGInterface(embedder, store, mock_llm)
    result = rag.answer_question("Test question")

    assert result["answer"] == "Mocked response"
    mock_llm.generate.assert_called_once()
```

### Mocking External APIs

```python
@patch('src.llm_providers.OpenAI')
def test_openai_provider(mock_openai):
    """Test OpenAI provider with mocked API."""
    # Setup mock
    mock_client = Mock()
    mock_openai.return_value = mock_client

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_response

    # Test
    llm = OpenAILLM(api_key="test-key")
    response = llm.generate("Test prompt")

    assert response == "Test response"
```

## Test Markers

```python
# Mark tests
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

# Run specific markers
# pytest -m unit
# pytest -m "not slow"
```

## Coverage Goals

Maintain minimum 80% code coverage:

```bash
# Run with coverage
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80
```

## Performance Testing

```python
import pytest
import time

def test_performance():
    """Test performance requirements."""
    chunker = TextChunker(chunk_size=500)
    text = "word " * 10000

    start = time.time()
    chunks = chunker.chunk_text(text)
    duration = time.time() - start

    # Should complete in < 1 second
    assert duration < 1.0
    assert len(chunks) > 0
```

## Test Data

### Fixtures Directory

```
tests/
├── fixtures/
│   ├── sample.pdf
│   ├── sample_text.txt
│   └── expected_output.json
└── conftest.py
```

### Loading Fixtures

```python
# conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    """Get fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_pdf(fixtures_dir):
    """Load sample PDF."""
    return fixtures_dir / "sample.pdf"

# tests/test_parser.py
def test_parser(sample_pdf):
    """Test with fixture."""
    parser = PDFParser()
    text = parser.extract_text(str(sample_pdf))
    assert len(text) > 0
```

## Continuous Integration

Tests run automatically on:

- Every push
- Every pull request
- Scheduled (daily)

See `.github/workflows/ci.yml` for configuration.

## Best Practices

1. **Test one thing**: Each test should verify one behavior
2. **Independent tests**: Tests should not depend on each other
3. **Fast tests**: Keep tests fast (<1s per test when possible)
4. **Clear names**: Test names should describe what they test
5. **Arrange-Act-Assert**: Follow AAA pattern
6. **Use fixtures**: Reuse test setup with fixtures
7. **Mock external services**: Don't hit real APIs in tests
8. **Test edge cases**: Empty input, None, max values, etc.
9. **Maintain coverage**: Keep coverage >80%
10. **Update tests with code**: Tests are code too!

## Debugging Failed Tests

```bash
# Run with verbose output
pytest -vv

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Run last failed tests
pytest --lf

# Run only failed tests
pytest --failed-first
```

## Next Steps

- [Contributing Guide](contributing.md)
- [Code Style Guide](code-style.md)
- [API Reference](../api/overview.md)
