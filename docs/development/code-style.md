# Code Style Guide

Code style standards and best practices for the PDF Q&A System.

## Overview

We follow [PEP 8](https://pep8.org/) with tools to enforce consistency:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## Auto-Formatting

### Black

Format code with black (line length: 100):

```bash
# Format all code
black src/ tests/

# Check without modifying
black --check src/

# Format specific file
black src/chunk.py
```

### isort

Sort imports with isort:

```bash
# Sort all imports
isort src/ tests/

# Check without modifying
isort --check src/

# Sort specific file
isort src/chunk.py
```

### Combined

```bash
# Format and sort
black src/ tests/ && isort src/ tests/
```

## Code Style

### Naming Conventions

```python
# Classes: PascalCase
class TextChunker:
    pass

class RAGInterface:
    pass

# Functions and variables: snake_case
def chunk_text(text: str) -> List[str]:
    chunk_size = 500
    overlap = 50
    return chunks

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 2000
DEFAULT_TEMPERATURE = 0.7

# Private methods: _leading_underscore
def _internal_helper(self):
    pass

# Protected attributes: _leading_underscore
self._cache = {}
```

### Line Length

Maximum 100 characters per line:

```python
# Good
result = some_function(
    arg1="value",
    arg2="another value",
    arg3="yet another value"
)

# Bad
result = some_function(arg1="value", arg2="another value", arg3="yet another value", arg4="more")
```

### Imports

```python
# Order: standard library, third-party, local
import os
import sys
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embed import EmbeddingModel
from src.chunk import TextChunker

# Group imports
from typing import (
    Dict,
    List,
    Optional,
    Union,
)
```

### Docstrings

Google-style docstrings for all public functions/classes:

```python
def embed_text(self, text: str) -> np.ndarray:
    """
    Generate embedding for a single text.

    Args:
        text: Input text to embed

    Returns:
        Embedding vector as numpy array

    Raises:
        ValueError: If text is empty

    Example:
        >>> embedder = EmbeddingModel()
        >>> embedding = embedder.embed_text("Hello world")
        >>> embedding.shape
        (384,)
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Union

def process_chunks(
    chunks: List[str],
    metadata: Optional[Dict[str, str]] = None,
    batch_size: int = 100
) -> List[Dict[str, Union[str, int]]]:
    """Process chunks with metadata."""
    pass

# For complex types
from typing import Callable, TypeVar

T = TypeVar('T')

def process_batch(
    items: List[T],
    processor: Callable[[T], str]
) -> List[str]:
    """Process batch of items."""
    pass
```

### String Formatting

Prefer f-strings for string formatting:

```python
# Good
name = "John"
age = 30
message = f"Hello {name}, you are {age} years old"

# Acceptable for logging
logger.info("Processing chunk %d of %d", idx, total)

# Avoid
message = "Hello " + name + ", you are " + str(age) + " years old"
message = "Hello {}, you are {} years old".format(name, age)
```

### Classes

```python
class VectorStore:
    """
    Manage Chroma vector database.

    Attributes:
        collection_name: Name of the collection
        persist_directory: Storage directory path

    Example:
        >>> store = VectorStore(collection_name="docs")
        >>> store.add_chunks(chunks, embeddings)
    """

    def __init__(
        self,
        collection_name: str = "pdf_chunks",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Chroma client (private method)."""
        pass
```

### Error Handling

```python
# Specific exceptions
try:
    text = parser.extract_text(pdf_path)
except FileNotFoundError:
    logger.error(f"PDF not found: {pdf_path}")
    raise
except PermissionError:
    logger.error(f"Cannot read PDF: {pdf_path}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError(f"Failed to process PDF: {e}") from e

# Don't catch generic Exception unless re-raising
# Bad
try:
    do_something()
except Exception:
    pass  # Silent failure

# Good
try:
    do_something()
except SpecificError as e:
    logger.warning(f"Expected error: {e}")
    # Handle appropriately
```

### Context Managers

Use context managers for resource management:

```python
# Good
with open(file_path) as f:
    content = f.read()

# For custom resources
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Time a block of code."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"{name} took {duration:.2f}s")

# Usage
with timer("embedding"):
    embeddings = embedder.embed_batch(chunks)
```

### List Comprehensions

```python
# Good - simple and readable
squares = [x ** 2 for x in range(10)]
even_numbers = [x for x in numbers if x % 2 == 0]

# Avoid - too complex
result = [
    process_item(item, config)
    for category in categories
    for item in category.items
    if item.is_valid() and item.score > threshold
]

# Better - use explicit loop
result = []
for category in categories:
    for item in category.items:
        if item.is_valid() and item.score > threshold:
            result.append(process_item(item, config))
```

### Function Length

Keep functions focused and short (ideally <50 lines):

```python
# Good - single responsibility
def validate_pdf(path: str) -> bool:
    """Validate PDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.endswith('.pdf'):
        raise ValueError(f"Not a PDF file: {path}")
    return True

def extract_text(path: str) -> str:
    """Extract text from PDF."""
    validate_pdf(path)
    result = converter.convert(path)
    return result.document.export_to_markdown()

# Bad - too many responsibilities
def process_pdf(path: str) -> str:
    # 100 lines of validation, extraction, processing...
    pass
```

### Comments

```python
# Comments explain WHY, not WHAT

# Good - explains reasoning
def calculate_similarity(emb1, emb2):
    # Use cosine similarity instead of euclidean distance
    # because it's scale-invariant
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Bad - restates code
def calculate_similarity(emb1, emb2):
    # Calculate dot product
    dot = np.dot(emb1, emb2)
    # Calculate norms
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    # Return result
    return dot / (norm1 * norm2)
```

### Constants

```python
# Define at module level
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_TOKENS = 4096

# Use in code
def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
    self.chunk_size = chunk_size

# Not
def __init__(self, chunk_size: int = 500):  # Magic number
    self.chunk_size = chunk_size
```

## Linting

### Flake8

```bash
# Check code
flake8 src/ tests/

# Configuration in pyproject.toml or .flake8
[flake8]
max-line-length = 100
max-complexity = 10
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv
```

Common issues:

- `E501`: Line too long
- `F401`: Imported but unused
- `E302`: Expected 2 blank lines
- `F841`: Local variable assigned but never used

### Mypy

```bash
# Type check
mypy src/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

## Pre-commit Hooks

Automatically run checks on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Configuration in .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
```

## Editor Configuration

### VS Code

```json
// .vscode/settings.json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

Settings → Editor → Code Style → Python:

- Line length: 100
- Use spaces: Yes
- Indent: 4

## Best Practices

### 1. Keep It Simple (KISS)

```python
# Good
def is_valid(text: str) -> bool:
    return bool(text and text.strip())

# Overcomplicated
def is_valid(text: str) -> bool:
    if text is not None:
        if len(text) > 0:
            if text.strip() != "":
                return True
    return False
```

### 2. Don't Repeat Yourself (DRY)

```python
# Good
def process_pdf(path: str, chunk_size: int) -> List[str]:
    text = extract_text(path)
    return chunk_text(text, chunk_size)

# Bad - repeated logic
def process_pdf_small(path: str) -> List[str]:
    text = extract_text(path)
    return chunk_text(text, 300)

def process_pdf_medium(path: str) -> List[str]:
    text = extract_text(path)
    return chunk_text(text, 500)
```

### 3. Single Responsibility

Each function/class should do one thing well:

```python
# Good - focused responsibilities
class PDFParser:
    """Extract text from PDFs."""
    def extract_text(self, path: str) -> str:
        pass

class TextChunker:
    """Split text into chunks."""
    def chunk_text(self, text: str) -> List[str]:
        pass

# Bad - too many responsibilities
class PDFProcessor:
    """Does everything."""
    def extract_text(self, path: str) -> str:
        pass
    def chunk_text(self, text: str) -> List[str]:
        pass
    def embed_chunks(self, chunks: List[str]):
        pass
    def store_embeddings(self, embeddings):
        pass
```

### 4. Explicit Over Implicit

```python
# Good - clear intent
def embed_batch(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    pass

# Bad - unclear behavior
def embed(*args, **kwargs):
    pass
```

### 5. Fail Fast

```python
# Good
def process(text: str, chunks: List[str]):
    if not text:
        raise ValueError("Text cannot be empty")
    if not chunks:
        raise ValueError("Chunks cannot be empty")
    # Continue processing

# Bad
def process(text: str, chunks: List[str]):
    # Process for a while
    if text:  # Check late
        # More processing
        if chunks:  # Check even later
            pass
```

## Cheatsheet

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check code
flake8 src/ tests/
mypy src/

# Run tests
pytest

# All checks
black --check src/ && isort --check src/ && flake8 src/ && mypy src/ && pytest
```

## Next Steps

- [Contributing Guide](contributing.md)
- [Testing Guide](testing.md)
- [API Reference](../api/overview.md)
