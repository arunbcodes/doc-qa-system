# Contributing Guide

Thank you for your interest in contributing to the PDF Q&A System!

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/doc-qa-system.git
cd doc-qa-system

# Add upstream remote
git remote add upstream https://github.com/arunbcodes/doc-qa-system.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs,llm]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

## Development Workflow

### 1. Make Changes

```bash
# Edit files
vim src/your_module.py

# Run tests frequently
pytest tests/test_your_module.py -v

# Check code style
black src/
isort src/
flake8 src/
```

### 2. Write Tests

All new code should include tests. See [Testing Guide](testing.md) for details.

```python
# tests/test_your_feature.py
def test_your_feature():
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = your_function(input_data)

    # Assert
    assert result == expected
```

### 3. Update Documentation

```python
# Add docstrings
def your_function(param: str) -> int:
    """
    Brief description.

    Args:
        param: Parameter description

    Returns:
        Return value description

    Example:
        >>> your_function("test")
        42
    """
    pass
```

```bash
# Update user docs if needed
vim docs/user-guide/your-feature.md

# Build docs locally
mkdocs serve
# Visit http://localhost:8000
```

### 4. Run Pre-commit Checks

```bash
# Pre-commit hooks run automatically on commit
git add .
git commit -m "Add your feature"

# Or run manually
pre-commit run --all-files
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Open Pull Request on GitHub
# Fill in the PR template
```

## Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:

- **black**: Code formatting (line length 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### Code Quality

- **Docstrings**: All public functions/classes must have docstrings
- **Type hints**: Use type hints for function parameters and returns
- **Tests**: Maintain >80% code coverage
- **Comments**: Explain "why", not "what"

**Good Example:**

```python
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts.

    Args:
        text1: First text for comparison
        text2: Second text for comparison

    Returns:
        Similarity score between -1 and 1

    Raises:
        ValueError: If either text is empty

    Example:
        >>> calculate_similarity("hello", "hello world")
        0.87
    """
    if not text1 or not text2:
        raise ValueError("Cannot compare empty texts")

    # Use embeddings for semantic similarity (not just string matching)
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    return cosine_similarity(emb1, emb2)
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Feature
git commit -m "feat(rag): add chat history support"

# Bug fix
git commit -m "fix(chunk): handle empty text correctly"

# Documentation
git commit -m "docs: update installation guide"

# With body
git commit -m "feat(llm): add streaming support

Implements streaming responses for compatible LLM providers.
Adds stream() method to BaseLLM interface.

Closes #123"
```

### Commit Best Practices

- Keep commits small and focused
- One logical change per commit
- Write clear, descriptive messages
- Reference issues when applicable

## Pull Request Guidelines

### PR Title

Use the same format as commit messages:

```
feat(module): add new feature
fix(module): resolve issue with X
```

### PR Description Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
```

### Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. Address review feedback
4. Maintainer merges PR

## Testing Requirements

See [Testing Guide](testing.md) for detailed information.

### Minimum Requirements

- All tests must pass: `pytest`
- Coverage must be >80%: `pytest --cov=src`
- No linting errors: `flake8 src/`
- Type checks pass: `mypy src/`

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_chunk.py::test_basic_chunking -v

# With coverage
pytest --cov=src --cov-report=html

# Integration tests
pytest tests/ -m integration
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 0) -> bool:
    """
    Brief one-line description.

    More detailed description if needed.
    Can span multiple lines.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input
        RuntimeError: When operation fails

    Example:
        >>> function("test", 5)
        True

    Note:
        Additional information
    """
    pass
```

### User Documentation

- Add new features to relevant user guide pages
- Include code examples
- Update API reference if needed
- Build docs locally to verify: `mkdocs serve`

## Issue Guidelines

### Reporting Bugs

Include:

- Python version
- OS/platform
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Minimal reproducible example

### Requesting Features

Include:

- Use case / problem to solve
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

## Development Tips

### Quick Development Cycle

```bash
# Watch for changes and auto-run tests
pytest-watch

# Or use entr
find src tests -name "*.py" | entr pytest
```

### Debugging

```python
# Use pytest with pdb
pytest --pdb

# Or add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb
import ipdb; ipdb.set_trace()
```

### Performance Profiling

```python
# Profile code
python -m cProfile -o profile.stats main.py

# View results
python -m pstats profile.stats
>>> sort cumulative
>>> stats 10
```

## Getting Help

- **Documentation**: Check [docs](https://arunbcodes.github.io/doc-qa-system/)
- **Issues**: Search [existing issues](https://github.com/arunbcodes/doc-qa-system/issues)
- **Discussions**: Start a [discussion](https://github.com/arunbcodes/doc-qa-system/discussions)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and improve the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be added to:

- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

Thank you for contributing! 🎉
