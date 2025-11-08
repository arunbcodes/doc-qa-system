# Installation

This guide covers different ways to install the PDF Q&A System.

## Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB+ recommended for large models)
- 10GB disk space (for models)

## Installation Methods

### Method 1: Package Installation (Recommended)

Install the package with pip:

```bash
# Basic installation
pip install -e .

# With LLM provider support
pip install -e ".[llm]"

# With development tools
pip install -e ".[dev]"

# Full installation (all dependencies)
pip install -e ".[all]"
```

### Method 2: Dependencies Only

If you just want to run the scripts without installing as a package:

```bash
pip install -r requirements.txt
```

### Method 3: Docker

Use Docker for isolated deployment:

```bash
# Build the image
docker build -t pdf-qa-system .

# Run Phase 1 (Semantic Search)
docker run -it --rm -v $(pwd)/data:/app/data pdf-qa-system python main.py /app/data/sample.pdf

# Run Phase 2 (RAG with LLM)
docker run -it --rm \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  pdf-qa-system python main_rag.py /app/data/sample.pdf
```

See [Docker Usage](../user-guide/docker.md) for more details.

## Dependency Groups

The project uses optional dependency groups for different use cases:

### Core Dependencies

Installed by default with `pip install -e .`:

- `docling>=1.0.0` - PDF parsing
- `langchain-text-splitters>=0.2.0` - Text chunking
- `sentence-transformers>=2.2.0` - Embeddings
- `torch>=2.0.0` - ML framework
- `chromadb>=0.4.0` - Vector database
- `numpy>=1.24.0` - Numerical computing

### LLM Dependencies (`[llm]`)

For LLM provider support:

- `openai>=1.0.0` - OpenAI API
- `anthropic>=0.18.0` - Anthropic API
- `transformers>=4.30.0` - HuggingFace models
- `requests>=2.31.0` - HTTP for Ollama

### Development Dependencies (`[dev]`)

For development and testing:

- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.4.0` - Type checking
- `isort>=5.12.0` - Import sorting

## Verification

Verify your installation:

```bash
# Check Python version
python --version

# Verify package installation
pip show pdf-qa-system

# Run tests (if dev dependencies installed)
pytest

# Check Docker installation
docker --version
```

## Platform-Specific Notes

### macOS

On Apple Silicon (M1/M2), ensure you're using native ARM packages:

```bash
# Check architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64
```

### Windows

Use PowerShell or Command Prompt:

```powershell
# Activate virtual environment
venv\Scripts\activate

# Install package
pip install -e .
```

### Linux

Install system dependencies first:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Fedora/RHEL
sudo dnf install python3-devel gcc gcc-c++
```

## Troubleshooting

### Import Errors

If you see import errors after installation:

```bash
# Reinstall in editable mode
pip install -e . --force-reinstall
```

### PyTorch Installation Issues

For GPU support or specific PyTorch versions:

```bash
# CPU-only (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Docker Build Fails

If Docker build is slow or fails:

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t pdf-qa-system .

# Clear cache if needed
docker builder prune
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Basic Usage](basic-usage.md)
- [Configuration](../user-guide/configuration.md)
