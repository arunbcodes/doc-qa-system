#!/bin/bash
# Docker entrypoint script for PDF Q&A System

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PDF Q&A System - Docker Container${NC}"
echo "=================================="

# Check if data directory is mounted
if [ ! -d "/app/data" ]; then
    echo -e "${RED}Error: /app/data directory not found${NC}"
    echo "Please mount a volume with your PDF files: -v \$(pwd)/data:/app/data"
    exit 1
fi

# Check if any PDF files exist
if [ -z "$(ls -A /app/data/*.pdf 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: No PDF files found in /app/data${NC}"
    echo "Please place your PDF files in the data directory"
fi

# Show environment info
echo ""
echo "Environment:"
echo "- Python version: $(python --version)"
echo "- Working directory: $(pwd)"
echo "- User: $(whoami)"
echo ""

# Health check function
health_check() {
    python -c "
import sys
try:
    import docling
    import sentence_transformers
    import chromadb
    from src import PDFParser, EmbeddingModel
    print('✓ All core dependencies available')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
"
}

# Run health check
if [ "$1" = "health" ]; then
    health_check
    exit $?
fi

# If no arguments or help requested, show usage
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage:"
    echo "  Phase 1 (Semantic Search):"
    echo "    docker run -v \$(pwd)/data:/app/data pdf-qa python main.py /app/data/sample.pdf"
    echo ""
    echo "  Phase 2 (RAG with LLM):"
    echo "    docker run -v \$(pwd)/data:/app/data pdf-qa python main_rag.py /app/data/sample.pdf"
    echo ""
    echo "  With API keys:"
    echo "    docker run -e OPENAI_API_KEY=sk-... -v \$(pwd)/data:/app/data pdf-qa python main_rag.py /app/data/sample.pdf"
    echo ""
    echo "  Health check:"
    echo "    docker run pdf-qa health"
    echo ""
    exit 0
fi

# Execute the command
exec "$@"
