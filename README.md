# PDF Q&A System - Phase 1 MVP

A local PDF question-answering system that extracts text from PDFs, converts it to embeddings, and enables semantic search through document content.

## Features

- **PDF Text Extraction**: Uses docling to extract clean, structured text from PDF files
- **Intelligent Chunking**: Splits text into manageable chunks with overlap for better context
- **Local Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for vector embeddings
- **Vector Search**: Stores and searches chunks using Chroma vector database
- **Interactive CLI**: Simple command-line interface for querying documents

## Architecture

```
PDF → Docling Parser → Text Chunker → Embedding Model → Vector Store → Query Interface
```

### Components

1. **PDF Parser** (`pdf_parser.py`) - Extracts text from PDF using docling
2. **Text Chunker** (`text_chunker.py`) - Splits text using LangChain's RecursiveCharacterTextSplitter
3. **Embeddings** (`embeddings.py`) - Generates embeddings using sentence-transformers
4. **Vector Store** (`vector_store.py`) - Manages Chroma vector database
5. **Query Interface** (`query_interface.py`) - Handles search and retrieval
6. **Main CLI** (`main.py`) - Orchestrates the entire workflow

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the system with a PDF file:

```bash
python main.py path/to/your/document.pdf
```

The system will:
1. Extract and process the PDF content
2. Display a summary of chunks created
3. Enter an interactive query mode

### Query Mode

Once the PDF is processed, you can ask questions:

```
> What is the main topic of this document?
[System displays top 3 relevant chunks]

> Tell me about the methodology
[System displays relevant sections]

> quit
```

Type `quit` or `exit` to end the session.

## Configuration

You can adjust parameters in the respective modules:

- **Chunk size/overlap**: Edit `text_chunker.py`
- **Number of results**: Edit `query_interface.py`
- **Embedding model**: Edit `embeddings.py`

## Phase 1 Scope

This MVP processes **one PDF per session** with in-memory storage. The architecture is designed to be modular for easy expansion.

## Future Enhancements (Phase 2)

- Persistent vector storage across sessions
- Multi-PDF knowledge base
- Document management (add, list, delete)
- Integration with local LLM for answer generation
- Web interface

## Requirements

- Python 3.8+
- ~500MB disk space for models (first run downloads)
- 2GB+ RAM recommended

## License

MIT

