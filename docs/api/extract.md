# PDFParser API

Extract text from PDF files using docling.

## Class: PDFParser

::: src.extract.PDFParser

### Overview

The `PDFParser` class provides a clean interface for extracting text from PDF files using the docling library. It handles PDF conversion and exports the content as markdown for better structure preservation.

## Basic Usage

```python
from src import PDFParser

# Initialize parser
parser = PDFParser()

# Extract text
text = parser.extract_text("document.pdf")
print(f"Extracted {len(text)} characters")
```

## Methods

### `__init__()`

Initialize the PDF parser.

**Parameters:** None

**Returns:** `PDFParser` instance

**Example:**

```python
parser = PDFParser()
```

### `extract_text(pdf_path: str) -> str`

Extract text from a PDF file.

**Parameters:**

- `pdf_path` (str): Path to the PDF file

**Returns:**

- `str`: Extracted text as markdown

**Raises:**

- `FileNotFoundError`: If the PDF file doesn't exist
- `Exception`: If extraction fails

**Example:**

```python
try:
    text = parser.extract_text("data/document.pdf")
    print(text)
except FileNotFoundError:
    print("PDF file not found")
except Exception as e:
    print(f"Extraction failed: {e}")
```

### `extract_with_metadata(pdf_path: str) -> dict`

Extract text along with metadata from PDF.

**Parameters:**

- `pdf_path` (str): Path to the PDF file

**Returns:**

- `dict`: Dictionary containing:
  - `text` (str): Extracted text content
  - `metadata` (dict): Metadata including:
    - `source` (str): Original file path
    - `num_pages` (int|None): Number of pages if available

**Raises:**

- `Exception`: If extraction fails

**Example:**

```python
result = parser.extract_with_metadata("data/document.pdf")
print(f"Text: {result['text'][:100]}...")
print(f"Pages: {result['metadata']['num_pages']}")
print(f"Source: {result['metadata']['source']}")
```

## Complete Example

```python
from src import PDFParser

# Initialize
parser = PDFParser()

# Extract with metadata
result = parser.extract_with_metadata("research_paper.pdf")

text = result["text"]
metadata = result["metadata"]

print(f"Extracted from: {metadata['source']}")
print(f"Number of pages: {metadata['num_pages']}")
print(f"Total characters: {len(text)}")
print(f"\nFirst 500 characters:\n{text[:500]}")
```

## Integration with Pipeline

```python
from src import PDFParser, TextChunker

# Extract text
parser = PDFParser()
result = parser.extract_with_metadata("document.pdf")

# Pass to chunker
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_with_metadata(
    result["text"],
    source_metadata=result["metadata"]
)

print(f"Created {len(chunks)} chunks from {result['metadata']['num_pages']} pages")
```

## Error Handling

```python
import os
from src import PDFParser

parser = PDFParser()
pdf_path = "document.pdf"

# Check file exists
if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found")
    exit(1)

# Check file is PDF
if not pdf_path.lower().endswith('.pdf'):
    print("Error: File must be a PDF")
    exit(1)

# Extract with error handling
try:
    text = parser.extract_text(pdf_path)

    if not text or not text.strip():
        print("Warning: No text extracted from PDF")
    else:
        print(f"Successfully extracted {len(text)} characters")

except Exception as e:
    print(f"Extraction failed: {e}")
    exit(1)
```

## Command Line Usage

```bash
# Direct execution
python -m src.extract document.pdf

# With path
python src/extract.py /path/to/document.pdf
```

## Performance Considerations

### Large PDFs

```python
# For large PDFs, extract with metadata first
result = parser.extract_with_metadata("large_document.pdf")

print(f"Pages: {result['metadata']['num_pages']}")
print(f"Size: {len(result['text']) / 1024:.2f} KB")

# Then process in chunks
chunker = TextChunker(chunk_size=1000)  # Larger chunks for large docs
chunks = chunker.chunk_text(result['text'])
```

### Batch Processing

```python
import os
from pathlib import Path

parser = PDFParser()
pdf_dir = Path("data/pdfs")

for pdf_file in pdf_dir.glob("*.pdf"):
    try:
        print(f"Processing {pdf_file.name}...")
        result = parser.extract_with_metadata(str(pdf_file))

        # Save extracted text
        output_file = pdf_dir / f"{pdf_file.stem}.txt"
        output_file.write_text(result["text"])

        print(f"  ✓ Extracted {len(result['text'])} characters")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

## Supported PDF Features

- ✅ Text extraction
- ✅ Multi-page documents
- ✅ Formatted text (markdown export)
- ✅ Basic metadata extraction
- ✅ Unicode characters
- ⚠️ Images (extracted as references, not content)
- ⚠️ Tables (preserved in markdown format)
- ❌ Encrypted PDFs (requires decryption first)

## Dependencies

- `docling>=1.0.0`: Core PDF processing library

## See Also

- [TextChunker API](chunk.md) - Split text into chunks
- [EmbeddingModel API](embed.md) - Generate embeddings
- [Getting Started](../getting-started/quickstart.md) - Quick start guide
