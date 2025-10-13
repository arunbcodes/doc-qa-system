"""
PDF Parser Module
Uses docling to extract clean, structured text from PDF files.
"""

from docling.document_converter import DocumentConverter


class PDFParser:
    """Extract text from PDF files using docling."""
    
    def __init__(self):
        """Initialize the document converter."""
        self.converter = DocumentConverter()
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If extraction fails
        """
        try:
            # Convert PDF to document
            result = self.converter.convert(pdf_path)
            
            # Export as markdown (cleaner structure)
            text = result.document.export_to_markdown()
            
            return text
            
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_with_metadata(self, pdf_path: str) -> dict:
        """
        Extract text along with metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text and metadata
        """
        try:
            result = self.converter.convert(pdf_path)
            
            # Extract text
            text = result.document.export_to_markdown()
            
            # Extract basic metadata if available
            metadata = {
                "source": pdf_path,
                "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else None,
            }
            
            return {
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            raise Exception(f"Failed to extract text with metadata: {str(e)}")


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) > 1:
        parser = PDFParser()
        text = parser.extract_text(sys.argv[1])
        print(f"Extracted {len(text)} characters")
        print("\nFirst 500 characters:")
        print(text[:500])
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")

