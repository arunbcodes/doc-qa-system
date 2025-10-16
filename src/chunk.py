"""
Text Chunker Module
Splits text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Split text into chunks with overlap for better context preservation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the splitter with common separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text)
        return chunks
    
    def chunk_with_metadata(self, text: str, source_metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text to split
            source_metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = self.chunk_text(text)
        
        # Create metadata for each chunk
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": idx,
                "chunk_size": len(chunk),
                **(source_metadata or {})
            }
            chunk_data.append({
                "text": chunk,
                "metadata": metadata
            })
        
        return chunk_data
    
    def get_stats(self, text: str) -> Dict:
        """
        Get statistics about how text will be chunked.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with chunking statistics
        """
        chunks = self.chunk_text(text)
        
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_chunk_size": 0,
                "total_characters": 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        return {
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": len(text)
        }


if __name__ == "__main__":
    # Simple test
    test_text = """
    This is a test document. It contains multiple paragraphs.
    
    This is the second paragraph. It has more text to test the chunking functionality.
    We want to ensure that the text is split properly with overlap.
    
    This is the third paragraph with even more content to demonstrate how the chunker works.
    """
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(test_text)
    stats = chunker.get_stats(test_text)
    
    print(f"Statistics: {stats}")
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk)

