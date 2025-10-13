"""
Vector Store Module
Manages Chroma vector database for storing and retrieving document chunks.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import uuid


class VectorStore:
    """Wrapper for Chroma vector database."""
    
    def __init__(self, collection_name: str = "pdf_chunks", persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist data (None for in-memory)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Chroma client
        if persist_directory:
            # Persistent storage (for Phase 2)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # In-memory storage (Phase 1)
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document chunks with embeddings"}
        )
    
    def add_chunks(self, chunks: List[str], embeddings: List, metadatas: Optional[List[Dict]] = None):
        """
        Add text chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
        """
        if not chunks:
            return
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Convert embeddings to list format if needed
        embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas if metadatas else [{"chunk_index": i} for i in range(len(chunks))]
        )
    
    def search(self, query_embedding, n_results: int = 5) -> Dict:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Convert embedding to list format if needed
        query_emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        
        results = self.collection.query(
            query_embeddings=[query_emb_list],
            n_results=n_results
        )
        
        return results
    
    def get_count(self) -> int:
        """
        Get the number of chunks in the vector store.
        
        Returns:
            Number of stored chunks
        """
        return self.collection.count()
    
    def clear(self):
        """Clear all data from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document chunks with embeddings"}
        )
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "collection_name": self.collection_name,
            "num_chunks": self.get_count(),
            "persist_directory": self.persist_directory or "in-memory"
        }


if __name__ == "__main__":
    # Simple test
    import numpy as np
    
    # Create vector store
    store = VectorStore()
    
    # Create sample data
    chunks = ["First chunk", "Second chunk", "Third chunk"]
    embeddings = [np.random.rand(384).tolist() for _ in range(3)]  # 384 is dimension for MiniLM
    metadatas = [{"index": i} for i in range(3)]
    
    # Add chunks
    store.add_chunks(chunks, embeddings, metadatas)
    print(f"Added {store.get_count()} chunks")
    
    # Search
    query_emb = np.random.rand(384)
    results = store.search(query_emb, n_results=2)
    print(f"Search results: {results}")
    
    # Stats
    print(f"Stats: {store.get_stats()}")

