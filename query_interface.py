"""
Query Interface Module
Handles user queries and retrieves relevant chunks from the vector store.
"""

from typing import List, Dict, Tuple
from embeddings import EmbeddingModel
from vector_store import VectorStore


class QueryInterface:
    """Interface for querying the vector store and displaying results."""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore, n_results: int = 3):
        """
        Initialize the query interface.
        
        Args:
            embedding_model: Embedding model for query vectorization
            vector_store: Vector store to search
            n_results: Default number of results to return
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.n_results = n_results
    
    def query(self, query_text: str, n_results: Optional[int] = None) -> List[Dict]:
        """
        Query the vector store and retrieve relevant chunks.
        
        Args:
            query_text: User's query
            n_results: Number of results to return (uses default if None)
            
        Returns:
            List of result dictionaries with text, metadata, and distance
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query cannot be empty")
        
        n = n_results if n_results is not None else self.n_results
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query_text)
        
        # Search vector store
        raw_results = self.vector_store.search(query_embedding, n_results=n)
        
        # Format results
        formatted_results = self._format_results(raw_results)
        
        return formatted_results
    
    def _format_results(self, raw_results: Dict) -> List[Dict]:
        """
        Format raw search results into a cleaner structure.
        
        Args:
            raw_results: Raw results from vector store
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        # Chroma returns results as lists within a dict
        documents = raw_results.get('documents', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        for i, doc in enumerate(documents):
            result = {
                'text': doc,
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None,
                'rank': i + 1
            }
            formatted.append(result)
        
        return formatted
    
    def display_results(self, results: List[Dict], query: str):
        """
        Display search results in a user-friendly format.
        
        Args:
            results: List of result dictionaries
            query: Original query text
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Found {len(results)} relevant chunks:")
        print(f"{'='*80}\n")
        
        for result in results:
            rank = result['rank']
            text = result['text']
            metadata = result['metadata']
            distance = result.get('distance', 0)
            
            # Calculate similarity score (lower distance = higher similarity)
            similarity = 1 - distance if distance is not None else 0
            
            print(f"--- Result #{rank} (Similarity: {similarity:.4f}) ---")
            
            # Display metadata if available
            if metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                print(f"Metadata: {metadata_str}")
            
            print(f"\nText:\n{text}")
            print(f"\n{'-'*80}\n")
    
    def interactive_query_loop(self):
        """
        Run an interactive query loop for continuous querying.
        """
        print("\n" + "="*80)
        print("Interactive Query Mode")
        print("="*80)
        print("Enter your questions below. Type 'quit' or 'exit' to end the session.")
        print("="*80 + "\n")
        
        while True:
            try:
                # Get user input
                query = input("> ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting query mode. Goodbye!")
                    break
                
                # Skip empty queries
                if not query:
                    continue
                
                # Process query
                results = self.query(query)
                
                # Display results
                if results:
                    self.display_results(results, query)
                else:
                    print("\nNo results found for your query.\n")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting query mode.")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}\n")


if __name__ == "__main__":
    # Simple test (requires vector store with data)
    print("QueryInterface module - use via main.py for full functionality")

