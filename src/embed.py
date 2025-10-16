"""
Embeddings Module
Uses sentence-transformers to generate vector embeddings for text.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)


if __name__ == "__main__":
    # Simple test
    embedder = EmbeddingModel()
    
    # Test single embedding
    text = "This is a test sentence for embedding."
    embedding = embedder.embed_text(text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test batch embedding
    texts = [
        "Machine learning is fascinating.",
        "Deep learning uses neural networks.",
        "What's the weather like today?"
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = embedder.compute_similarity(texts[0], texts[1])
    print(f"Similarity between text 1 and 2: {sim:.4f}")
    
    sim = embedder.compute_similarity(texts[0], texts[2])
    print(f"Similarity between text 1 and 3: {sim:.4f}")

