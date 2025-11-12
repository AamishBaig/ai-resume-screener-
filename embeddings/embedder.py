"""
Text embedding module using sentence-transformers.
Provides semantic vector representations for similarity matching.
"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not available")


class Embedder:
    """Compute semantic embeddings for text using transformer models."""
    
    def __init__(self, config):
        """
        Initialize embedder with model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.model_name = config.embedding_model
        
        self._load_model()
    
    def _load_model(self, retry_count: int = 0) -> None:
        """
        Load the sentence transformer model.
        
        Args:
            retry_count: Current retry attempt
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers library not available")
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            
            if retry_count < 2:
                logger.info(f"Retrying model load (attempt {retry_count + 1})")
                import time
                time.sleep(1)
                self._load_model(retry_count + 1)
            else:
                raise RuntimeError(f"Failed to load embedding model after retries: {str(e)}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for encoding")
            # Return zero vector
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Encoding error: {str(e)}")
            raise
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[np.ndarray]:
        """
        Encode multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Batch encoding error: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.get_sentence_embedding_dimension()
