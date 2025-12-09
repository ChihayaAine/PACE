"""
Attention Scorer for Dynamic Context Focusing

This module implements the attention scoring mechanism that predicts
the relevance of each historical chunk for generating the next action.

Uses a vectorized approach (Method B):
1. Encode current state as query vector Q_t
2. Use pre-computed key vectors K_i for each historical chunk
3. Compute relevance scores via dot product similarity
4. Apply softmax for normalization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from memory_store import MemoryChunk, ExternalMemoryStore


class AttentionScorer:
    """
    Vectorized attention scorer using sentence embeddings.
    
    Computes relevance scores for historical chunks based on
    semantic similarity to the current state (recent chunks).
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_gpu: bool = False):
        """
        Initialize the attention scorer.
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
            use_gpu: Whether to use GPU for encoding
        """
        self.embedding_model_name = embedding_model
        self.use_gpu = use_gpu
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        # Lazy load the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.embedding_model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[AttentionScorer] Loaded model: {self.embedding_model_name}, dim={self.embedding_dim}")
        except ImportError:
            print("[AttentionScorer] Warning: sentence-transformers not installed. Using fallback.")
            self.model = None
        except Exception as e:
            print(f"[AttentionScorer] Error loading model: {e}. Using fallback.")
            self.model = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector.
        
        Args:
            text: Text to encode
        
        Returns:
            Embedding vector
        """
        if self.model is not None:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple TF-IDF-like encoding (very basic)
            return self._fallback_encode(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            Matrix of embedding vectors (N x dim)
        """
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            return np.stack([self._fallback_encode(t) for t in texts])
    
    def _fallback_encode(self, text: str) -> np.ndarray:
        """
        Fallback encoding using simple hash-based projection.
        Not semantically meaningful but provides consistent dimensions.
        """
        # Simple hash-based encoding
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def compute_embeddings_for_chunk(self, chunk: MemoryChunk) -> np.ndarray:
        """
        Compute embedding for a chunk using its summary_detailed representation.
        
        Args:
            chunk: MemoryChunk to encode
        
        Returns:
            Embedding vector
        """
        # Use summary_detailed for encoding (balance between detail and efficiency)
        text = chunk.representations.get("summary_detailed", "")
        if not text:
            text = chunk.representations.get("full", "")
        
        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000]
        
        return self.encode_text(text)
    
    def score_chunks(self,
                     memory_store: ExternalMemoryStore,
                     num_recent: int = 2,
                     temperature: float = 1.0) -> Dict[int, float]:
        """
        Compute attention scores for all historical chunks.
        
        Args:
            memory_store: The external memory store containing all chunks
            num_recent: Number of recent chunks to use as query (current state)
            temperature: Temperature for softmax (lower = sharper distribution)
        
        Returns:
            Dictionary mapping chunk_id to attention weight (after softmax)
        """
        all_chunks = memory_store.get_all_chunks()
        
        if len(all_chunks) <= num_recent:
            # Not enough history, all chunks get equal weight
            return {chunk.id: 1.0 / len(all_chunks) for chunk in all_chunks}
        
        # Split into recent (query) and history (keys)
        recent_chunks = all_chunks[-num_recent:]
        history_chunks = all_chunks[:-num_recent]
        
        # Build query vector from recent chunks
        query_vector = self._build_query_vector(recent_chunks)
        
        # Compute key vectors for history chunks (use cached if available)
        key_vectors, chunk_ids = self._get_key_vectors(history_chunks, memory_store)
        
        if len(key_vectors) == 0:
            # No valid history
            result = {chunk.id: 1.0 / len(all_chunks) for chunk in all_chunks}
            return result
        
        # Compute similarity scores (dot product)
        # query_vector: (dim,), key_vectors: (N, dim)
        similarities = np.dot(key_vectors, query_vector)
        
        # Apply softmax with temperature
        scores = self._softmax(similarities, temperature=temperature)
        
        # Build result dictionary
        result = {}
        for i, chunk_id in enumerate(chunk_ids):
            result[chunk_id] = float(scores[i])
        
        # Recent chunks always get high weight (not in softmax)
        for chunk in recent_chunks:
            result[chunk.id] = 1.0  # Max weight for recent chunks
        
        return result
    
    def _build_query_vector(self, recent_chunks: List[MemoryChunk]) -> np.ndarray:
        """
        Build query vector from recent chunks.
        
        Args:
            recent_chunks: List of recent MemoryChunks
        
        Returns:
            Query embedding vector
        """
        # Concatenate recent chunk contents
        texts = []
        for chunk in recent_chunks:
            text = chunk.representations.get("full", "")
            if len(text) > 1500:
                text = text[:1500]
            texts.append(text)
        
        combined_text = "\n\n".join(texts)
        
        # Encode combined text
        return self.encode_text(combined_text)
    
    def _get_key_vectors(self, 
                         chunks: List[MemoryChunk],
                         memory_store: ExternalMemoryStore) -> Tuple[np.ndarray, List[int]]:
        """
        Get key vectors for chunks, using cached embeddings when available.
        
        Args:
            chunks: List of chunks to get vectors for
            memory_store: Memory store (for updating cached embeddings)
        
        Returns:
            Tuple of (key_vectors matrix, chunk_ids list)
        """
        key_vectors = []
        chunk_ids = []
        texts_to_encode = []
        chunks_to_update = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                # Use cached embedding
                key_vectors.append(chunk.embedding)
                chunk_ids.append(chunk.id)
            else:
                # Need to compute embedding
                text = chunk.representations.get("summary_detailed", "")
                if not text:
                    text = chunk.representations.get("full", "")
                if len(text) > 2000:
                    text = text[:2000]
                texts_to_encode.append(text)
                chunks_to_update.append(chunk)
        
        # Batch encode texts that don't have embeddings
        if texts_to_encode:
            new_embeddings = self.encode_batch(texts_to_encode)
            for i, chunk in enumerate(chunks_to_update):
                chunk.embedding = new_embeddings[i]
                key_vectors.append(new_embeddings[i])
                chunk_ids.append(chunk.id)
        
        if key_vectors:
            return np.stack(key_vectors), chunk_ids
        return np.array([]), []
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Compute softmax with temperature.
        
        Args:
            x: Input array
            temperature: Temperature (lower = sharper distribution)
        
        Returns:
            Softmax probabilities
        """
        x = x / temperature
        # Subtract max for numerical stability
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def get_top_k_chunks(self,
                         memory_store: ExternalMemoryStore,
                         k: int = 5,
                         num_recent: int = 2) -> List[Tuple[int, float]]:
        """
        Get top-k most relevant historical chunks.
        
        Args:
            memory_store: The external memory store
            k: Number of top chunks to return
            num_recent: Number of recent chunks used as query
        
        Returns:
            List of (chunk_id, attention_weight) tuples, sorted by weight
        """
        scores = self.score_chunks(memory_store, num_recent=num_recent)
        
        # Exclude recent chunks from ranking
        recent_ids = set(c.id for c in memory_store.get_recent_chunks(num_recent))
        history_scores = [(cid, score) for cid, score in scores.items() 
                          if cid not in recent_ids]
        
        # Sort by score descending
        history_scores.sort(key=lambda x: x[1], reverse=True)
        
        return history_scores[:k]


# Singleton instance
_default_scorer = None

def get_attention_scorer() -> AttentionScorer:
    """Get or create the default attention scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = AttentionScorer()
    return _default_scorer

