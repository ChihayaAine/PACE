"""
Attention Scorer for Dynamic Context Focusing

This module implements the attention scoring mechanism that predicts
the relevance of each historical chunk for generating the next action.

Uses OpenRouter API to call OpenAI's text-embedding-3-small model:
1. Encode current state as query vector Q_t
2. Use pre-computed key vectors K_i for each historical chunk (Compute-on-Write)
3. Compute relevance scores via cosine similarity
4. Apply softmax for normalization
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from memory_store import MemoryChunk, ExternalMemoryStore


class AttentionScorer:
    """
    Attention scorer using OpenRouter API for embeddings.
    
    Computes relevance scores for historical chunks based on
    semantic similarity to the current state (recent chunks + user question).
    """
    
    def __init__(self, 
                 api_key: str = None,
                 api_base: str = None,
                 embedding_model: str = "openai/text-embedding-3-small"):
        """
        Initialize the attention scorer with OpenRouter API.
        
        Args:
            api_key: OpenRouter API key (defaults to env API_KEY)
            api_base: OpenRouter API base URL (defaults to env API_BASE)
            embedding_model: Embedding model to use
        """
        self.api_key = api_key or os.environ.get("API_KEY", "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")
        self.api_base = api_base or os.environ.get("API_BASE", "https://openrouter.ai/api/v1")
        self.embedding_model = embedding_model
        self.embedding_dim = 1536  # text-embedding-3-small outputs 1536 dimensions
        
        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=60.0
        )
        
        print(f"[AttentionScorer] Initialized with model: {self.embedding_model}")
    
    def encode_text(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Encode text into embedding vector using OpenRouter API.
        
        Args:
            text: Text to encode
            max_retries: Number of retry attempts
        
        Returns:
            Embedding vector (numpy array) or None on failure
        """
        # Truncate text if too long (embedding models have token limits)
        if len(text) > 8000:
            text = text[:8000]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                print(f"[AttentionScorer] Embedding attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return None
        return None
    
    def encode_batch(self, texts: List[str], max_retries: int = 3) -> List[Optional[np.ndarray]]:
        """
        Encode multiple texts into embedding vectors.
        
        Note: OpenAI embedding API supports batch input for efficiency.
        
        Args:
            texts: List of texts to encode
            max_retries: Number of retry attempts
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Truncate texts
        truncated = [t[:8000] if len(t) > 8000 else t for t in texts]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=truncated
                )
                embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                return embeddings
            except Exception as e:
                print(f"[AttentionScorer] Batch embedding attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback: try one by one
                    return [self.encode_text(t) for t in texts]
        return [None] * len(texts)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def compute_embeddings_for_chunk(self, chunk: MemoryChunk) -> Optional[np.ndarray]:
        """
        Compute and cache embedding for a chunk using its summary_brief representation.
        This implements the Compute-on-Write strategy.
        
        Args:
            chunk: MemoryChunk to encode
        
        Returns:
            Embedding vector (also cached in chunk.embedding)
        """
        # Use summary_brief for embedding (efficient, captures key info)
        text = chunk.representations.get("summary_brief", "")
        if not text:
            text = chunk.representations.get("summary_detailed", "")
        if not text:
            text = chunk.representations.get("full", "")
        
        if not text:
            return None
        
        embedding = self.encode_text(text)
        if embedding is not None:
            chunk.embedding = embedding
        return embedding
    
    def score_chunks(self,
                     memory_store: ExternalMemoryStore,
                     user_question: str = "",
                     num_recent: int = 2,
                     temperature: float = 1.0) -> Dict[int, float]:
        """
        Compute attention scores for all historical chunks.
        
        Args:
            memory_store: The external memory store containing all chunks
            user_question: The original user question (included in query)
            num_recent: Number of recent chunks to use as part of query (current state)
            temperature: Temperature for softmax (lower = sharper distribution)
        
        Returns:
            Dictionary mapping chunk_id to attention weight (after softmax normalization)
        """
        all_chunks = memory_store.get_all_chunks()
        
        if len(all_chunks) <= num_recent:
            # Not enough history, all chunks get equal weight
            if all_chunks:
                return {chunk.id: 1.0 / len(all_chunks) for chunk in all_chunks}
            return {}
        
        # Split into recent (part of query) and history (to be scored)
        recent_chunks = all_chunks[-num_recent:]
        history_chunks = all_chunks[:-num_recent]
        
        # Build query text from user question + recent chunks
        query_text = self._build_query_text(user_question, recent_chunks)
        
        # Encode query
        query_embedding = self.encode_text(query_text)
        if query_embedding is None:
            print("[AttentionScorer] Failed to encode query, using uniform weights")
            return {chunk.id: 1.0 / len(all_chunks) for chunk in all_chunks}
        
        # Compute/retrieve embeddings for history chunks and calculate similarities
        similarities = []
        chunk_ids = []
        
        # Batch process chunks that don't have embeddings yet
        chunks_needing_embedding = [c for c in history_chunks if c.embedding is None]
        if chunks_needing_embedding:
            texts_to_embed = []
            for chunk in chunks_needing_embedding:
                text = chunk.representations.get("summary_brief", "") or \
                       chunk.representations.get("summary_detailed", "") or \
                       chunk.representations.get("full", "")
                texts_to_embed.append(text if text else f"Step {chunk.id}: {chunk.type}")
            
            # Batch embed
            embeddings = self.encode_batch(texts_to_embed)
            for chunk, emb in zip(chunks_needing_embedding, embeddings):
                if emb is not None:
                    chunk.embedding = emb
        
        # Calculate similarities
        for chunk in history_chunks:
            if chunk.embedding is not None:
                sim = self.cosine_similarity(query_embedding, chunk.embedding)
                similarities.append(max(0, sim))  # Ensure non-negative
                chunk_ids.append(chunk.id)
            else:
                # Fallback for failed embeddings
                similarities.append(0.1)
                chunk_ids.append(chunk.id)
        
        # Apply softmax normalization
        if similarities:
            scores_array = np.array(similarities)
            weights = self._softmax(scores_array, temperature=temperature)
        else:
            weights = np.array([])
        
        # Build result dictionary
        result = {}
        for i, chunk_id in enumerate(chunk_ids):
            result[chunk_id] = float(weights[i])
            # Also update the chunk's relevance score
            if chunk_id in [c.id for c in history_chunks]:
                for c in history_chunks:
                    if c.id == chunk_id:
                        c.next_step_relevance = float(weights[i])
                        break
        
        # Recent chunks always get weight 1.0 (they're included in full anyway)
        for chunk in recent_chunks:
            result[chunk.id] = 1.0
            chunk.next_step_relevance = 1.0
        
        # Log statistics
        if similarities:
            print(f"[AttentionScorer] Scored {len(history_chunks)} history chunks, "
                  f"sim range: [{min(similarities):.3f}, {max(similarities):.3f}], "
                  f"weight range: [{min(weights):.4f}, {max(weights):.4f}]")
        
        return result
    
    def _build_query_text(self, user_question: str, recent_chunks: List[MemoryChunk]) -> str:
        """
        Build query text from user question and recent chunks.
        
        Args:
            user_question: The original user question
            recent_chunks: List of recent MemoryChunks
        
        Returns:
            Combined query text for embedding
        """
        parts = []
        
        # Include user question
        if user_question:
            parts.append(f"Task: {user_question}")
        
        # Include recent chunks (use summary_detailed for more context)
        for chunk in recent_chunks:
            content = chunk.representations.get("summary_detailed", "") or \
                      chunk.representations.get("full", "")
            if content:
                # Truncate each chunk's contribution
                if len(content) > 1500:
                    content = content[:1500]
                parts.append(f"Recent [{chunk.type}]: {content}")
        
        combined = "\n\n".join(parts)
        
        # Ensure total length is reasonable for embedding
        if len(combined) > 6000:
            combined = combined[:6000]
        
        return combined
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Compute softmax with temperature.
        
        Args:
            x: Input array of scores
            temperature: Temperature (lower = sharper distribution)
        
        Returns:
            Softmax probabilities (sum to 1)
        """
        x = x / temperature
        # Subtract max for numerical stability
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def get_top_k_chunks(self,
                         memory_store: ExternalMemoryStore,
                         user_question: str = "",
                         k: int = 5,
                         num_recent: int = 2) -> List[Tuple[int, float]]:
        """
        Get top-k most relevant historical chunks.
        
        Args:
            memory_store: The external memory store
            user_question: The original user question
            k: Number of top chunks to return
            num_recent: Number of recent chunks used as query
        
        Returns:
            List of (chunk_id, attention_weight) tuples, sorted by weight descending
        """
        scores = self.score_chunks(memory_store, user_question, num_recent=num_recent)
        
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
