"""
Attention Scorer for Dynamic Context Focusing

This module implements the attention scoring mechanism that predicts
the relevance of each historical chunk for generating the next action.

Supports two embedding backends:
1. Local BGE-M3 model (default, recommended)
2. OpenRouter API (commented out, can be enabled if needed)

Workflow:
1. Encode current state as query vector Q_t
2. Use pre-computed key vectors K_i for each historical chunk (Compute-on-Write)
3. Compute relevance scores via cosine similarity
4. Apply softmax for normalization
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from memory_store import MemoryChunk, ExternalMemoryStore


# ============================================================================
# Configuration
# ============================================================================

# BGE-M3 model path (local deployment)
BGE_M3_MODEL_PATH = os.getenv("BGE_M3_MODEL_PATH", "/root/.cache/modelscope/hub/models/BAAI/bge-m3")

# Whether to use FP16 (faster on GPU, set False for CPU)
BGE_M3_USE_FP16 = os.getenv("BGE_M3_USE_FP16", "true").lower() == "true"


class AttentionScorer:
    """
    Attention scorer using local BGE-M3 model for embeddings.
    
    Computes relevance scores for historical chunks based on
    semantic similarity to the current state (recent chunks + user question).
    """
    
    # ========================================================================
    # OpenRouter API configuration (commented out, kept for reference)
    # ========================================================================
    # OPENROUTER_API_KEY = "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"
    # OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    
    def __init__(self, 
                 model_path: str = None,
                 use_fp16: bool = None):
        """
        Initialize the attention scorer with local BGE-M3 model.
        
        Args:
            model_path: Path to the BGE-M3 model (defaults to BGE_M3_MODEL_PATH)
            use_fp16: Whether to use FP16 precision (defaults to BGE_M3_USE_FP16)
        """
        self.model_path = model_path or BGE_M3_MODEL_PATH
        self.use_fp16 = use_fp16 if use_fp16 is not None else BGE_M3_USE_FP16
        self.model = None
        self.embedding_dim = 1024  # BGE-M3 outputs 1024 dimensions
        
        # Lazy load the model
        self._load_model()
    
    def _load_model(self):
        """Load the BGE-M3 model."""
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"[AttentionScorer] Loading BGE-M3 model from: {self.model_path}")
            print(f"[AttentionScorer] use_fp16={self.use_fp16}")
            
            self.model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)
            self.embedding_dim = 1024  # BGE-M3 dense vector dimension
            
            print(f"[AttentionScorer] BGE-M3 model loaded successfully")
        except ImportError as e:
            print(f"[AttentionScorer] Warning: FlagEmbedding not installed: {e}")
            print(f"[AttentionScorer] Install with: pip install FlagEmbedding")
            self.model = None
        except Exception as e:
            print(f"[AttentionScorer] Error loading BGE-M3 model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text into embedding vector using BGE-M3.
        
        Args:
            text: Text to encode
        
        Returns:
            Embedding vector (numpy array) or None on failure
        """
        if self.model is None:
            print("[AttentionScorer] Model not loaded, cannot encode")
            return None
        
        # Truncate text if too long (BGE-M3 max_length is 8192)
        if len(text) > 20000:  # Rough char limit
            text = text[:20000]
        
        try:
            # BGE-M3 returns dict with 'dense_vecs' key
            result = self.model.encode([text], batch_size=1, max_length=8192)
            embedding = result['dense_vecs'][0]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"[AttentionScorer] Encoding failed: {type(e).__name__}: {e}")
            return None
    
    def encode_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            print("[AttentionScorer] Model not loaded, cannot encode")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        # Truncate texts
        truncated = [t[:20000] if len(t) > 20000 else t for t in texts]
        
        try:
            result = self.model.encode(truncated, batch_size=12, max_length=8192)
            embeddings = [np.array(vec, dtype=np.float32) for vec in result['dense_vecs']]
            print(f"[AttentionScorer] Batch encoded {len(texts)} texts")
            return embeddings
        except Exception as e:
            print(f"[AttentionScorer] Batch encoding failed: {type(e).__name__}: {e}")
            # Fallback: try one by one
            return [self.encode_text(t) for t in texts]
    
    # ========================================================================
    # OpenRouter API methods (commented out, kept for reference)
    # ========================================================================
    # def encode_text_api(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
    #     """
    #     Encode text into embedding vector using OpenRouter API.
    #     """
    #     from openai import OpenAI
    #     
    #     client = OpenAI(
    #         api_key=self.OPENROUTER_API_KEY,
    #         base_url=self.OPENROUTER_API_BASE,
    #         timeout=60.0
    #     )
    #     
    #     if len(text) > 8000:
    #         text = text[:8000]
    #     
    #     for attempt in range(max_retries):
    #         try:
    #             response = client.embeddings.create(
    #                 model="openai/text-embedding-3-small",
    #                 input=text
    #             )
    #             embedding = response.data[0].embedding
    #             return np.array(embedding, dtype=np.float32)
    #         except Exception as e:
    #             print(f"[AttentionScorer] API attempt {attempt + 1}/{max_retries} failed: {e}")
    #             if attempt == max_retries - 1:
    #                 return None
    #     return None
    
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
                     temperature: float = 0.3) -> Dict[int, float]:
        """
        Compute attention scores for all historical chunks.
        
        Args:
            memory_store: The external memory store containing all chunks
            user_question: The original user question (included in query)
            num_recent: Number of recent chunks to use as part of query (current state)
            temperature: Temperature for softmax (lower = sharper distribution, default 0.3)
        
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
        if len(combined) > 15000:
            combined = combined[:15000]
        
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
