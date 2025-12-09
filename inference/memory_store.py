"""
External Memory Store for Dynamic Context Focusing

This module implements the memory storage system that maintains all historical
interaction chunks with their multi-level representations.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class MemoryChunk:
    """
    A single memory chunk representing one step of agent interaction.
    
    Each chunk contains:
    - Basic metadata (id, type, timestamp)
    - Multi-level representations (full, summary_detailed, summary_brief, keywords)
    - Embedding vector for attention scoring
    - Dynamic relevance score for next-step prediction
    """
    id: int
    type: str  # "user_query", "assistant_response", "tool_call", "tool_observation"
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Multi-level representations
    representations: Dict[str, Any] = field(default_factory=lambda: {
        "full": "",
        "summary_detailed": "",
        "summary_brief": "",
        "keywords": []
    })
    
    # Embedding vector for vector-based attention (computed lazily)
    embedding: Optional[np.ndarray] = None
    
    # Dynamic relevance score (updated each step)
    next_step_relevance: float = 1.0
    
    # Normalized weight after softmax (for context building)
    attention_weight: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryChunk':
        """Create from dictionary."""
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)
    
    def get_representation(self, level: str) -> str:
        """
        Get representation at specified level.
        
        Args:
            level: One of 'full', 'summary_detailed', 'summary_brief', 'keywords', 'placeholder'
        
        Returns:
            The representation string at the specified level.
        """
        if level == 'placeholder':
            return f"[Step {self.id}: {self.type}]"
        elif level == 'keywords':
            keywords = self.representations.get('keywords', [])
            return f"[Step {self.id}: {self.type}] Keywords: {', '.join(keywords)}"
        else:
            return self.representations.get(level, self.representations.get('full', ''))
    
    def get_token_estimate(self, level: str) -> int:
        """
        Estimate token count for a given representation level.
        Rough estimate: 1 token ≈ 4 characters.
        """
        content = self.get_representation(level)
        return max(1, len(content) // 4)


class ExternalMemoryStore:
    """
    External memory store that maintains all historical chunks.
    
    Provides:
    - Storage and retrieval of memory chunks
    - Batch operations for attention scoring
    - Glimpse functionality for targeted retrieval
    """
    
    def __init__(self):
        self.chunks: Dict[int, MemoryChunk] = {}
        self.next_id: int = 0
        self.creation_order: List[int] = []  # Track insertion order
    
    def add_chunk(self, 
                  chunk_type: str,
                  full_content: str,
                  summary_detailed: str = "",
                  summary_brief: str = "",
                  keywords: List[str] = None,
                  metadata: Dict[str, Any] = None,
                  embedding: Optional[np.ndarray] = None) -> MemoryChunk:
        """
        Add a new memory chunk to the store.
        
        Args:
            chunk_type: Type of the chunk (e.g., "tool_observation", "assistant_response")
            full_content: The complete content of this interaction step
            summary_detailed: Detailed summary (if pre-computed)
            summary_brief: Brief summary (if pre-computed)
            keywords: List of keywords (if pre-computed)
            metadata: Additional metadata
            embedding: Pre-computed embedding vector
        
        Returns:
            The created MemoryChunk
        """
        chunk = MemoryChunk(
            id=self.next_id,
            type=chunk_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            representations={
                "full": full_content,
                "summary_detailed": summary_detailed or full_content,
                "summary_brief": summary_brief or "",
                "keywords": keywords or []
            },
            embedding=embedding,
            next_step_relevance=1.0  # New chunks start with max relevance
        )
        
        self.chunks[self.next_id] = chunk
        self.creation_order.append(self.next_id)
        self.next_id += 1
        
        return chunk
    
    def get_chunk(self, chunk_id: int) -> Optional[MemoryChunk]:
        """Get a specific chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_all_chunks(self) -> List[MemoryChunk]:
        """Get all chunks in creation order."""
        return [self.chunks[cid] for cid in self.creation_order if cid in self.chunks]
    
    def get_chunks_except_recent(self, num_recent: int = 2) -> List[MemoryChunk]:
        """Get all chunks except the most recent N chunks."""
        if len(self.creation_order) <= num_recent:
            return []
        older_ids = self.creation_order[:-num_recent]
        return [self.chunks[cid] for cid in older_ids if cid in self.chunks]
    
    def get_recent_chunks(self, num_recent: int = 2) -> List[MemoryChunk]:
        """Get the most recent N chunks."""
        recent_ids = self.creation_order[-num_recent:] if self.creation_order else []
        return [self.chunks[cid] for cid in recent_ids if cid in self.chunks]
    
    def update_relevance_scores(self, scores: Dict[int, float]):
        """
        Update next_step_relevance scores for chunks.
        
        Args:
            scores: Dictionary mapping chunk_id to relevance score
        """
        for chunk_id, score in scores.items():
            if chunk_id in self.chunks:
                self.chunks[chunk_id].next_step_relevance = score
    
    def update_attention_weights(self, weights: Dict[int, float]):
        """
        Update attention weights (after softmax normalization).
        
        Args:
            weights: Dictionary mapping chunk_id to attention weight
        """
        for chunk_id, weight in weights.items():
            if chunk_id in self.chunks:
                self.chunks[chunk_id].attention_weight = weight
    
    def glimpse(self, chunk_id: int = None, query: str = None) -> Optional[str]:
        """
        Glimpse mechanism: retrieve full content of a specific chunk.
        
        Args:
            chunk_id: Direct ID of the chunk to retrieve
            query: (Optional) Query string to find relevant chunk
        
        Returns:
            Full content of the requested chunk, or None if not found
        """
        if chunk_id is not None:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                return chunk.representations.get('full', '')
        
        # If query is provided, could implement semantic search here
        # For now, just return None if no direct ID match
        return None
    
    def search_by_keywords(self, query_keywords: List[str]) -> List[MemoryChunk]:
        """
        Search chunks by keyword overlap.
        
        Args:
            query_keywords: List of keywords to search for
        
        Returns:
            List of chunks that have keyword overlap, sorted by overlap count
        """
        query_set = set(kw.lower() for kw in query_keywords)
        results = []
        
        for chunk in self.get_all_chunks():
            chunk_keywords = set(kw.lower() for kw in chunk.representations.get('keywords', []))
            overlap = len(query_set & chunk_keywords)
            if overlap > 0:
                results.append((chunk, overlap))
        
        # Sort by overlap count descending
        results.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in results]
    
    def get_embeddings_matrix(self) -> tuple:
        """
        Get embeddings matrix for all chunks that have embeddings.
        
        Returns:
            Tuple of (chunk_ids, embeddings_matrix)
        """
        chunk_ids = []
        embeddings = []
        
        for chunk_id in self.creation_order:
            chunk = self.chunks.get(chunk_id)
            if chunk and chunk.embedding is not None:
                chunk_ids.append(chunk_id)
                embeddings.append(chunk.embedding)
        
        if embeddings:
            return chunk_ids, np.stack(embeddings)
        return [], np.array([])
    
    def size(self) -> int:
        """Return the number of chunks in the store."""
        return len(self.chunks)
    
    def clear(self):
        """Clear all chunks from the store."""
        self.chunks.clear()
        self.creation_order.clear()
        self.next_id = 0
    
    def to_json(self) -> str:
        """Serialize the memory store to JSON."""
        data = {
            'next_id': self.next_id,
            'creation_order': self.creation_order,
            'chunks': {str(k): v.to_dict() for k, v in self.chunks.items()}
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExternalMemoryStore':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        store = cls()
        store.next_id = data['next_id']
        store.creation_order = data['creation_order']
        store.chunks = {int(k): MemoryChunk.from_dict(v) for k, v in data['chunks'].items()}
        return store
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __repr__(self) -> str:
        return f"ExternalMemoryStore(chunks={len(self.chunks)}, next_id={self.next_id})"

