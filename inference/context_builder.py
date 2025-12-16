"""
Context Builder for Dynamic Context Focusing

This module implements the weighted context reconstruction logic:
- Uses attention weights to select representation levels for each chunk
- Builds a focused context within token budget
- Prioritizes recent chunks with full content
- Logs which representation level is used for each chunk for debugging
"""

import tiktoken
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from memory_store import MemoryChunk, ExternalMemoryStore


class ContextBuilder:
    """
    Builds dynamic context based on attention weights.
    
    Selects representation levels (full/detailed/brief/placeholder) for each
    historical chunk based on its attention weight, ensuring the final context
    stays within the token budget.
    
    NEW: Adaptive Compression - thresholds are dynamically adjusted based on:
    - Round progress (higher rounds → more compression)
    - Token usage (higher usage → more compression)
    """
    
    # Base thresholds for representation level selection (at low pressure)
    # These are applied AFTER softmax normalization
    # We use relative thresholds based on uniform distribution (1/N)
    BASE_THRESHOLDS = {
        "a": 0.4,   # Below 0.4x uniform: placeholder
        "b": 0.8,   # 0.4-0.8x uniform: summary_brief
        "c": 1.5,   # 0.8-1.5x uniform: summary_detailed
        # w > 1.5x uniform: full
    }
    
    # High pressure thresholds (at maximum pressure)
    # Much stricter - only very relevant chunks get detailed/full
    HIGH_PRESSURE_THRESHOLDS = {
        "a": 0.8,   # Below 0.8x uniform: placeholder (more keywords)
        "b": 1.2,   # 0.8-1.2x uniform: summary_brief (more brief)
        "c": 2.5,   # 1.2-2.5x uniform: summary_detailed
        # w > 2.5x uniform: full (only the most relevant)
    }
    
    # Pressure calculation parameters
    MAX_ROUNDS_DEFAULT = 100
    MAX_TOKENS_DEFAULT = 110 * 1024  # 110K
    PRESSURE_ONSET_RATIO = 0.5  # Start increasing pressure at 50% capacity
    
    def __init__(self,
                 max_context_tokens: int = 100000,
                 num_recent_full: int = 2,
                 thresholds: Dict[str, float] = None):
        """
        Initialize the context builder.
        
        Args:
            max_context_tokens: Maximum tokens for the dynamic context
            num_recent_full: Number of recent chunks to always include in full
            thresholds: Dictionary with keys 'a', 'b', 'c' for representation selection
        """
        self.max_context_tokens = max_context_tokens
        self.num_recent_full = num_recent_full
        self.base_thresholds = thresholds or self.BASE_THRESHOLDS
        
        # For backward compatibility
        self.thresholds = self.base_thresholds
        
        # Adaptive compression state
        self.current_pressure = 0.0
        self.max_rounds = int(os.getenv('MAX_LLM_CALL_PER_RUN', self.MAX_ROUNDS_DEFAULT))
        
        # Token encoder
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def _calculate_pressure(self, round_num: int, current_tokens: int) -> float:
        """
        Calculate compression pressure based on round progress and token usage.
        
        Pressure is a value between 0 and 1:
        - 0: Early stage, no pressure, use base thresholds
        - 1: Maximum pressure, use strict thresholds
        
        Args:
            round_num: Current round number (1-indexed)
            current_tokens: Current token count in context
        
        Returns:
            Pressure value between 0 and 1
        """
        # Round pressure: starts increasing after PRESSURE_ONSET_RATIO of max rounds
        round_onset = self.max_rounds * self.PRESSURE_ONSET_RATIO
        if round_num <= round_onset:
            round_pressure = 0.0
        else:
            # Linear increase from 0 to 1 between onset and max
            round_pressure = (round_num - round_onset) / (self.max_rounds - round_onset)
            round_pressure = min(1.0, round_pressure)
        
        # Token pressure: starts increasing after PRESSURE_ONSET_RATIO of max tokens
        token_onset = self.MAX_TOKENS_DEFAULT * self.PRESSURE_ONSET_RATIO
        if current_tokens <= token_onset:
            token_pressure = 0.0
        else:
            # Linear increase from 0 to 1 between onset and max
            token_pressure = (current_tokens - token_onset) / (self.MAX_TOKENS_DEFAULT - token_onset)
            token_pressure = min(1.0, token_pressure)
        
        # Use the maximum of both pressures
        pressure = max(round_pressure, token_pressure)
        
        return pressure
    
    def _get_adaptive_thresholds(self, round_num: int, current_tokens: int) -> Dict[str, float]:
        """
        Get dynamically adjusted thresholds based on current pressure.
        
        Uses linear interpolation between base thresholds and high-pressure thresholds.
        
        Args:
            round_num: Current round number
            current_tokens: Current token count
        
        Returns:
            Dictionary with adjusted thresholds 'a', 'b', 'c'
        """
        pressure = self._calculate_pressure(round_num, current_tokens)
        self.current_pressure = pressure  # Store for logging
        
        # Linear interpolation: base + pressure * (high - base)
        adaptive_thresholds = {}
        for key in ['a', 'b', 'c']:
            base_val = self.base_thresholds.get(key, self.BASE_THRESHOLDS[key])
            high_val = self.HIGH_PRESSURE_THRESHOLDS[key]
            adaptive_thresholds[key] = base_val + pressure * (high_val - base_val)
        
        return adaptive_thresholds
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4
    
    def select_representation_level(self, 
                                      attention_weight: float,
                                      num_history_chunks: int,
                                      round_num: int = 1,
                                      current_tokens: int = 0) -> str:
        """
        Select representation level based on attention weight.
        
        The weight is compared against thresholds that are relative to
        the uniform distribution (1/N where N is number of history chunks).
        
        NEW: Thresholds are adaptively adjusted based on round progress
        and token usage (pressure-based compression).
        
        Args:
            attention_weight: The softmax-normalized attention weight
            num_history_chunks: Total number of history chunks being scored
            round_num: Current round number (for adaptive thresholds)
            current_tokens: Current token count (for adaptive thresholds)
        
        Returns:
            Representation level: 'full', 'summary_detailed', 'summary_brief', 'placeholder', or 'omit'
        """
        if num_history_chunks == 0:
            return 'full'
        
        # Uniform weight if all chunks were equally relevant
        uniform_weight = 1.0 / num_history_chunks
        
        # Relative weight compared to uniform
        relative_weight = attention_weight / (uniform_weight + 1e-8)
        
        # Get adaptive thresholds based on current pressure
        thresholds = self._get_adaptive_thresholds(round_num, current_tokens)
        
        a = thresholds.get("a", 0.5)
        b = thresholds.get("b", 1.0)
        c = thresholds.get("c", 2.0)
        
        if relative_weight > c:
            return 'full'
        elif relative_weight > b:
            return 'summary_detailed'
        elif relative_weight > a:
            return 'summary_brief'
        else:
            return 'placeholder'  # or 'omit' for very low weights
    
    def build_context(self,
                      memory_store: ExternalMemoryStore,
                      attention_weights: Dict[int, float],
                      system_prompt: str = "",
                      user_question: str = "") -> Tuple[List[Dict], Dict[str, any]]:
        """
        Build the dynamic context based on attention weights.
        
        Args:
            memory_store: The external memory store
            attention_weights: Dictionary mapping chunk_id to attention weight
            system_prompt: The system prompt to include
            user_question: The original user question
        
        Returns:
            Tuple of:
                - List of messages in OpenAI format
                - Metadata dict with stats about the context
        """
        all_chunks = memory_store.get_all_chunks()
        if not all_chunks:
            return self._build_empty_context(system_prompt, user_question)
        
        # Separate recent chunks (always full) from history
        recent_chunks = memory_store.get_recent_chunks(self.num_recent_full)
        recent_ids = set(c.id for c in recent_chunks)
        history_chunks = [c for c in all_chunks if c.id not in recent_ids]
        
        num_history = len(history_chunks)
        
        # Calculate token budget
        base_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_question)
        available_tokens = self.max_context_tokens - base_tokens - 5000  # Reserve for response
        
        # First, allocate tokens for recent chunks (always full)
        recent_content_parts = []
        recent_tokens = 0
        for chunk in recent_chunks:
            content = chunk.get_representation('full')
            recent_content_parts.append(self._format_chunk_content(chunk, content, 'full'))
            recent_tokens += self.count_tokens(content)
        
        # Remaining budget for history
        history_budget = available_tokens - recent_tokens
        
        # Build history content with adaptive representation levels
        history_content_parts = []
        history_tokens = 0
        level_counts = {'full': 0, 'summary_detailed': 0, 'summary_brief': 0, 'placeholder': 0, 'omit': 0}
        
        # Sort history by chunk id (chronological order)
        history_chunks_sorted = sorted(history_chunks, key=lambda c: c.id)
        
        for chunk in history_chunks_sorted:
            weight = attention_weights.get(chunk.id, 0.0)
            level = self.select_representation_level(weight, num_history)
            
            # Check if we have budget
            content = chunk.get_representation(level)
            content_tokens = self.count_tokens(content)
            
            # If over budget, try lower levels
            while content_tokens > 0 and history_tokens + content_tokens > history_budget:
                if level == 'full':
                    level = 'summary_detailed'
                elif level == 'summary_detailed':
                    level = 'summary_brief'
                elif level == 'summary_brief':
                    level = 'placeholder'
                else:
                    level = 'omit'
                    break
                content = chunk.get_representation(level)
                content_tokens = self.count_tokens(content)
            
            level_counts[level] += 1
            
            if level != 'omit':
                formatted = self._format_chunk_content(chunk, content, level)
                history_content_parts.append(formatted)
                history_tokens += content_tokens
        
        # Combine all parts
        messages = self._assemble_messages(
            system_prompt=system_prompt,
            user_question=user_question,
            history_parts=history_content_parts,
            recent_parts=recent_content_parts
        )
        
        metadata = {
            "total_chunks": len(all_chunks),
            "history_chunks": num_history,
            "recent_chunks": len(recent_chunks),
            "level_counts": level_counts,
            "history_tokens": history_tokens,
            "recent_tokens": recent_tokens,
            "total_context_tokens": base_tokens + history_tokens + recent_tokens
        }
        
        return messages, metadata
    
    def _format_chunk_content(self, 
                               chunk: MemoryChunk, 
                               content: str, 
                               level: str) -> str:
        """
        Format a chunk's content for inclusion in context.
        
        Args:
            chunk: The memory chunk
            content: The representation content
            level: The representation level used
        
        Returns:
            Formatted string
        """
        if level == 'placeholder':
            return f"[Step {chunk.id}: {chunk.type} - details available via glimpse]"
        elif level == 'full':
            return f"=== Step {chunk.id} ({chunk.type}) ===\n{content}"
        else:
            return f"=== Step {chunk.id} ({chunk.type}) [Summary] ===\n{content}"
    
    def _assemble_messages(self,
                           system_prompt: str,
                           user_question: str,
                           history_parts: List[str],
                           recent_parts: List[str]) -> List[Dict]:
        """
        Assemble the final message list.
        
        Structure:
        1. System message (with tools description)
        2. User message (original question)
        3. Assistant+User alternating for history (compressed as single context block)
        4. Recent full interactions
        """
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # User question
        messages.append({"role": "user", "content": user_question})
        
        # If we have history, add it as context
        if history_parts:
            history_context = (
                "=== Previous Interaction History (summarized based on relevance) ===\n\n" +
                "\n\n".join(history_parts) +
                "\n\n=== End of History ===\n\n" +
                "Note: Use `glimpse` tool to retrieve full details of any historical step if needed."
            )
            messages.append({"role": "assistant", "content": history_context})
            messages.append({"role": "user", "content": "Continue with the task. The above is a summary of previous steps."})
        
        # Recent full interactions (as separate messages to preserve turn structure)
        for part in recent_parts:
            # Parse the part to determine if it's assistant or user content
            # For simplicity, add as context continuation
            if "tool_call" in part.lower() or "assistant" in part.lower():
                messages.append({"role": "assistant", "content": part})
            else:
                messages.append({"role": "user", "content": part})
        
        return messages
    
    def _build_empty_context(self, 
                              system_prompt: str, 
                              user_question: str) -> Tuple[List[Dict], Dict]:
        """Build context when memory store is empty."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_question})
        
        metadata = {
            "total_chunks": 0,
            "history_chunks": 0,
            "recent_chunks": 0,
            "level_counts": {},
            "total_context_tokens": self.count_tokens(system_prompt) + self.count_tokens(user_question)
        }
        
        return messages, metadata
    
    def build_context_simple(self,
                              memory_store: ExternalMemoryStore,
                              attention_weights: Dict[int, float],
                              system_prompt: str = "",
                              user_question: str = "",
                              round_num: int = 1) -> Tuple[List[Dict], Dict[int, str]]:
        """
        Simplified context building that returns messages compatible with
        the existing react_agent format.
        
        NEW: Supports adaptive compression based on round progress and token usage.
        
        Args:
            memory_store: External memory store with historical chunks
            attention_weights: Dictionary mapping chunk_id to attention weight
            system_prompt: System prompt to include
            user_question: Original user question
            round_num: Current round number (for adaptive compression)
        
        Returns:
            Tuple of:
                - messages: List of messages in standard OpenAI format
                - chunk_levels: Dict mapping chunk_id to representation level used
        """
        all_chunks = memory_store.get_all_chunks()
        chunk_levels = {}  # Track which level was used for each chunk
        
        # Start with system and user
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_question})
        
        if not all_chunks:
            return messages, chunk_levels
        
        # Separate recent from history
        recent_chunks = memory_store.get_recent_chunks(self.num_recent_full)
        recent_ids = set(c.id for c in recent_chunks)
        history_chunks = [c for c in all_chunks if c.id not in recent_ids]
        num_history = len(history_chunks)
        
        # Token budget tracking
        current_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        available = self.max_context_tokens - current_tokens - 10000  # Reserve for output
        
        # Build history part
        if history_chunks:
            history_text_parts = []
            used_tokens = 0
            
            for chunk in sorted(history_chunks, key=lambda c: c.id):
                weight = attention_weights.get(chunk.id, 0.0)
                # Use adaptive thresholds based on round and token usage
                level = self.select_representation_level(
                    weight, num_history, 
                    round_num=round_num, 
                    current_tokens=current_tokens + used_tokens
                )
                original_level = level  # Track original selection
                
                content = chunk.get_representation(level)
                tokens = self.count_tokens(content)
                
                # Downgrade if over budget
                downgraded = False
                while tokens > 0 and used_tokens + tokens > available * 0.6:
                    if level == 'full':
                        level = 'summary_detailed'
                    elif level == 'summary_detailed':
                        level = 'summary_brief'
                    elif level == 'summary_brief':
                        level = 'placeholder'
                    else:
                        break
                    content = chunk.get_representation(level)
                    tokens = self.count_tokens(content)
                    downgraded = True
                
                # Record the final level used
                chunk_levels[chunk.id] = level
                
                if level != 'omit' and tokens > 0:
                    formatted = self._format_chunk_content(chunk, content, level)
                    history_text_parts.append(formatted)
                    used_tokens += tokens
            
            if history_text_parts:
                history_block = (
                    "[Historical Context - summarized by relevance]\n\n" +
                    "\n\n".join(history_text_parts) +
                    "\n\n[End Historical Context]"
                )
                messages.append({"role": "assistant", "content": history_block})
                messages.append({"role": "user", "content": "Noted. Continue with the investigation."})
        
        # Add recent chunks in full (as proper turn alternation)
        for chunk in sorted(recent_chunks, key=lambda c: c.id):
            content = chunk.get_representation('full')
            chunk_levels[chunk.id] = 'full (recent)'  # Mark as recent
            
            if chunk.type in ["assistant_response", "tool_call"]:
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})
        
        # Print detailed log for debugging (now with raw similarities and adaptive thresholds)
        self._log_context_build(chunk_levels, attention_weights, num_history, memory_store, round_num)
        
        return messages, chunk_levels
    
    def _log_context_build(self, 
                           chunk_levels: Dict[int, str], 
                           attention_weights: Dict[int, float],
                           num_history: int,
                           memory_store: ExternalMemoryStore = None,
                           round_num: int = 0):
        """
        Log detailed information about context building for debugging.
        
        Args:
            chunk_levels: Dictionary mapping chunk_id to representation level
            attention_weights: Dictionary mapping chunk_id to attention weight
            num_history: Number of history chunks
            memory_store: Optional memory store to retrieve raw similarities
            round_num: Current round number
        """
        print("\n" + "=" * 80)
        print("[Context Build Log] Representation levels used for each chunk:")
        
        # Show adaptive compression info
        pressure = self.current_pressure
        pressure_pct = int(pressure * 100)
        thresholds = self._get_adaptive_thresholds(round_num, 0)  # Get current thresholds
        print(f"[Adaptive Compression] Round: {round_num}, Pressure: {pressure_pct}%")
        print(f"[Adaptive Thresholds] a={thresholds['a']:.2f}, b={thresholds['b']:.2f}, c={thresholds['c']:.2f}")
        
        print("-" * 80)
        print(f"{'ID':<6} {'Type':<20} {'Raw Sim':<10} {'Weight':<12} {'Rel':<8} {'Level':<15}")
        print("-" * 80)
        
        # Build chunk lookup for raw similarity
        chunk_lookup = {}
        if memory_store:
            for chunk in memory_store.get_all_chunks():
                chunk_lookup[chunk.id] = chunk
        
        # Sort by chunk ID for readability
        for chunk_id in sorted(chunk_levels.keys()):
            level = chunk_levels[chunk_id]
            weight = attention_weights.get(chunk_id, 0.0)
            
            # Get raw similarity from chunk metadata
            raw_sim = 0.0
            chunk_type = ""
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                raw_sim = chunk.metadata.get("raw_similarity", 0.0)
                chunk_type = chunk.type[:18] if len(chunk.type) > 18 else chunk.type
            
            # Determine uniform weight for comparison
            uniform = 1.0 / num_history if num_history > 0 else 1.0
            relative = weight / uniform if uniform > 0 else 0
            
            # Add visual indicator
            if 'recent' in level:
                indicator = "🔵"  # Recent chunk
            elif level == 'full':
                indicator = "🟢"  # Full content
            elif level == 'summary_detailed':
                indicator = "🟡"  # Detailed summary
            elif level == 'summary_brief':
                indicator = "🟠"  # Brief summary
            else:
                indicator = "⚪"  # Placeholder
            
            print(f"{indicator} {chunk_id:<4} {chunk_type:<20} {raw_sim:<10.4f} {weight:<12.4f} {relative:<8.2f}x {level}")
        
        print("=" * 80 + "\n")


# Singleton instance
_default_builder = None

def get_context_builder(max_tokens: int = 100000) -> ContextBuilder:
    """Get or create the default context builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = ContextBuilder(max_context_tokens=max_tokens)
    return _default_builder

