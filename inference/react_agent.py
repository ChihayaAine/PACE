"""
Multi-Turn React Agent with Dynamic Context Focusing

This module implements the main agent with:
- External memory store for all historical interactions
- Multi-level representations for each memory chunk
- Vectorized attention scoring for relevance prediction
- Dynamic context building based on attention weights
- Glimpse tool for retrieving compressed information
"""

import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import random
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import threading

# Tool imports
from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

# Dynamic Context Focusing imports
from memory_store import MemoryChunk, ExternalMemoryStore
from representation_generator import RepresentationGenerator, get_representation_generator
from attention_scorer import AttentionScorer, get_attention_scorer
from context_builder import ContextBuilder, get_context_builder
# from tool_glimpse import Glimpse, set_glimpse_memory_store  # TODO: 暂时禁用 glimpse

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

# Dynamic context configuration
ENABLE_DYNAMIC_CONTEXT = os.getenv('ENABLE_DYNAMIC_CONTEXT', 'true').lower() == 'true'
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', 100000))
NUM_RECENT_FULL = int(os.getenv('NUM_RECENT_FULL', 2))

# Memory logs directory
MEMORY_LOGS_DIR = os.path.join(os.path.dirname(__file__), "memory_logs")
os.makedirs(MEMORY_LOGS_DIR, exist_ok=True)

# Dynamic context logs directory (for analyzing repeated tool_call issues)
CONTEXT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "context_logs")
os.makedirs(CONTEXT_LOGS_DIR, exist_ok=True)

# Attention thresholds (relative to uniform distribution)
ATTENTION_THRESHOLD_A = float(os.getenv('ATTENTION_THRESHOLD_A', 0.3))  # Below: placeholder
ATTENTION_THRESHOLD_B = float(os.getenv('ATTENTION_THRESHOLD_B', 0.7))  # a-b: summary_brief
ATTENTION_THRESHOLD_C = float(os.getenv('ATTENTION_THRESHOLD_C', 1.5))  # b-c: summary_detailed, >c: full

# Initialize tools
TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
    # Glimpse(),  # TODO: 暂时禁用 glimpse 工具
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    return dt.date.today().strftime("%Y-%m-%d")


class MultiTurnReactAgent(FnCallAgent):
    """
    Multi-turn React Agent with Dynamic Context Focusing.
    
    This agent maintains an external memory store and uses attention-based
    scoring to dynamically select which historical information to include
    in the context at each step.
    """
    
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        
        # Initialize dynamic context components
        self.memory_store = None
        self.rep_generator = None
        self.attention_scorer = None
        self.context_builder = None
        self.current_log_filepath = None  # Current task's log file path
        self.current_context_log_filepath = None  # Context log file for debugging
        
        # Thread pool for async representation generation (non-blocking)
        # Since recent chunks always use "full", we don't need to wait for summaries
        self._rep_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rep_gen")
        self._pending_rep_futures = {}  # chunk_id -> Future
        self._rep_lock = threading.Lock()
        
        if ENABLE_DYNAMIC_CONTEXT:
            self._init_dynamic_context()

    def _init_dynamic_context(self):
        """Initialize dynamic context focusing components."""
        print("[Agent] Initializing Dynamic Context Focusing system...")
        
        # External memory store
        self.memory_store = ExternalMemoryStore()
        
        # Representation generator (uses summary model)
        self.rep_generator = get_representation_generator()
        
        # Attention scorer (uses sentence-transformers)
        try:
            self.attention_scorer = get_attention_scorer()
        except Exception as e:
            print(f"[Agent] Warning: Could not initialize attention scorer: {e}")
            self.attention_scorer = None
        
        # Context builder with custom thresholds
        self.context_builder = ContextBuilder(
            max_context_tokens=MAX_CONTEXT_TOKENS,
            num_recent_full=NUM_RECENT_FULL,
            thresholds={
                "a": ATTENTION_THRESHOLD_A,
                "b": ATTENTION_THRESHOLD_B,
                "c": ATTENTION_THRESHOLD_C
            }
        )
        
        # Set memory store for glimpse tool (disabled)
        # set_glimpse_memory_store(self.memory_store)
        
        print(f"[Agent] Dynamic Context Focusing initialized (max_tokens={MAX_CONTEXT_TOKENS}, thresholds={ATTENTION_THRESHOLD_A}/{ATTENTION_THRESHOLD_B}/{ATTENTION_THRESHOLD_C})")
    
    def _init_memory_log_file(self, question: str = ""):
        """
        Initialize a new memory log file at the start of each task.
        Creates the file immediately with header information.
        Also creates a context log file for debugging repeated tool_call issues.
        
        Args:
            question: The user's question for this task
        
        Returns:
            The filepath of the created log file
        """
        if not ENABLE_DYNAMIC_CONTEXT:
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_{timestamp}.json"
        self.current_log_filepath = os.path.join(MEMORY_LOGS_DIR, filename)
        
        # Also create context log file
        context_filename = f"context_{timestamp}.json"
        self.current_context_log_filepath = os.path.join(CONTEXT_LOGS_DIR, context_filename)
        
        # Initialize log data structure
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "status": "in_progress",
            "chunks": []
        }
        
        # Initialize context log data structure
        context_log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "status": "in_progress",
            "contexts": []  # Each element: {"round": N, "messages": [...], "chunk_levels": {...}}
        }
        
        # Write initial files
        try:
            with open(self.current_log_filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"[Memory] Created log file: {self.current_log_filepath}")
            
            with open(self.current_context_log_filepath, 'w', encoding='utf-8') as f:
                json.dump(context_log_data, f, ensure_ascii=False, indent=2)
            print(f"[Context] Created context log file: {self.current_context_log_filepath}")
            
            return self.current_log_filepath
        except Exception as e:
            print(f"[Memory] Error creating log file: {e}")
            self.current_log_filepath = None
            self.current_context_log_filepath = None
            return None
    
    def _append_chunk_to_log(self, chunk):
        """
        Append a single chunk to the current log file in real-time.
        
        Args:
            chunk: The MemoryChunk to append
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.current_log_filepath is None:
            return
        
        try:
            # Read current file
            with open(self.current_log_filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Add new chunk
            chunk_data = {
                "id": chunk.id,
                "type": chunk.type,
                "timestamp": chunk.timestamp,
                "representations": {
                    "full": chunk.representations.get("full", ""),
                    "summary_detailed": chunk.representations.get("summary_detailed", ""),
                    "summary_brief": chunk.representations.get("summary_brief", ""),
                    "keywords": chunk.representations.get("keywords", [])
                },
                "next_step_relevance": chunk.next_step_relevance,
                "attention_weight": chunk.attention_weight,
                "has_embedding": chunk.embedding is not None,
                "metadata": chunk.metadata
            }
            log_data["chunks"].append(chunk_data)
            
            # Write back to file
            with open(self.current_log_filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Memory] Appended chunk {chunk.id} to log file")
        except Exception as e:
            print(f"[Memory] Error appending chunk to log: {e}")
    
    def _finalize_memory_log(self, termination: str = ""):
        """
        Mark the log file as complete when task finishes.
        Wait for any pending representation generations first.
        
        Args:
            termination: The termination reason
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.current_log_filepath is None:
            return
        
        # First, wait for any pending representation generations to complete
        # This ensures all chunks are written to log with complete representations
        self._wait_for_all_pending_representations()
        
        try:
            with open(self.current_log_filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            log_data["status"] = "completed"
            log_data["termination"] = termination
            log_data["total_chunks"] = len(log_data.get("chunks", []))
            log_data["completed_at"] = datetime.now().isoformat()
            
            with open(self.current_log_filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Memory] Finalized log file: {self.current_log_filepath}")
        except Exception as e:
            print(f"[Memory] Error finalizing log: {e}")
    
    def _wait_for_all_pending_representations(self, timeout: float = 60.0):
        """
        Wait for all pending representation generations to complete.
        Called before finalizing memory log.
        
        Args:
            timeout: Max seconds to wait total
        """
        with self._rep_lock:
            pending = list(self._pending_rep_futures.items())
        
        if not pending:
            return
        
        print(f"[Memory] Waiting for {len(pending)} pending representation generations before finalizing...")
        import time
        start = time.time()
        for chunk_id, future in pending:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                print(f"[Memory] Timeout waiting for representations, some may be incomplete")
                break
            try:
                future.result(timeout=min(remaining, 15.0))
            except Exception as e:
                print(f"[Memory] Warning: Representation generation timeout for chunk {chunk_id}")
        
        # Also finalize context log
        if self.current_context_log_filepath:
            try:
                with open(self.current_context_log_filepath, 'r', encoding='utf-8') as f:
                    context_data = json.load(f)
                
                context_data["status"] = "completed"
                context_data["termination"] = termination
                context_data["completed_at"] = datetime.now().isoformat()
                
                with open(self.current_context_log_filepath, 'w', encoding='utf-8') as f:
                    json.dump(context_data, f, ensure_ascii=False, indent=2)
                
                print(f"[Context] Finalized context log file: {self.current_context_log_filepath}")
            except Exception as e:
                print(f"[Context] Error finalizing context log: {e}")

    def _log_dynamic_context(self, round_num: int, messages: List[Dict], chunk_levels: Dict = None):
        """
        Log the dynamic context sent to the main agent for debugging.
        
        Args:
            round_num: The current round number
            messages: The messages sent to the LLM
            chunk_levels: Optional dict mapping chunk_id to representation level used
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.current_context_log_filepath is None:
            return
        
        try:
            # Read current file
            with open(self.current_context_log_filepath, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            
            # Prepare context entry
            context_entry = {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "num_messages": len(messages),
                "chunk_levels": chunk_levels or {},
                "messages": []
            }
            
            # Store messages (with content truncation for large messages)
            for msg in messages:
                msg_copy = {
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                }
                # Keep full content for analysis
                context_entry["messages"].append(msg_copy)
            
            context_data["contexts"].append(context_entry)
            
            # Write back to file
            with open(self.current_context_log_filepath, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Context] Logged context for round {round_num} ({len(messages)} messages)")
        except Exception as e:
            print(f"[Context] Error logging context: {e}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):
        
        openai_api_key = "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"
        openai_api_base = "https://openrouter.ai/api/v1"
        model_name = "alibaba/tongyi-deepresearch-30b-a3b"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                content = reasoning_content + content                
                
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        """
        Estimate tokens. For OpenRouter/API models, fall back to tiktoken to
        avoid loading a local tokenizer. For local HF models, use the tokenizer.
        """
        # Try tiktoken first (works well for API usage)
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            text = "\n".join([str(m.get("content", "")) for m in messages])
            return len(encoding.encode(text))
        except Exception:
            pass

        # Fallback: try HF tokenizer if a local model path is provided
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(full_prompt, return_tensors="pt")
            return len(tokens["input_ids"][0])
        except Exception:
            # Last resort: rough estimate (1 token ≈ 4 chars)
            text = "\n".join([str(m.get("content", "")) for m in messages])
            return max(1, len(text) // 4)

    def _add_to_memory(self, content: str, chunk_type: str, metadata: dict = None):
        """
        Add a new interaction to the memory store with multi-level representations.
        
        OPTIMIZATION: Representation generation is now ASYNC/non-blocking.
        Since recent NUM_RECENT_FULL chunks always use "full" representation,
        we don't need to wait for summaries before the next step.
        Summaries are generated in background and updated when ready.
        
        Args:
            content: The full content of this step
            chunk_type: Type of chunk (e.g., "assistant_response", "tool_observation")
            metadata: Optional metadata
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.memory_store is None:
            return None
        
        # Step 1: Add chunk immediately with fallback representations (NON-BLOCKING)
        # This allows the next step to proceed without waiting for LLM summarization
        fallback_representations = {
            "full": content,
            "summary_detailed": content[:2000] if len(content) > 2000 else content,
            "summary_brief": content[:200] if len(content) > 200 else content,
            "keywords": []
        }
        
        chunk = self.memory_store.add_chunk(
            chunk_type=chunk_type,
            full_content=fallback_representations["full"],
            summary_detailed=fallback_representations["summary_detailed"],
            summary_brief=fallback_representations["summary_brief"],
            keywords=fallback_representations["keywords"],
            metadata=metadata or {}
        )
        
        print(f"[Memory] Added chunk {chunk.id} ({chunk_type}) [async rep generation started]")
        
        # Step 2: Compute embedding immediately (fast, using full content)
        # This is needed for attention scoring and is fast enough to do sync
        if self.attention_scorer is not None:
            try:
                self.attention_scorer.compute_embeddings_for_chunk(chunk)
                if chunk.embedding is not None:
                    print(f"[Memory] Pre-computed embedding for chunk {chunk.id}")
            except Exception as e:
                print(f"[Memory] Warning: Failed to pre-compute embedding for chunk {chunk.id}: {e}")
        
        # Step 3: Submit async task to generate proper representations (NON-BLOCKING)
        # Since recent chunks use "full" anyway, this doesn't block next step
        # Log file will be written AFTER Gemini generation completes (in _generate_representations_async)
        user_goal = getattr(self, 'user_prompt', '') or ''
        future = self._rep_executor.submit(
            self._generate_representations_async,
            chunk.id, content, chunk_type, user_goal
        )
        
        with self._rep_lock:
            self._pending_rep_futures[chunk.id] = future
        
        # NOTE: Don't write to log here - wait for Gemini to finish first
        # The log will be written in _generate_representations_async after completion
        
        return chunk
    
    def _generate_representations_async(self, chunk_id: int, content: str, chunk_type: str, user_goal: str):
        """
        Generate representations in background thread and update chunk when ready.
        After generation completes, write the chunk to log file.
        
        This is called async by ThreadPoolExecutor.
        """
        chunk = None
        try:
            print(f"[Memory] Async generating representations for chunk {chunk_id}...")
            representations = self.rep_generator.generate_representations(content, chunk_type, user_goal=user_goal)
            
            # Update chunk with proper representations
            if self.memory_store is not None:
                chunk = self.memory_store.get_chunk(chunk_id)
                if chunk is not None:
                    chunk.representations["summary_detailed"] = representations["summary_detailed"]
                    chunk.representations["summary_brief"] = representations["summary_brief"]
                    chunk.representations["keywords"] = representations["keywords"]
                    print(f"[Memory] ✓ Updated chunk {chunk_id} with LLM representations, keywords: {representations['keywords'][:5]}")
                    
                    # NOW write the complete chunk to log file
                    self._append_chunk_to_log(chunk)
            
            # Remove from pending
            with self._rep_lock:
                if chunk_id in self._pending_rep_futures:
                    del self._pending_rep_futures[chunk_id]
                    
        except Exception as e:
            print(f"[Memory] ❌ Async rep generation failed for chunk {chunk_id}: {e}")
            # Still write chunk to log with fallback representations
            if self.memory_store is not None:
                chunk = self.memory_store.get_chunk(chunk_id)
                if chunk is not None:
                    print(f"[Memory] Writing chunk {chunk_id} with fallback representations")
                    self._append_chunk_to_log(chunk)
            
            with self._rep_lock:
                if chunk_id in self._pending_rep_futures:
                    del self._pending_rep_futures[chunk_id]
    
    def _wait_for_pending_representations(self, timeout: float = 5.0):
        """
        Wait for pending representation generations to complete.
        Called before building context for old chunks that might need summaries.
        
        Args:
            timeout: Max seconds to wait per chunk
        """
        with self._rep_lock:
            pending = list(self._pending_rep_futures.items())
        
        if not pending:
            return
        
        print(f"[Memory] Waiting for {len(pending)} pending representation generations...")
        for chunk_id, future in pending:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"[Memory] Warning: Representation generation timeout for chunk {chunk_id}: {e}")

    def _build_dynamic_context(self, system_prompt: str, user_question: str, round_num: int = 0) -> List[Dict]:
        """
        Build dynamic context using attention-based focusing.
        
        Args:
            system_prompt: The system prompt
            user_question: The original user question
            round_num: Current round number (for logging)
        
        Returns:
            List of messages for the LLM
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.memory_store is None or self.memory_store.size() == 0:
            # Fallback to simple context
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        
        # Wait for any pending representations for OLD chunks only
        # (Recent chunks use "full" anyway, so we don't need to wait for them)
        # Only wait for chunks that are NOT in the recent window
        recent_chunks = self.memory_store.get_recent_chunks(NUM_RECENT_FULL)
        recent_ids = set(c.id for c in recent_chunks)
        
        with self._rep_lock:
            old_pending = {cid: fut for cid, fut in self._pending_rep_futures.items() 
                         if cid not in recent_ids}
        
        if old_pending:
            print(f"[Memory] Waiting for {len(old_pending)} old chunk representations...")
            for chunk_id, future in old_pending.items():
                try:
                    future.result(timeout=10.0)  # Wait up to 10s per old chunk
                except Exception as e:
                    print(f"[Memory] Warning: Rep generation timeout for chunk {chunk_id}")
        
        # Compute attention weights
        if self.attention_scorer is not None:
            try:
                attention_weights = self.attention_scorer.score_chunks(
                    self.memory_store,
                    user_question=user_question,
                    num_recent=NUM_RECENT_FULL
                    # temperature defaults to 0.1 for sharper distribution
                )
                # Update weights in memory store
                self.memory_store.update_attention_weights(attention_weights)
            except Exception as e:
                print(f"[Agent] Warning: Attention scoring failed: {e}")
                # Fallback: uniform weights
                attention_weights = {c.id: 1.0 for c in self.memory_store.get_all_chunks()}
        else:
            # Fallback: uniform weights
            attention_weights = {c.id: 1.0 for c in self.memory_store.get_all_chunks()}
        
        # Build context using attention weights
        # Pass round_num for adaptive compression (higher rounds → more compression)
        try:
            messages, chunk_levels = self.context_builder.build_context_simple(
                memory_store=self.memory_store,
                attention_weights=attention_weights,
                system_prompt=system_prompt,
                user_question=user_question,
                round_num=round_num
            )
            
            # Log context stats
            total_tokens = self.count_tokens(messages)
            print(f"[Context] Built dynamic context: {len(messages)} messages, ~{total_tokens} tokens")
            
            # Log dynamic context to file for debugging repeated tool_call issues
            self._log_dynamic_context(round_num, messages, chunk_levels)
            
            return messages
        except Exception as e:
            print(f"[Agent] Warning: Context building failed: {e}")
            # Fallback to simple context
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model = model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        
        # Initialize or reset memory store for this run
        if ENABLE_DYNAMIC_CONTEXT:
            self.memory_store = ExternalMemoryStore()
            # set_glimpse_memory_store(self.memory_store)  # glimpse disabled
            
            # Create log file at task start (real-time logging)
            self._init_memory_log_file(question)
            
            # Add initial user question to memory
            self._add_to_memory(
                content=f"User Question: {question}",
                chunk_type="user_query",
                metadata={"is_initial": True}
            )
        
        # For compatibility: maintain full message history
        full_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                
                # Finalize memory log before returning
                self._finalize_memory_log(termination)
                
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": full_messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            
            round_num += 1
            num_llm_calls_available -= 1
            
            # Build context for LLM
            if ENABLE_DYNAMIC_CONTEXT and self.memory_store.size() > NUM_RECENT_FULL:
                # Use dynamic context focusing
                messages = self._build_dynamic_context(system_prompt, question, round_num)
            else:
                # Use full messages for early rounds or when dynamic context is disabled
                messages = full_messages
                # Still log the context even when using full messages
                if ENABLE_DYNAMIC_CONTEXT:
                    self._log_dynamic_context(round_num, messages, {})
            
            content = self.call_server(messages, planning_port)
            print(f'Round {round_num}: {content}')
            
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            
            # Add assistant response to full messages
            full_messages.append({"role": "assistant", "content": content.strip()})
            
            if '<tool_call>' in content and '</tool_call>' in content:
                # Detect repetition loop: count how many <tool_call> tags
                tool_call_count = content.count('<tool_call>')
                if tool_call_count > 1:
                    print(f"[Agent] WARNING: Detected {tool_call_count} repeated <tool_call> tags - model repetition loop!")
                    # Truncate content to keep only the FIRST tool_call
                    first_end = content.find('</tool_call>') + len('</tool_call>')
                    # Keep thinking part (before first tool_call) + first tool_call only
                    first_start = content.find('<tool_call>')
                    content = content[:first_end]
                    print(f"[Agent] Truncated content to first tool_call only")
                    # Update the message we just added
                    full_messages[-1]["content"] = content.strip()
                
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                tool_name = "unknown"
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                            tool_name = "PythonInterpreter"
                        except:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tool_call_parsed = json5.loads(tool_call)
                        tool_name = tool_call_parsed.get('name', '')
                        tool_args = tool_call_parsed.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                
                tool_response = "<tool_response>\n" + result + "\n</tool_response>"
                full_messages.append({"role": "user", "content": tool_response})
                
                # Add merged round_interaction to memory store (assistant + tool_observation)
                # This combines thinking+tool_call+result into ONE chunk per round
                if ENABLE_DYNAMIC_CONTEXT:
                    round_content = (
                        f"=== Round {round_num} Interaction ===\n"
                        f"[Assistant Response]\n{content.strip()}\n\n"
                        f"[Tool Observation]\n{tool_response}"
                    )
                    self._add_to_memory(
                        content=round_content,
                        chunk_type="round_interaction",
                        metadata={"round": round_num, "tool_name": tool_name}
                    )
            else:
                # No tool call, just add assistant response to memory
                if ENABLE_DYNAMIC_CONTEXT:
                    self._add_to_memory(
                        content=content.strip(),
                        chunk_type="assistant_response",
                        metadata={"round": round_num}
                    )
            
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                full_messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            # Check token count on full messages (for logging)
            token_count = self.count_tokens(full_messages)
            print(f"round: {round_num}, full history token count: {token_count}")

            # With dynamic context, we don't need the hard token limit since context is compressed
            # But keep a sanity check for extremely long sessions
            max_tokens = 128 * 1024  # Allow more since we're compressing
            if token_count > max_tokens:
                print(f"Token quantity exceeds safety limit: {token_count} > {max_tokens}")
                
                # Build final context and request answer
                if ENABLE_DYNAMIC_CONTEXT:
                    final_messages = self._build_dynamic_context(system_prompt, question, round_num)
                else:
                    final_messages = full_messages
                
                # Add termination request
                final_messages.append({
                    "role": "user", 
                    "content": "You have now reached the maximum context length. Stop making tool calls and provide your final answer based on all gathered information:\n<think>your final thinking</think>\n<answer>your answer</answer>"
                })
                
                content = self.call_server(final_messages, planning_port)
                full_messages.append({"role": "assistant", "content": content.strip()})
                
                if '<answer>' in content and '</answer>' in content:
                    prediction = full_messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = full_messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                
                # Finalize memory log before returning
                self._finalize_memory_log(termination)
                
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": full_messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in full_messages[-1]['content']:
            prediction = full_messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        
        # Finalize memory log before returning
        self._finalize_memory_log(termination)
        
        result = {
            "question": question,
            "answer": answer,
            "messages": full_messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            # elif tool_name == "glimpse":
            #     # Glimpse tool uses the memory store directly
            #     result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
