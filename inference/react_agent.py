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
        
        Args:
            content: The full content of this step
            chunk_type: Type of chunk (e.g., "assistant_response", "tool_observation")
            metadata: Optional metadata
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.memory_store is None:
            return None
        
        # Generate multi-level representations
        try:
            representations = self.rep_generator.generate_representations(content, chunk_type)
        except Exception as e:
            print(f"[Agent] Warning: Failed to generate representations: {e}")
            representations = {
                "full": content,
                "summary_detailed": content[:2000] if len(content) > 2000 else content,
                "summary_brief": content[:200] if len(content) > 200 else content,
                "keywords": []
            }
        
        # Add to memory store
        chunk = self.memory_store.add_chunk(
            chunk_type=chunk_type,
            full_content=representations["full"],
            summary_detailed=representations["summary_detailed"],
            summary_brief=representations["summary_brief"],
            keywords=representations["keywords"],
            metadata=metadata or {}
        )
        
        print(f"[Memory] Added chunk {chunk.id} ({chunk_type}), keywords: {representations['keywords'][:5]}")
        return chunk

    def _build_dynamic_context(self, system_prompt: str, user_question: str) -> List[Dict]:
        """
        Build dynamic context using attention-based focusing.
        
        Args:
            system_prompt: The system prompt
            user_question: The original user question
        
        Returns:
            List of messages for the LLM
        """
        if not ENABLE_DYNAMIC_CONTEXT or self.memory_store is None or self.memory_store.size() == 0:
            # Fallback to simple context
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        
        # Compute attention weights
        if self.attention_scorer is not None:
            try:
                attention_weights = self.attention_scorer.score_chunks(
                    self.memory_store,
                    num_recent=NUM_RECENT_FULL,
                    temperature=1.0
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
        try:
            messages = self.context_builder.build_context_simple(
                memory_store=self.memory_store,
                attention_weights=attention_weights,
                system_prompt=system_prompt,
                user_question=user_question
            )
            
            # Log context stats
            total_tokens = self.count_tokens(messages)
            print(f"[Context] Built dynamic context: {len(messages)} messages, ~{total_tokens} tokens")
            
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
                messages = self._build_dynamic_context(system_prompt, question)
            else:
                # Use full messages for early rounds or when dynamic context is disabled
                messages = full_messages
            
            content = self.call_server(messages, planning_port)
            print(f'Round {round_num}: {content}')
            
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            
            # Add assistant response to full messages
            full_messages.append({"role": "assistant", "content": content.strip()})
            
            # Add assistant response to memory store
            if ENABLE_DYNAMIC_CONTEXT:
                self._add_to_memory(
                    content=content.strip(),
                    chunk_type="assistant_response",
                    metadata={"round": round_num}
                )
            
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
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
                
                # Add tool observation to memory store
                if ENABLE_DYNAMIC_CONTEXT:
                    self._add_to_memory(
                        content=tool_response,
                        chunk_type="tool_observation",
                        metadata={"round": round_num, "tool_name": tool_name if 'tool_name' in dir() else "unknown"}
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
                    final_messages = self._build_dynamic_context(system_prompt, question)
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
