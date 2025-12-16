"""
Representation Generator for Dynamic Context Focusing

This module generates multi-level representations for memory chunks:
- full: Complete original content
- summary_detailed: Detailed summary preserving key information
- summary_brief: Brief summary with core facts only
- keywords: List of important keywords/entities

Uses a "Compute-on-Write" strategy to avoid high latency during context building.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import tiktoken


class RepresentationGenerator:
    """
    Generates multi-level representations for memory chunks.
    
    Uses the existing summary model (from environment variables) to create
    hierarchical summaries at different detail levels.
    """
    
    # Unified prompt for all representation levels (single LLM call)
    MULTI_LEVEL_EXTRACTOR_PROMPT = """Please process the following content and user goal to generate multi-level representations:

## **Content** 
{content}

## **User Goal**
{goal}

## **Task Guidelines**

Generate THREE levels of representation:

### 1. Detailed Extraction (for summary_detailed)
- **rational**: Locate the **specific sections/data** directly related to the user's goal
- **evidence**: Extract the **most relevant information**, never miss important info, output **full original context** as far as possible (can be 2-3 paragraphs)
- **summary**: Organize into a concise paragraph with logical flow, judge the contribution to the goal

### 2. Brief Summary (for summary_brief)
- 1-2 sentences capturing only the core action and result
- Be extremely concise

### 3. Keywords
- Extract 5-10 important keywords/entities (names, numbers, technical terms, key concepts)

## **Output Format (JSON)**
{{
  "rational": "...",
  "evidence": "...",
  "summary": "...",
  "summary_brief": "...",
  "keywords": ["keyword1", "keyword2", ...]
}}

Output ONLY the JSON, no other text."""

    def __init__(self, 
                 api_key: str = None,
                 api_base: str = None,
                 model_name: str = None,
                 max_input_tokens: int = 8000):
        """
        Initialize the representation generator.
        
        Args:
            api_key: API key for the summary model
            api_base: API base URL
            model_name: Model name to use for summarization
            max_input_tokens: Maximum input tokens before truncation
        """
        self.max_input_tokens = max_input_tokens
        
        # Token encoder for truncation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
        
        # ========================================================================
        # Alibaba IAI Gemini Client (same as accio-agent-task)
        # ========================================================================
        PROJECT_ENV = os.getenv('PROJECT_ENV', 'local')
        
        if PROJECT_ENV == 'local':
            empid = os.getenv('EMPID')
            if not empid:
                raise ValueError(
                    "EMPID environment variable is required for local environment. "
                    "Please set EMPID environment variable with your employee ID. "
                    "Example: export EMPID=your-employee-id"
                )
        else:
            empid = 'buyer-agent-online' if PROJECT_ENV == 'online' else 'buyer-agent-pre'
        
        # Use /google endpoint for Gemini models
        google_endpoint = (
            'http://iai.vipserver:7001/google'
            if PROJECT_ENV == 'online'
            else 'https://iai.alibaba-inc.com/google'
        )
        api_key = "accio-agent" if PROJECT_ENV in ["online", "pre"] else "icbu-buyer-agent-algo"
        
        self.client = OpenAI(
            default_headers={'empId': empid, 'iai-tag': 'accio'},
            api_key=api_key,
            base_url=google_endpoint,
            timeout=120.0,
            max_retries=3
        )
        self.model_name = "google/gemini-2.5-pro"
        
        print(f"[RepresentationGenerator] ========================================")
        print(f"[RepresentationGenerator] Using Alibaba IAI (NOT OpenRouter)")
        print(f"[RepresentationGenerator] Model: {self.model_name}")
        print(f"[RepresentationGenerator] EmpID: {empid}")
        print(f"[RepresentationGenerator] Endpoint: {google_endpoint}")
        print(f"[RepresentationGenerator] API Key: {api_key}")
        print(f"[RepresentationGenerator] ========================================")
        
        # ========================================================================
        # OpenRouter Client (Backup - commented out)
        # ========================================================================
        # self.api_key = api_key or os.environ.get("API_KEY", "")
        # self.api_base = api_base or os.environ.get("API_BASE", "")
        # self.model_name = model_name or os.environ.get("SUMMARY_MODEL_NAME", "")
        # 
        # # Initialize OpenAI client if credentials available
        # self.client = None
        # if self.api_key and self.api_base:
        #     self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4  # Rough estimate
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.encoding.decode(tokens[:max_tokens])
        else:
            # Rough truncation
            char_limit = max_tokens * 4
            return text[:char_limit] if len(text) > char_limit else text
    
    def generate_representations(self, content: str, content_type: str = "unknown", user_goal: str = "") -> Dict:
        """
        Generate multi-level representations for the given content.
        
        Args:
            content: The original content to summarize
            content_type: Type of content (e.g., "tool_observation", "assistant_response")
            user_goal: The user's original question/goal (used for goal-aware extraction)
        
        Returns:
            Dictionary with keys: full, summary_detailed, summary_brief, keywords
        """
        # Always include full content
        result = {
            "full": content,
            "summary_detailed": "",
            "summary_brief": "",
            "keywords": []
        }
        
        # If content is short, use it directly for all levels
        token_count = self.count_tokens(content)
        if token_count < 200:
            result["summary_detailed"] = content
            result["summary_brief"] = self._extract_first_sentence(content)
            result["keywords"] = self._extract_keywords_simple(content)
            return result
        
        # Try to use LLM for summarization (single call for all levels)
        # Now using Alibaba IAI Gemini API
        if self.client:
            try:
                llm_result = self._generate_all_with_llm(content, user_goal)
                if llm_result:
                    result.update(llm_result)
                    print(f"[RepresentationGenerator] ✓ LLM summarization successful")
                    return result
                else:
                    print(f"[RepresentationGenerator] ⚠ LLM returned None, using fallback")
            except Exception as e:
                print(f"[RepresentationGenerator] ❌ LLM summarization exception: {e}")
        else:
            print(f"[RepresentationGenerator] ⚠ No client available, using fallback")
        
        # Fallback: rule-based summarization
        print(f"[RepresentationGenerator] Using FALLBACK (simple truncation)...")
        result["summary_detailed"] = self._generate_detailed_fallback(content)
        result["summary_brief"] = self._generate_brief_fallback(content)
        result["keywords"] = self._extract_keywords_simple(content)
        
        return result
    
    def _generate_all_with_llm(self, content: str, user_goal: str = "") -> Optional[Dict]:
        """
        Generate all representation levels in a single LLM call.
        
        Uses Alibaba IAI Gemini API (gemini-2.5-pro).
        
        Returns:
            Dict with summary_detailed, summary_brief, and keywords
        """
        # Truncate content if too long
        truncated_content = self.truncate_to_tokens(content, self.max_input_tokens)
        
        # Use user_goal if provided, otherwise use a generic goal
        goal = user_goal if user_goal else "Extract and summarize the key information from this content."
        
        prompt = self.MULTI_LEVEL_EXTRACTOR_PROMPT.format(
            content=truncated_content,
            goal=goal
        )
        
        try:
            # ================================================================
            # Alibaba IAI Gemini API call
            # ================================================================
            print(f"[RepresentationGenerator] Summarizing with Alibaba IAI Gemini ({self.model_name})...")
            print(f"[RepresentationGenerator] Content length: {len(content)} chars, Goal: {goal[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            # Debug: print full response structure
            print(f"[RepresentationGenerator] Response type: {type(response)}")
            
            # Handle API error response (Alibaba IAI specific)
            if hasattr(response, 'success') and response.success == False:
                print(f"[RepresentationGenerator] ❌ API Error: code={getattr(response, 'code', 'N/A')}, message={getattr(response, 'message', 'N/A')}")
                return None
            
            # Handle None or empty response
            if not response or not response.choices:
                print(f"[RepresentationGenerator] ❌ Empty response from API")
                print(f"[RepresentationGenerator] Full response: {response}")
                return None
            
            choice = response.choices[0]
            if not choice.message or choice.message.content is None:
                # Some models return content in different fields
                print(f"[RepresentationGenerator] ❌ No content in response. Choice: {choice}")
                return None
            
            response_text = choice.message.content.strip()
            print(f"[RepresentationGenerator] ✓ Got response: {len(response_text)} chars")
            
            # ================================================================
            # OpenRouter API call (commented out - backup)
            # ================================================================
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.3,
            #     max_tokens=3000
            # )
            # response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
                print(f"[RepresentationGenerator] Extracted JSON from code block")
            
            # Find JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                try:
                    parsed = json.loads(json_str)
                    print(f"[RepresentationGenerator] ✓ Parsed JSON successfully, keys: {list(parsed.keys())}")
                except json.JSONDecodeError as je:
                    print(f"[RepresentationGenerator] ❌ JSON parse error: {je}")
                    print(f"[RepresentationGenerator] JSON string (first 500 chars): {json_str[:500]}")
                    return None
                
                # Build summary_detailed from rational + evidence + summary
                detailed_parts = []
                if parsed.get("rational"):
                    detailed_parts.append(f"[Rational] {parsed['rational']}")
                if parsed.get("evidence"):
                    detailed_parts.append(f"[Evidence] {parsed['evidence']}")
                if parsed.get("summary"):
                    detailed_parts.append(f"[Summary] {parsed['summary']}")
                
                summary_detailed = "\n\n".join(detailed_parts) if detailed_parts else parsed.get("evidence", "")
                
                result = {
                    "summary_detailed": summary_detailed,
                    "summary_brief": parsed.get("summary_brief", ""),
                    "keywords": parsed.get("keywords", [])
                }
                print(f"[RepresentationGenerator] ✓ Generated representations: detailed={len(summary_detailed)} chars, brief={len(result['summary_brief'])} chars, keywords={len(result['keywords'])}")
                return result
            else:
                print(f"[RepresentationGenerator] ❌ No JSON object found in response")
                print(f"[RepresentationGenerator] Response text (first 500 chars): {response_text[:500]}")
                return None
                
        except Exception as e:
            print(f"[RepresentationGenerator] ❌ Error in LLM extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _generate_detailed_fallback(self, content: str) -> str:
        """
        Fallback: Generate detailed summary by extracting key sections.
        """
        lines = content.split('\n')
        
        # Keep first 30% and last 20% of lines, plus lines with numbers/URLs
        important_lines = []
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Keep early lines
            if i < total_lines * 0.3:
                important_lines.append(line_stripped)
            # Keep late lines
            elif i > total_lines * 0.8:
                important_lines.append(line_stripped)
            # Keep lines with numbers, URLs, or key patterns
            elif re.search(r'\d+\.?\d*%?|\$\d+|https?://|www\.', line_stripped):
                important_lines.append(line_stripped)
            # Keep lines that look like headers or key points
            elif re.match(r'^#+\s|^\*\s|^-\s|^\d+\.\s', line_stripped):
                important_lines.append(line_stripped)
        
        result = '\n'.join(important_lines)
        
        # Ensure we don't exceed a reasonable length
        max_chars = 2000
        if len(result) > max_chars:
            result = result[:max_chars] + "..."
        
        return result
    
    def _generate_brief_fallback(self, content: str) -> str:
        """
        Fallback: Generate brief summary from first meaningful sentence.
        """
        # Extract first paragraph or first few sentences
        paragraphs = content.split('\n\n')
        first_para = paragraphs[0] if paragraphs else content
        
        # Get first 1-2 sentences
        sentences = re.split(r'[.!?]\s+', first_para)
        brief = '. '.join(sentences[:2]).strip()
        
        # Truncate if still too long
        if len(brief) > 300:
            brief = brief[:297] + "..."
        
        return brief
    
    def _extract_first_sentence(self, content: str) -> str:
        """Extract the first sentence from content."""
        # Find first sentence ending
        match = re.search(r'^[^.!?]*[.!?]', content)
        if match:
            return match.group(0).strip()
        # If no sentence ending, take first 100 chars
        return content[:100].strip() + ("..." if len(content) > 100 else "")
    
    def _extract_keywords_simple(self, content: str) -> List[str]:
        """
        Simple keyword extraction using pattern matching.
        """
        keywords = set()
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        for url in urls[:3]:
            # Extract domain
            domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain:
                keywords.add(domain.group(1))
        
        # Extract numbers with context
        numbers = re.findall(r'\$?[\d,]+\.?\d*%?', content)
        for num in numbers[:5]:
            if len(num) > 2:  # Skip single digits
                keywords.add(num)
        
        # Extract capitalized phrases (potential entities)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for cap in caps[:10]:
            if len(cap) > 3:
                keywords.add(cap)
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', content)
        for q in quoted[:5]:
            if len(q) < 50:
                keywords.add(q)
        
        return list(keywords)[:10]
    
    def generate_batch(self, contents: List[Tuple[str, str]]) -> List[Dict]:
        """
        Generate representations for multiple contents.
        
        Args:
            contents: List of (content, content_type) tuples
        
        Returns:
            List of representation dictionaries
        """
        return [self.generate_representations(content, ctype) 
                for content, ctype in contents]


# Singleton instance for convenience
_default_generator = None

def get_representation_generator() -> RepresentationGenerator:
    """Get or create the default representation generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = RepresentationGenerator()
    return _default_generator

