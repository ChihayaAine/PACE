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
    
    # NEW: Prompt for summary_detailed - extracts goal-relevant information
    DETAILED_EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content

2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.

3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

    # Prompt template for summary_brief and keywords (unchanged)
    BRIEF_KEYWORDS_PROMPT = """You are a content summarization expert. Given the following content, generate a brief summary and keywords.

## Original Content
{content}

## Task
1. **summary_brief**: A brief summary (1-2 sentences) capturing only the core action and result. Be extremely concise.

2. **keywords**: Extract 5-10 important keywords/entities (names, numbers, technical terms, key concepts).

## Output Format (JSON)
{{
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
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.api_base = api_base or os.environ.get("API_BASE", "")
        self.model_name = model_name or os.environ.get("SUMMARY_MODEL_NAME", "")
        self.max_input_tokens = max_input_tokens
        
        # Initialize OpenAI client if credentials available
        self.client = None
        if self.api_key and self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Token encoder for truncation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
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
        
        # Try to use LLM for summarization
        if self.client and self.model_name:
            try:
                # Generate summary_detailed using the new DETAILED_EXTRACTOR_PROMPT
                detailed_result = self._generate_detailed_with_llm(content, user_goal)
                if detailed_result:
                    result["summary_detailed"] = detailed_result
                
                # Generate summary_brief and keywords using BRIEF_KEYWORDS_PROMPT
                brief_result = self._generate_brief_with_llm(content)
                if brief_result:
                    result["summary_brief"] = brief_result.get("summary_brief", "")
                    result["keywords"] = brief_result.get("keywords", [])
                
                # If we got at least one result, return
                if detailed_result or brief_result:
                    return result
            except Exception as e:
                print(f"[RepresentationGenerator] LLM summarization failed: {e}")
        
        # Fallback: rule-based summarization
        result["summary_detailed"] = self._generate_detailed_fallback(content)
        result["summary_brief"] = self._generate_brief_fallback(content)
        result["keywords"] = self._extract_keywords_simple(content)
        
        return result
    
    def _generate_detailed_with_llm(self, content: str, user_goal: str = "") -> Optional[str]:
        """
        Generate summary_detailed using the DETAILED_EXTRACTOR_PROMPT.
        
        This prompt extracts goal-relevant information with rational, evidence, and summary.
        The 'evidence' field is used as summary_detailed.
        """
        # Truncate content if too long
        truncated_content = self.truncate_to_tokens(content, self.max_input_tokens)
        
        # Use user_goal if provided, otherwise use a generic goal
        goal = user_goal if user_goal else "Extract and summarize the key information from this content."
        
        prompt = self.DETAILED_EXTRACTOR_PROMPT.format(
            webpage_content=truncated_content,
            goal=goal
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Find JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                
                # Combine rational + evidence + summary for comprehensive summary_detailed
                parts = []
                if parsed.get("rational"):
                    parts.append(f"[Rational] {parsed['rational']}")
                if parsed.get("evidence"):
                    parts.append(f"[Evidence] {parsed['evidence']}")
                if parsed.get("summary"):
                    parts.append(f"[Summary] {parsed['summary']}")
                
                return "\n\n".join(parts) if parts else parsed.get("evidence", "")
                
        except Exception as e:
            print(f"[RepresentationGenerator] Error in detailed extraction: {e}")
        
        return None
    
    def _generate_brief_with_llm(self, content: str) -> Optional[Dict]:
        """
        Generate summary_brief and keywords using BRIEF_KEYWORDS_PROMPT.
        """
        # Truncate content if too long
        truncated_content = self.truncate_to_tokens(content, self.max_input_tokens)
        
        prompt = self.BRIEF_KEYWORDS_PROMPT.format(content=truncated_content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Find JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                
                return {
                    "summary_brief": parsed.get("summary_brief", ""),
                    "keywords": parsed.get("keywords", [])
                }
        except Exception as e:
            print(f"[RepresentationGenerator] Error parsing brief/keywords: {e}")
        
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

