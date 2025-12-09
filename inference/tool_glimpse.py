"""
Glimpse Tool for Dynamic Context Focusing

This tool allows the agent to retrieve full details of historical steps
that may have been compressed or omitted in the dynamic context.

It serves as a fallback mechanism when the attention scorer incorrectly
assigns low relevance to important information.
"""

from typing import Dict, List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool


# Global reference to the memory store (set during agent initialization)
_memory_store = None


def set_glimpse_memory_store(store):
    """Set the memory store for the glimpse tool to use."""
    global _memory_store
    _memory_store = store


def get_glimpse_memory_store():
    """Get the current memory store."""
    return _memory_store


@register_tool("glimpse", allow_overwrite=True)
class Glimpse(BaseTool):
    """
    Tool for retrieving full details of historical interaction steps.
    
    Use this tool when you need complete information about a previous step
    that may have been summarized or compressed in the context.
    """
    
    name = "glimpse"
    description = """Retrieve full details of a historical interaction step.
Use this tool when you need complete information about a previous step that appears 
summarized or compressed (e.g., "[Step X: tool_call - details available via glimpse]").

You can either:
1. Provide a specific step ID to get its full content
2. Provide keywords to search for relevant steps
3. List recent N steps to see what's available"""
    
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {
                "type": "integer",
                "description": "The specific step ID to retrieve (e.g., 3 for Step 3)"
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keywords to search for in historical steps"
            },
            "list_steps": {
                "type": "integer",
                "description": "List the last N steps with their IDs and types"
            }
        }
    }
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute the glimpse tool.
        
        Args:
            params: Dictionary with optional keys: step_id, keywords, list_steps
        
        Returns:
            The requested information or an error message
        """
        global _memory_store
        
        if _memory_store is None:
            return "[Glimpse] Error: Memory store not initialized. Cannot access historical steps."
        
        # Parse parameters
        step_id = params.get("step_id")
        keywords = params.get("keywords", [])
        list_steps = params.get("list_steps")
        
        results = []
        
        # Handle list_steps request
        if list_steps is not None:
            results.append(self._list_recent_steps(int(list_steps)))
        
        # Handle step_id request
        if step_id is not None:
            results.append(self._get_step_by_id(int(step_id)))
        
        # Handle keywords search
        if keywords:
            results.append(self._search_by_keywords(keywords))
        
        # If no specific request, show usage
        if not results:
            return self._usage_help()
        
        return "\n\n".join(results)
    
    def _get_step_by_id(self, step_id: int) -> str:
        """Retrieve full content of a specific step."""
        chunk = _memory_store.get_chunk(step_id)
        
        if chunk is None:
            return f"[Glimpse] Step {step_id} not found. Use list_steps to see available steps."
        
        full_content = chunk.representations.get("full", "")
        
        return f"""=== Glimpse: Step {step_id} ({chunk.type}) ===
Timestamp: {chunk.timestamp}

Full Content:
{full_content}

=== End Glimpse ==="""
    
    def _search_by_keywords(self, keywords: List[str]) -> str:
        """Search for steps containing the specified keywords."""
        matching_chunks = _memory_store.search_by_keywords(keywords)
        
        if not matching_chunks:
            return f"[Glimpse] No steps found matching keywords: {', '.join(keywords)}"
        
        results = [f"=== Glimpse: Found {len(matching_chunks)} steps matching '{', '.join(keywords)}' ===\n"]
        
        for chunk in matching_chunks[:5]:  # Limit to top 5
            full_content = chunk.representations.get("full", "")
            # Truncate if too long
            if len(full_content) > 2000:
                full_content = full_content[:2000] + "\n... [truncated, use step_id for full content]"
            
            results.append(f"""--- Step {chunk.id} ({chunk.type}) ---
{full_content}
""")
        
        results.append("=== End Glimpse Search ===")
        return "\n".join(results)
    
    def _list_recent_steps(self, n: int) -> str:
        """List the most recent N steps."""
        all_chunks = _memory_store.get_all_chunks()
        
        if not all_chunks:
            return "[Glimpse] No historical steps available."
        
        # Get last N chunks
        recent = all_chunks[-n:] if len(all_chunks) >= n else all_chunks
        
        lines = [f"=== Glimpse: Last {len(recent)} Steps ===\n"]
        
        for chunk in recent:
            keywords = chunk.representations.get("keywords", [])
            keywords_str = f" | Keywords: {', '.join(keywords[:5])}" if keywords else ""
            brief = chunk.representations.get("summary_brief", "")
            if len(brief) > 100:
                brief = brief[:100] + "..."
            
            lines.append(f"Step {chunk.id}: {chunk.type}{keywords_str}")
            if brief:
                lines.append(f"    Brief: {brief}")
            lines.append("")
        
        lines.append("Use glimpse with step_id to get full details of any step.")
        lines.append("=== End List ===")
        
        return "\n".join(lines)
    
    def _usage_help(self) -> str:
        """Return usage instructions."""
        return """[Glimpse Tool Usage]

1. Get full content of a specific step:
   {"step_id": 5}

2. Search steps by keywords:
   {"keywords": ["price", "GPU"]}

3. List recent steps:
   {"list_steps": 10}

4. Combine multiple queries:
   {"step_id": 3, "list_steps": 5}

Use this tool when you see summarized history like "[Step X: tool_call - details available via glimpse]"
and need the complete information to continue your investigation."""

