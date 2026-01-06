import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os


# ============ Google Search API 配置 ============
RLAB_GOOGLE_SEARCH_API = os.getenv("RLAB_GOOGLE_SEARCH_API", "")
RLAB_API_URL = os.getenv("RLAB_API_URL", "")
RLAB_API_HEADERS = {
    'Content-Type': 'application/json',
    'rlab-api-key': os.getenv("RLAB_API_KEY", ""),
}


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文字符"""
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    def google_search_with_rlab(self, query: str) -> str:
        """
        使用 RLAB Google Search API 进行搜索
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            str: 格式化的搜索结果
        """
        # 根据查询语言选择搜索区域
        if self.contains_chinese(query):
            req = {
                "payload": {
                    "query": query,
                    "method": "google",
                    "gl": "cn",
                    "hl": "zh-cn"
                }
            }
        else:
            req = {
                "payload": {
                    "query": query,
                    "method": "google",
                    "gl": "us",
                    "hl": "en"
                }
            }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=RLAB_API_URL + RLAB_GOOGLE_SEARCH_API,
                    headers=RLAB_API_HEADERS,
                    data=json.dumps(req),
                    timeout=10
                )
                response.raise_for_status()
                res = response.json()
                search_results = res.get("payload", {}).get("result", [])
                break
            except Exception as e:
                print(f"[Search] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return f"Google search Timeout, return None, Please try again later."
                continue

        try:
            if not search_results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = []
            for idx, page in enumerate(search_results, 1):
                # 提取各字段
                title = page.get("title", "No title")
                link = page.get("link", "")
                date_published = ""
                if page.get("date"):
                    date_published = "\nDate published: " + page["date"]
                
                source = ""
                if page.get("source"):
                    source = "\nSource: " + page["source"]
                
                snippet = ""
                if page.get("snippet"):
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
            
        except Exception as e:
            return f"No results found for '{query}'. Try with a more general query."

    def search_with_rlab(self, query: str) -> str:
        """执行搜索的包装方法"""
        result = self.google_search_with_rlab(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_rlab(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                if isinstance(q, str):
                    responses.append(self.search_with_rlab(q))
                else:
                    print(f"[Search] Skipping non-string query: {q}")
            response = "\n=======\n".join(responses)
            
        return response
