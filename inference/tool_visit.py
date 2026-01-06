import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT 
from openai import OpenAI
import tiktoken


# ============ You.com Scrape API 配置 ============
RLAB_YOU_COM_SEARCH = os.getenv("RLAB_YOU_COM_SEARCH", "")
RLAB_API_URL = os.getenv("RLAB_API_URL", "")
RLAB_API_HEADERS = {
    'Content-Type': 'application/json',
    'rlab-api-key': os.getenv("RLAB_API_KEY", ""),
    'rlab-request-source': 'deep-research',
}

# ============ Summary LLM API 配置 ============
def _init_gemini_client():
    """Initialize LLM client for webpage summarization."""
    api_key = os.getenv('SUMMARY_API_KEY', '')
    api_base = os.getenv('SUMMARY_API_BASE', '')
    model_name = os.getenv('SUMMARY_MODEL_NAME', 'google/gemini-2.5-pro')
    
    if not api_key or not api_base:
        print("[Visit] WARNING: SUMMARY_API_KEY or SUMMARY_API_BASE not set, falling back to OpenRouter")
        return None, None
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=120.0,
        max_retries=3
    )
    
    print(f"[Visit] Using Summary LLM: {model_name}")
    return client, model_name

# Initialize Gemini client at module load
_GEMINI_CLIENT, _GEMINI_MODEL = _init_gemini_client()

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """将文本截断到指定的 token 数量"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            response = self.readpage_youcom(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url: 
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage_youcom(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def call_server(self, msgs, max_retries=2):
        """调用 LLM 服务进行内容摘要"""
        
        # ================================================================
        # Summary LLM API (Primary)
        # ================================================================
        if _GEMINI_CLIENT is not None:
            print(f"[Visit] Summarizing with {_GEMINI_MODEL}...")
            for attempt in range(max_retries):
                try:
                    chat_response = _GEMINI_CLIENT.chat.completions.create(
                        model=_GEMINI_MODEL,
                        messages=msgs,
                        temperature=0.7,
                        max_tokens=3000
                    )
                    
                    if hasattr(chat_response, 'success') and chat_response.success == False:
                        print(f"[Visit] API Error: {getattr(chat_response, 'message', 'Unknown')}")
                        continue
                    
                    if not chat_response or not chat_response.choices:
                        print(f"[Visit] Empty response from Gemini API")
                        continue
                    
                    content = chat_response.choices[0].message.content
                    if content:
                        try:
                            json.loads(content)
                        except:
                            # extract json from string 
                            left = content.find('{')
                            right = content.rfind('}') 
                            if left != -1 and right != -1 and left <= right: 
                                content = content[left:right+1]
                        return content
                except Exception as e:
                    print(f"[Visit] Gemini API attempt {attempt + 1} failed: {e}")
                    if attempt == (max_retries - 1):
                        break
                    continue
        
        # ================================================================
        # OpenRouter Fallback
        # ================================================================
        print("[Visit] Falling back to OpenRouter...")
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        
        if not api_key or not url_llm:
            print("[Visit] WARNING: No API credentials available")
            return ""
        
        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                if attempt == (max_retries - 1):
                    return ""
                continue
        
        return ""

    def youcom_readpage(self, url: str) -> str:
        """
        使用 You.com API 读取网页内容
        
        Args:
            url: 需要读取的网页 URL
            
        Returns:
            str: 网页内容（Markdown 格式）或错误信息
        """
        max_retries = 3
        timeout = 30
        
        for attempt in range(max_retries):
            try:
                # 构建请求数据
                data = {
                    'payload': {
                        'payload': {
                            "urls": [url],
                            "format": "markdown"
                        },
                        "method": "visit"
                    }
                }
                
                response = requests.post(
                    url=RLAB_API_URL + RLAB_YOU_COM_SEARCH,
                    headers=RLAB_API_HEADERS,
                    data=json.dumps(data),
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    res = response.json()
                    results = res.get("payload", {}).get("result", [])
                    
                    if results and len(results) > 0:
                        content = results[0].get("markdown", "")
                        if content and len(content) > 10:
                            return content
                        else:
                            print(f"[Visit] Empty content from You.com for {url}")
                            if attempt < max_retries - 1:
                                time.sleep(0.5)
                                continue
                            return "[visit] Empty content."
                    else:
                        print(f"[Visit] No results from You.com for {url}")
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        return "[visit] Failed to read page."
                else:
                    print(f"[Visit] You.com API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return "[visit] Failed to read page."
                    
            except Exception as e:
                print(f"[Visit] Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."

    def youcom_batch_readpage(self, urls: List[str]) -> dict:
        """
        批量读取网页内容
        
        Args:
            urls: URL 列表
            
        Returns:
            dict: URL 到内容的映射
        """
        max_batch_size = 3  # You.com API 每批最多处理 3 个 URL
        timeout = 30
        html_maps = {}
        
        # 分批处理
        batches = [urls[i:i + max_batch_size] for i in range(0, len(urls), max_batch_size)]
        
        for batch in batches:
            try:
                data = {
                    'payload': {
                        'payload': {
                            "urls": batch,
                            "format": "markdown"
                        },
                        "method": "visit"
                    }
                }
                
                response = requests.post(
                    url=RLAB_API_URL + RLAB_YOU_COM_SEARCH,
                    headers=RLAB_API_HEADERS,
                    data=json.dumps(data),
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    res = response.json()
                    results = res.get("payload", {}).get("result", [])
                    
                    for result in results:
                        url = result.get("url", "")
                        content = result.get("markdown", "")
                        if url and content and len(content) > 10:
                            html_maps[url] = content
                            
            except Exception as e:
                print(f"[Visit] Batch request failed: {e}")
                continue
                
        return html_maps

    def html_readpage_youcom(self, url: str) -> str:
        """尝试读取网页内容，最多重试 8 次"""
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.youcom_readpage(url)
            print(f"[Visit] Attempt {attempt + 1}/{max_attempts} using You.com")
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage_youcom(self, url: str, goal: str) -> str:
        """
        读取网页内容并生成摘要
        
        Args:
            url: 需要读取的网页 URL
            goal: 读取目的/目标
            
        Returns:
            str: 格式化的摘要结果
        """
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

        content = self.html_readpage_youcom(url)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            return useful_information

        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information
