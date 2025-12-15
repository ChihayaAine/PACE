#!/usr/bin/env python3
"""
Simple test script for Alibaba IAI Gemini API
Usage: EMPID=your_empid python test_gemini.py
"""

import os
from openai import OpenAI

def test_gemini():
    # 环境配置
    PROJECT_ENV = os.getenv('PROJECT_ENV', 'local')
    print(f"PROJECT_ENV: {PROJECT_ENV}")
    
    # EMPID 检查
    if PROJECT_ENV == 'local':
        empid = os.getenv('EMPID')
        if not empid:
            print("ERROR: EMPID environment variable not set!")
            print("Usage: EMPID=your_empid python test_gemini.py")
            return
    else:
        empid = 'buyer-agent-online' if PROJECT_ENV == 'online' else 'buyer-agent-pre'
    
    print(f"EMPID: {empid}")
    
    # Endpoint 配置 (同 accio-agent-task)
    google_endpoint = (
        'http://iai.vipserver:7001/google'
        if PROJECT_ENV == 'online'
        else 'https://iai.alibaba-inc.com/google'
    )
    api_key = "accio-agent" if PROJECT_ENV in ["online", "pre"] else "icbu-buyer-agent-algo"
    model_name = "google/gemini-2.5-pro"
    
    print(f"Endpoint: {google_endpoint}")
    print(f"API Key: {api_key}")
    print(f"Model: {model_name}")
    print("-" * 50)
    
    # 创建客户端
    client = OpenAI(
        default_headers={'empId': empid, 'iai-tag': 'accio'},
        api_key=api_key,
        base_url=google_endpoint,
        timeout=60.0,
        max_retries=3
    )
    
    # 测试调用
    print("Testing API call...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, please respond with 'API test successful!'"}],
            temperature=0.3,
            max_tokens=100
        )
        
        # 检查阿里 IAI 特有的错误响应
        if hasattr(response, 'success') and response.success == False:
            print(f"API Error!")
            print(f"  Code: {getattr(response, 'code', 'N/A')}")
            print(f"  Message: {getattr(response, 'message', 'N/A')}")
            print(f"  Request ID: {getattr(response, 'iai_request_id', 'N/A')}")
            return
        
        # 检查响应
        if response and response.choices:
            content = response.choices[0].message.content
            print(f"SUCCESS! Response: {content}")
        else:
            print(f"Empty response: {response}")
            
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini()

