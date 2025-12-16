"""
Process benchmark results to extract only <answer> tags from content.

Usage:
    python process_results.py --input iter1.jsonl --output iter1_processed.jsonl
"""

import argparse
import json
import re
from typing import Any, Dict


def extract_answer(content: str) -> str:
    """Extract content between <answer> and </answer> tags."""
    if not content or not isinstance(content, str):
        return content
    
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content  # Return original if no answer tags found


def process_message(message: Dict) -> Dict:
    """Process a single message, extracting answer from content."""
    if not isinstance(message, dict):
        return message
    
    processed = message.copy()
    if "content" in processed and isinstance(processed["content"], str):
        # Only extract answer if content contains answer tags
        if "<answer>" in processed["content"] and "</answer>" in processed["content"]:
            processed["content"] = extract_answer(processed["content"])
    
    return processed


def process_item(item: Dict) -> Dict:
    """Process a single result item."""
    if not isinstance(item, dict):
        return item
    
    processed = item.copy()
    
    # Process messages list if exists
    if "messages" in processed:
        if isinstance(processed["messages"], list):
            processed["messages"] = [process_message(msg) for msg in processed["messages"]]
        elif isinstance(processed["messages"], dict):
            # Handle nested structure
            processed["messages"] = process_item(processed["messages"])
    
    # Also check for prediction field (common in results)
    if "prediction" in processed and isinstance(processed["prediction"], str):
        if "<answer>" in processed["prediction"] and "</answer>" in processed["prediction"]:
            processed["prediction"] = extract_answer(processed["prediction"])
    
    return processed


def process_file(input_path: str, output_path: str):
    """Process the entire results file."""
    processed_items = []
    
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                processed = process_item(item)
                processed_items.append(processed)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Processed {len(processed_items)} items")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Written to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Sample of processed data:")
    print("=" * 50)
    if processed_items:
        sample = processed_items[0]
        # Show truncated version
        sample_str = json.dumps(sample, ensure_ascii=False, indent=2)
        if len(sample_str) > 1000:
            print(sample_str[:1000] + "\n... (truncated)")
        else:
            print(sample_str)


def main():
    parser = argparse.ArgumentParser(description="Process benchmark results to extract answer tags")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSONL file path (default: input_processed.jsonl)")
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        if args.input.endswith('.jsonl'):
            args.output = args.input.replace('.jsonl', '_processed.jsonl')
        elif args.input.endswith('.json'):
            args.output = args.input.replace('.json', '_processed.json')
        else:
            args.output = args.input + '_processed'
    
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()

