"""
Process benchmark results - extract answer and prediction fields.

Usage:
    python process_results.py
"""

import json
import re


def extract_answer_tag(content: str) -> str:
    """Extract content between <answer> and </answer> tags."""
    if not content or not isinstance(content, str):
        return content
    
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def main():
    input_file = "iter1.jsonl"
    output_file = "iter1_processed.jsonl"
    
    print(f"Reading from: {input_file}")
    
    processed_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                processed = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "prediction": item.get("prediction", ""),
                    "termination": item.get("termination", ""),
                }
                if "error" in item:
                    processed["error"] = item.get("error", "")
                processed_items.append(processed)
            except json.JSONDecodeError:
                continue
    
    print(f"Processed {len(processed_items)} items")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Written to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"{'#':<4} {'Answer':<20} {'Prediction':<30} {'Match':<6}")
    print("=" * 80)
    
    correct = 0
    for i, item in enumerate(processed_items, 1):
        answer = str(item.get("answer", ""))[:18]
        pred = str(item.get("prediction", ""))[:28]
        
        match = "✓" if str(item.get("answer", "")).strip() == str(item.get("prediction", "")).strip() else "✗"
        if match == "✓":
            correct += 1
        
        print(f"{i:<4} {answer:<20} {pred:<30} {match:<6}")
    
    print("=" * 80)
    print(f"Accuracy (exact match): {correct}/{len(processed_items)} = {correct/len(processed_items)*100:.1f}%")


if __name__ == "__main__":
    main()

