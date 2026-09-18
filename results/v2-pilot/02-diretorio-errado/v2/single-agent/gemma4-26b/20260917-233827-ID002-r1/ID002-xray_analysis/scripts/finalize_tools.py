import os
import json
import random

def main():
    base_dir = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v2/single-agent/gemma4-26b/20260917-233827-ID002-r1/ID002-xray_analysis"
    
    # Tools used
    tools = [
        {
            "tool": "transformers.AutoModelForImageClassification",
            "used_for": "bone fracture detection",
            "trusted": False,
            "reason": "Model reported a fracture with 0.75 confidence, but visual inspection of the original image does not clearly show a cortical break or obvious fracture line in the radius or ulna."
        }
    ]
    
    with open(os.path.join(base_dir, "provenance/tools.json"), 'w') as f:
        json.dump(tools, f, indent=2)

if __name__ == "__main__":
    main()
