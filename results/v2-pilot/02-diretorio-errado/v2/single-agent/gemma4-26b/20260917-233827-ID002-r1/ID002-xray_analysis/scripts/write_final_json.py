import os
import json
import random

def main():
    base_dir = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v2/single-agent/gemma4-26b/20260917-233827-ID002-r1/ID002-xray_analysis"
    
    # Final JSON content
    final_json = {
      "image": "ID002-xray.png",
      "findings": "No definitive fracture or cortical discontinuity is visible in the radius, ulna, or elbow joint on this lateral view.",
      "impression": "Normal lateral radiograph of the forearm and elbow.",
      "limitations": "Lateral projection limits assessment of certain fracture planes; single view provided."
    }
    
    # Write the final JSON file with the EXACT filename requested
    with open(os.path.join(base_dir, "ID002-xray.png.json"), 'w') as f:
        json.dump(final_json, f, indent=2)

if __name__ == "__main__":
    main()
