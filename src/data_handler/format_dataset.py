import os
import json
import argparse
from tqdm import tqdm

def format_data_for_llama_factory(original_data_path: str, output_file: str):
    """
    Converts the original QA dataset into a JSONL format compatible with LLaMA-Factory.

    Args:
        original_data_path (str): Path to the directory containing the original _qa.json files.
        output_file (str): Path to the output .jsonl file.
    """
    print(f"Reading data from: {original_data_path}")
    
    formatted_data = []
    
    for filename in tqdm(os.listdir(original_data_path), desc="Processing files"):
        if not filename.endswith("_qa.json"):
            continue
            
        file_path = os.path.join(original_data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
                continue

            # Construct the new format
            new_entry = {
                "id": data.get("id", filename.replace("_qa.json", "")),
                # Prepend <image> token for LLaMA-Factory's processor
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n" + data["conversations"][0]["value"]
                    },
                    {
                        "from": "gpt",
                        "value": json.dumps(data["conversations"][1]["value"], ensure_ascii=False)
                    }
                ],
                # Add custom keys for our multimodal data
                "image": data["image"],
                "ground_truth_trajectory": json.loads(data["ego_fut_trajs"]),
                "initial_trajectory": json.loads(data["ego_his_trajs"]) # Using history as initial for now
            }
            formatted_data.append(new_entry)
            
    # Write to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Successfully formatted {len(formatted_data)} samples.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format dataset for LLaMA-Factory.")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./data/train/qa",
        help="Path to the directory with original _qa.json files."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./data/train/formatted_dataset.jsonl",
        help="Path to the output .jsonl file."
    )
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    format_data_for_llama_factory(args.data_path, args.output_file) 