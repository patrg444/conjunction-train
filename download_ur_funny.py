#!/usr/bin/env python3
"""
This script downloads the UR-FUNNY dataset and processes it into our
standard manifest format. The dataset contains ~3800 jokes from stand-up comedy
videos with manually annotated humor labels.

Reference: Hasan et al. (2019) - "UR-FUNNY: A Multimodal Language Dataset for Understanding Humor"
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import requests # Use requests for direct download

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("datasets/humor_datasets/ur_funny", exist_ok=True)
    os.makedirs("datasets/manifests/humor", exist_ok=True)

def download_ur_funny():
    """Download UR-FUNNY dataset JSON directly from GitHub"""
    ur_funny_dir = "datasets/humor_datasets/ur_funny"
    json_path = f"{ur_funny_dir}/ur_funny_final.json"

    # Check if the final JSON file already exists
    if os.path.exists(json_path):
        print("UR-FUNNY dataset JSON already exists. Skipping download.")
        return json_path

    print("Attempting to download UR-FUNNY dataset JSON directly from GitHub...")
    raw_url = "https://raw.githubusercontent.com/ROC-HCI/UR-FUNNY/master/annotations/ur_funny_final.json"

    try:
        response = requests.get(raw_url, timeout=30)
        if response.status_code == 200:
            # Save the file
            with open(json_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded UR-FUNNY dataset to {json_path}")
            return json_path
        else:
            print(f"Error: Failed to download from {raw_url} (Status code: {response.status_code})")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during direct download: {e}")
        print("Please ensure you have network access and the URL is correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

def process_ur_funny_to_manifest(json_path):
    """Process UR-FUNNY dataset JSON file into our manifest format"""
    if not json_path or not os.path.exists(json_path):
        print("Error: UR-FUNNY dataset JSON file not found. Cannot build manifest.")
        return {}

    print(f"Processing {json_path}...")

    try:
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create a list of unique video IDs
        video_ids = list(set(item['video_id'] for item in data))

        # Shuffle the video IDs (ensures random split)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(video_ids)

        # Split by video_id to avoid data leakage (80/20 split)
        train_size = int(0.8 * len(video_ids))
        train_video_ids = set(video_ids[:train_size])
        val_video_ids = set(video_ids[train_size:])

        # Process data into train and validation manifests
        train_data = []
        val_data = []

        # Track statistics
        train_humor_count = 0
        train_non_humor_count = 0
        val_humor_count = 0
        val_non_humor_count = 0

        for i, item in enumerate(tqdm(data)):
            video_id = item['video_id']

            # Create manifest row
            manifest_row = {
                "talk_id": f"urfunny_{video_id}_{i}",
                "title": "UR-FUNNY",
                "text": item.get('sentence', ''),
                "label": int(item.get('humor', 0))
            }

            # Add to appropriate split based on video_id
            if video_id in train_video_ids:
                train_data.append(manifest_row)
                if manifest_row["label"] == 1:
                    train_humor_count += 1
                else:
                    train_non_humor_count += 1
            else:
                val_data.append(manifest_row)
                if manifest_row["label"] == 1:
                    val_humor_count += 1
                else:
                    val_non_humor_count += 1

        # Create DataFrames and save as CSV
        output_files = {}

        if train_data:
            train_output = "datasets/manifests/humor/ur_funny_train_humor.csv"
            train_df = pd.DataFrame(train_data)
            train_df.to_csv(train_output, index=False)
            output_files["train"] = train_output
            print(f"Created train manifest: {train_output}")
            print(f"  Total samples: {len(train_df)}")
            print(f"  Label 0 (non-humorous): {train_non_humor_count}")
            print(f"  Label 1 (humorous): {train_humor_count}")

        if val_data:
            val_output = "datasets/manifests/humor/ur_funny_val_humor.csv"
            val_df = pd.DataFrame(val_data)
            val_df.to_csv(val_output, index=False)
            output_files["val"] = val_output
            print(f"Created validation manifest: {val_output}")
            print(f"  Total samples: {len(val_df)}")
            print(f"  Label 0 (non-humorous): {val_non_humor_count}")
            print(f"  Label 1 (humorous): {val_humor_count}")

        return output_files

    except Exception as e:
        print(f"Error processing UR-FUNNY dataset: {e}")
        return {}

def main():
    """Main function to download and process UR-FUNNY dataset"""
    setup_directories()
    json_path = download_ur_funny()
    # Only proceed to processing if the JSON was successfully downloaded/found
    if json_path:
        output_files = process_ur_funny_to_manifest(json_path)

        if output_files:
            print("\nUR-FUNNY dataset processed successfully!")
            print("\nManifest files created:")
            for split, file in output_files.items():
                print(f"  {split}: {file}")

            print("\nYou can now train a model using these manifest files.")
            # Example command is for text-only, which is not the current goal.
            # We will rely on the shell script to call the correct training script.
            # print("Example command:")
            # print(f"  python enhanced_train_distil_humor.py --train_manifest {output_files.get('train', '')} --val_manifest {output_files.get('val', '')}")
        else:
            print("\nFailed to process UR-FUNNY dataset.")
    else:
        print("\nUR-FUNNY dataset download failed. Cannot proceed with processing.")


if __name__ == "__main__":
    main()
