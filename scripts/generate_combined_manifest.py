#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# --- Configuration ---
RAVDESS_MANIFEST_PATH = "/home/ubuntu/datasets/video_manifest.csv"
HUBERT_SPLITS_DIR = "/home/ubuntu/conjunction-train/splits"
CREMA_D_VIDEO_DIR = "/home/ubuntu/datasets/crema_d_videos"
OUTPUT_DIR = "/home/ubuntu/datasets"
OUTPUT_PREFIX = "combined_manifest"
# --- End Configuration ---

# Emotion labels common to both datasets (as defined in train_slowfast_emotion.py)
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

def get_crema_video_path(basename):
    """Constructs the expected path for a CREMA-D video file."""
    # Check for .flv first, then maybe .mp4 if conversion happened
    flv_path = Path(CREMA_D_VIDEO_DIR) / f"{basename}.flv"
    if flv_path.exists():
        return str(flv_path)
    mp4_path = Path(CREMA_D_VIDEO_DIR) / f"{basename}.mp4"
    if mp4_path.exists():
         return str(mp4_path)
    print(f"Warning: Video file not found for CREMA-D basename: {basename}")
    return None

print("Starting combined manifest generation...")

# 1. Load existing RAVDESS manifest
print(f"Loading RAVDESS manifest: {RAVDESS_MANIFEST_PATH}")
try:
    ravdess_df = pd.read_csv(RAVDESS_MANIFEST_PATH)
    print(f"Loaded {len(ravdess_df)} RAVDESS entries.")
    # Ensure columns match expected output: path, dataset, label, split
    if 'dataset' not in ravdess_df.columns:
        ravdess_df['dataset'] = 'ravdess' # Add dataset column if missing
    ravdess_df = ravdess_df[['path', 'dataset', 'label', 'split']] # Select/reorder columns
    # Filter for common emotions
    ravdess_df = ravdess_df[ravdess_df['label'].isin(EMOTION_LABELS)].copy()
    print(f"Filtered to {len(ravdess_df)} RAVDESS entries with common emotions.")

except FileNotFoundError:
    print(f"Error: RAVDESS manifest not found at {RAVDESS_MANIFEST_PATH}")
    exit(1)
except Exception as e:
    print(f"Error loading RAVDESS manifest: {e}")
    exit(1)


# 2. Load HuBERT splits and extract CREMA-D info
crema_data = []
hubert_splits_path = Path(HUBERT_SPLITS_DIR)
split_files = list(hubert_splits_path.glob("*.csv")) # e.g., train.csv, val.csv, test.csv

if not split_files:
    print(f"Error: No CSV files found in HuBERT splits directory: {HUBERT_SPLITS_DIR}")
    exit(1)

print(f"Loading HuBERT splits from: {HUBERT_SPLITS_DIR}")
for split_file in split_files:
    split_name = split_file.stem # 'train', 'val', or 'test'
    # Handle potential non-standard split names like 'crema_d_train'
    if split_name.endswith('_train'):
        split_name_std = 'train'
    elif split_name.endswith('_val'):
        split_name_std = 'val'
    elif split_name.endswith('_test'):
         split_name_std = 'test'
    else:
         split_name_std = split_name # Use as is if it doesn't match pattern

    print(f"Processing split file: {split_file.name} (interpreting as split: {split_name_std})")
    try:
        hubert_df = pd.read_csv(split_file)
        # Filter for CREMA-D entries and common emotions
        crema_df = hubert_df[
            (hubert_df['dataset'] == 'crema_d') &
            (hubert_df['emotion'].isin(EMOTION_LABELS))
        ].copy()
        print(f"Found {len(crema_df)} CREMA-D entries in {split_file.name} with common emotions.")

        for _, row in crema_df.iterrows():
            audio_path = Path(row['path'])
            basename = audio_path.stem # e.g., 1078_IEO_SAD_HI
            video_path = get_crema_video_path(basename)
            if video_path:
                crema_data.append({
                    'path': video_path,
                    'dataset': 'crema_d',
                    'label': row['emotion'],
                    'split': split_name_std # Use standardized split name
                })

    except FileNotFoundError:
        print(f"Warning: Split file not found: {split_file}")
    except Exception as e:
        print(f"Error processing split file {split_file}: {e}")

crema_combined_df = pd.DataFrame(crema_data)
# Deduplicate based on path in case a file appeared in multiple input splits
crema_combined_df = crema_combined_df.drop_duplicates(subset=['path'], keep='first')
print(f"Collected {len(crema_combined_df)} unique CREMA-D video entries across splits.")

# 3. Combine RAVDESS and CREMA-D DataFrames
print("Combining RAVDESS and CREMA-D data...")
combined_df = pd.concat([ravdess_df, crema_combined_df], ignore_index=True)
# Final deduplication just in case
combined_df = combined_df.drop_duplicates(subset=['path'], keep='first')
print(f"Total unique combined entries: {len(combined_df)}")

# 4. Write out ONE new manifest file containing ALL splits
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# Define the single output filename
output_filename = output_path / f"{OUTPUT_PREFIX}_all.csv" # e.g., combined_manifest_all.csv

# Ensure the necessary columns are present (path, dataset, label, split)
# The train_slowfast_emotion.py script uses these columns internally via the VideoDataset class
output_df = combined_df[['path', 'dataset', 'label', 'split']]

print(f"Writing {len(output_df)} total entries (all splits) to {output_filename}")
# Write with header as the VideoDataset class expects it
output_df.to_csv(output_filename, index=False, header=True)

print("Combined manifest generation complete.")
