#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path
import warnings

# --- Configuration (Paths on EC2 Instance) ---
RAVDESS_MANIFEST_PATH = "/home/ubuntu/datasets/video_manifest.csv"
HUBERT_SPLITS_DIR = "/home/ubuntu/conjunction-train/splits"
CREMA_D_VIDEO_DIR = "/home/ubuntu/datasets/crema_d_videos"
# --- End Configuration ---

# Suppress specific pandas warnings if necessary
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_crema_video_path(basename):
    """Constructs the expected path for a CREMA-D video file."""
    flv_path = Path(CREMA_D_VIDEO_DIR) / f"{basename}.flv"
    if flv_path.exists():
        return str(flv_path.resolve()) # Use resolved absolute path
    mp4_path = Path(CREMA_D_VIDEO_DIR) / f"{basename}.mp4"
    if mp4_path.exists():
         return str(mp4_path.resolve())
    # print(f"Debug: Video file not found for CREMA-D basename: {basename}")
    return None

print("--- Starting Split Overlap Check ---")

train_paths = set()
val_paths = set()

# 1. Process RAVDESS Manifest
print(f"Processing RAVDESS: {RAVDESS_MANIFEST_PATH}")
try:
    ravdess_df = pd.read_csv(RAVDESS_MANIFEST_PATH)
    # Ensure columns exist
    if not all(col in ravdess_df.columns for col in ['path', 'split']):
         print(f"Error: Missing 'path' or 'split' column in {RAVDESS_MANIFEST_PATH}")
         exit(1)

    ravdess_train = ravdess_df[ravdess_df['split'] == 'train']['path'].apply(lambda p: str(Path(p).resolve())).tolist()
    ravdess_val = ravdess_df[ravdess_df['split'] == 'val']['path'].apply(lambda p: str(Path(p).resolve())).tolist()
    train_paths.update(ravdess_train)
    val_paths.update(ravdess_val)
    print(f"RAVDESS: Found {len(ravdess_train)} train paths, {len(ravdess_val)} val paths.")
except Exception as e:
    print(f"Error processing RAVDESS manifest: {e}")
    exit(1)

# 2. Process HuBERT Splits for CREMA-D
print(f"Processing CREMA-D splits from: {HUBERT_SPLITS_DIR}")
hubert_splits_path = Path(HUBERT_SPLITS_DIR)
split_files_info = {
    'train': ['train.csv', 'crema_d_train.csv'],
    'val': ['val.csv', 'crema_d_val.csv']
}

crema_train_count = 0
crema_val_count = 0

for split_type, filenames in split_files_info.items():
    target_set = train_paths if split_type == 'train' else val_paths
    current_count = 0
    for filename in filenames:
        split_file = hubert_splits_path / filename
        if not split_file.exists():
            print(f"Warning: Split file not found, skipping: {split_file}")
            continue

        print(f"  Reading {filename} for '{split_type}' split...")
        try:
            hubert_df = pd.read_csv(split_file)
            # Ensure columns exist
            if not all(col in hubert_df.columns for col in ['path', 'dataset']):
                 print(f"Error: Missing 'path' or 'dataset' column in {split_file}")
                 continue # Skip this file

            crema_df = hubert_df[hubert_df['dataset'] == 'crema_d'].copy()

            for _, row in crema_df.iterrows():
                audio_path_str = row.get('path', None)
                if not audio_path_str:
                    # print(f"Warning: Missing 'path' in row: {row}")
                    continue
                audio_path = Path(audio_path_str)
                basename = audio_path.stem
                video_path = get_crema_video_path(basename)
                if video_path:
                    target_set.add(video_path) # Add resolved path
                    current_count += 1

        except Exception as e:
            print(f"Error processing split file {split_file}: {e}")

    if split_type == 'train':
        crema_train_count = len(target_set) - len(ravdess_train) # Count only newly added CREMA paths
    else:
        crema_val_count = len(target_set) - len(ravdess_val) # Count only newly added CREMA paths

print(f"CREMA-D: Found {crema_train_count} unique train video paths.")
print(f"CREMA-D: Found {crema_val_count} unique val video paths.")

# 3. Check for Overlap
print("\n--- Checking for Overlap ---")
overlap = train_paths.intersection(val_paths)
num_overlap = len(overlap)

if num_overlap > 0:
    print(f"ERROR: Found {num_overlap} overlapping video paths between train and val sets!")
    # print("Overlapping paths:")
    # for i, path in enumerate(overlap):
    #     print(f"  {i+1}. {path}")
else:
    print("Success: No overlap found between training and validation sets.")

print("--- Split Overlap Check Complete ---")
