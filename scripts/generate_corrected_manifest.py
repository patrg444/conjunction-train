#!/usr/bin/env python3
"""
Generates a corrected master manifest CSV by deriving labels from source info.

- Reads individual split CSVs (train, val, test, crema_d_train, etc.).
- For RAVDESS files (identified by filename pattern), derives the 6-class label
  from the filename's emotion code (mapping calm->neutral, omitting surprised).
- For other files (assumed CREMA-D), trusts the label provided in its
  original split CSV.
- Combines all entries into a single master manifest CSV with 'path', 'label', 'split'.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Target 6 classes: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

# Mapping from RAVDESS filename emotion code (3rd part) to the desired 6-class index
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_CODE_TO_TARGET_IDX = {
    '01': LABEL_TO_IDX['neutral'],  # neutral -> neutral (4)
    '02': LABEL_TO_IDX['neutral'],  # calm -> neutral (4)
    '03': LABEL_TO_IDX['happy'],    # happy -> happy (3)
    '04': LABEL_TO_IDX['sad'],      # sad -> sad (5)
    '05': LABEL_TO_IDX['angry'],    # angry -> angry (0)
    '06': LABEL_TO_IDX['fearful'],  # fearful -> fear (2)
    '07': LABEL_TO_IDX['disgust'],  # disgust -> disgust (1)
    '08': None,                     # surprised -> omit
}

# Mapping for CREMA-D labels (assuming these abbreviations are in the CSVs)
CREMA_D_LABEL_TO_TARGET_IDX = {
    'ANG': LABEL_TO_IDX['angry'],    # ANG -> angry (0)
    'DIS': LABEL_TO_IDX['disgust'],  # DIS -> disgust (1)
    'FEA': LABEL_TO_IDX['fearful'],  # FEA -> fearful (2)
    'HAP': LABEL_TO_IDX['happy'],    # HAP -> happy (3)
    'NEU': LABEL_TO_IDX['neutral'],  # NEU -> neutral (4)
    'SAD': LABEL_TO_IDX['sad'],      # SAD -> sad (5)
    # Add lowercase versions for robustness
    'ang': LABEL_TO_IDX['angry'],
    'dis': LABEL_TO_IDX['disgust'],
    'fea': LABEL_TO_IDX['fearful'],
    'hap': LABEL_TO_IDX['happy'],
    'neu': LABEL_TO_IDX['neutral'],
    'sad': LABEL_TO_IDX['sad'],
}

# Consolidated map for all known non-RAVDESS labels (case-insensitive keys)
NON_RAVDESS_LABEL_MAP = {
    # Full names (lowercase)
    'angry': LABEL_TO_IDX['angry'],
    'disgust': LABEL_TO_IDX['disgust'],
    'fearful': LABEL_TO_IDX['fearful'],
    'happy': LABEL_TO_IDX['happy'],
    'neutral': LABEL_TO_IDX['neutral'],
    'sad': LABEL_TO_IDX['sad'],
    # CREMA-D abbreviations (lowercase)
    'ang': LABEL_TO_IDX['angry'],
    'dis': LABEL_TO_IDX['disgust'],
    'fea': LABEL_TO_IDX['fearful'],
    'hap': LABEL_TO_IDX['happy'],
    'neu': LABEL_TO_IDX['neutral'],
    'sad': LABEL_TO_IDX['sad'],
     # Observed variations (lowercase)
    'fear': LABEL_TO_IDX['fearful'],
}


def get_ravdess_emotion_code(filename):
    """Extracts the 3rd part (emotion code) from a RAVDESS filename."""
    parts = filename.stem.split('-') # Use stem to ignore extension
    if len(parts) == 7:
        return parts[2]
    return None

def is_ravdess_audio_path(filepath_str):
    """
    Checks if a filepath string likely represents a RAVDESS *audio* file
    based on its path components and filename structure.
    Assumes paths like 'ravdess/AudioWAV/Actor_XX/filename.wav'.
    """
    p = Path(filepath_str)
    parts = p.parts
    filename_parts = p.stem.split('-')
    # Check path structure and filename structure
    return (len(parts) >= 4 and
            parts[-4].lower() == 'ravdess' and
            parts[-3].lower() == 'audiowav' and
            parts[-2].lower().startswith('actor_') and
            len(filename_parts) == 7 and
            all(part.isdigit() for part in filename_parts) and
            p.suffix.lower() == '.wav')

def construct_video_path(audio_path_str, is_ravdess):
    """Constructs the absolute EC2 video path from the audio path."""
    audio_p = Path(audio_path_str)
    filename_stem = audio_p.stem
    # Determine video extension based on dataset type
    video_extension = ".mp4" if is_ravdess else ".flv"
    video_filename = f"{filename_stem}{video_extension}"

    if is_ravdess:
        # Assumes original path is like 'ravdess/AudioWAV/Actor_XX/...'
        actor_dir = audio_p.parts[-2] # e.g., 'Actor_01'
        # EC2 Base: /home/ubuntu/datasets/ravdess_videos/
        return f"/home/ubuntu/datasets/ravdess_videos/{actor_dir}/{video_filename}"
    else:
        # Assumes original path is like 'crema_d/AudioWAV/...'
        # EC2 Base: /home/ubuntu/datasets/crema_d_videos/
        return f"/home/ubuntu/datasets/crema_d_videos/{video_filename}"

def main():
    parser = argparse.ArgumentParser(description="Generate a corrected manifest from split CSVs and filename conventions.")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory containing the original train/val/test/crema_d CSV files.")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to save the generated corrected master manifest CSV.")
    parser.add_argument("--path_column", type=str, default="path", help="Column name in split CSVs containing the file paths.")
    parser.add_argument("--label_column", type=str, default="emotion", help="Column name in split CSVs containing the original labels (used for non-RAVDESS).")

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_manifest_path = Path(args.output_manifest)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    all_manifest_data = []
    processed_files = 0
    omitted_surprised = 0
    ravdess_labelled = 0
    other_labelled = 0
    label_errors = 0

    # Find all relevant CSV files in the splits directory
    split_files = list(splits_dir.glob('*.csv'))
    if not split_files:
        print(f"Error: No CSV files found in {args.splits_dir}")
        return

    print(f"Found {len(split_files)} split CSV files to process in {args.splits_dir}")

    for csv_path in tqdm(split_files, desc="Processing split CSVs"):
        split_name = csv_path.stem # e.g., 'train', 'val', 'crema_d_train'
        print(f"\nProcessing: {csv_path.name} (Split: {split_name})")
        try:
            df = pd.read_csv(csv_path)
            if args.path_column not in df.columns or args.label_column not in df.columns:
                 print(f"  Warning: Skipping {csv_path.name}. Missing required columns '{args.path_column}' or '{args.label_column}'.")
                 continue

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Scanning {split_name}", unit="sample", leave=False):
                processed_files += 1
                filepath = row[args.path_column]
                original_label = row[args.label_column] # Label from the split CSV
                filename_str = Path(filepath).name # Get filename as string
                original_audio_path = filepath # Keep original path for checks/label derivation
                filename_path = Path(original_audio_path) # Path object for filename checks
                corrected_label_idx = None

                # --- Determine Dataset Type based on ORIGINAL audio path ---
                is_ravdess = is_ravdess_audio_path(original_audio_path)

                # --- Construct Absolute EC2 Video Path ---
                # This assumes a corresponding video file exists for every audio file listed
                video_filepath = construct_video_path(original_audio_path, is_ravdess)

                # --- Label Derivation (using original audio filename logic) ---
                if is_ravdess:
                    # Derive label from RAVDESS audio filename
                    emotion_code = get_ravdess_emotion_code(filename_path) # Use original audio filename path
                    if emotion_code:
                        target_idx = RAVDESS_CODE_TO_TARGET_IDX.get(emotion_code)
                        if target_idx is not None:
                            corrected_label_idx = target_idx
                            ravdess_labelled += 1
                        elif emotion_code == '08': # Surprised
                            omitted_surprised += 1
                            continue # Skip this file
                        else:
                            print(f"  Warning: Unknown RAVDESS code '{emotion_code}' in {filename}. Skipping.")
                            label_errors += 1
                            continue
                    else:
                        print(f"  Warning: Could not parse RAVDESS code from {filename_str}. Skipping.")
                        label_errors += 1
                        continue
                else:
                    # Assume non-RAVDESS (e.g., CREMA-D)
                    # Assume non-RAVDESS (e.g., CREMA-D)
                    label_to_check = str(original_label).strip().lower() # Convert to lower string and strip whitespace

                    # --- DEBUG PRINTS ---
                    if 'fea' in filename_str.lower(): # Print only for potential 'fear' files to reduce noise
                        print(f"\nDEBUG: Filename: {filename_str}")
                        print(f"DEBUG: Original Label: '{original_label}' (Type: {type(original_label)})")
                        print(f"DEBUG: Label to Check: '{label_to_check}' (Type: {type(label_to_check)})")
                        print(f"DEBUG: Checking if '{label_to_check}' is in NON_RAVDESS_LABEL_MAP keys: {list(NON_RAVDESS_LABEL_MAP.keys())}")
                    # --- END DEBUG PRINTS ---

                    if label_to_check in NON_RAVDESS_LABEL_MAP:
                        # Check the consolidated map using the lowercase label
                        corrected_label_idx = NON_RAVDESS_LABEL_MAP[label_to_check]
                        other_labelled += 1
                    else:
                        # Fallback: Try converting original label (before lowercasing) to int
                        try:
                            original_label_idx = int(original_label)
                            if 0 <= original_label_idx < len(EMOTION_LABELS):
                                corrected_label_idx = original_label_idx
                                other_labelled += 1
                            else:
                                print(f"  Warning: Invalid original label index '{original_label}' for non-RAVDESS file {filename_str}. Skipping.")
                                label_errors += 1
                                continue
                        except (ValueError, TypeError):
                             # If it's not in the map and not a valid integer index, skip it
                             # Check if it's NaN before printing warning
                             if pd.isna(original_label):
                                 # Allow missing labels for non-RAVDESS if needed, or handle differently
                                 # For now, let's skip them if the label is truly missing/NaN
                                 # print(f"  Warning: Missing label for non-RAVDESS file {filename_str}. Skipping.")
                                 label_errors += 1 # Count as error for now
                                 continue
                             else:
                                 print(f"  Warning: Unrecognized or invalid label '{original_label}' (processed as '{label_to_check}') for non-RAVDESS file {filename_str}. Skipping.")
                                 label_errors += 1
                                 continue

                if corrected_label_idx is not None:
                    # Use the label string corresponding to the index
                    corrected_label_str = EMOTION_LABELS[corrected_label_idx]
                    all_manifest_data.append({
                        'path': video_filepath, # <<< STORE THE CONSTRUCTED ABSOLUTE VIDEO FILE PATH
                        'label': corrected_label_str, # Store the string label
                        'split': split_name # Add split info from the original CSV
                    })

        except FileNotFoundError:
            print(f"  Warning: File not found: {csv_path}. Skipping.")
        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")

    # Create DataFrame and save
    if not all_manifest_data:
        print("Error: No valid data collected to write to manifest.")
        return

    final_df = pd.DataFrame(all_manifest_data)
    final_count = len(final_df)
    # --- NO DEDUPLICATION: Maintain original order from concatenated CSVs ---
    # initial_count = len(final_df)
    # final_df = final_df.drop_duplicates(subset='path', keep='first')
    # final_count = len(final_df)
    # duplicates_removed = initial_count - final_count
    # if duplicates_removed > 0:
    #     print(f"\nRemoved {duplicates_removed} duplicate entries based on file path.")

    # Ensure consistent order (columns only, row order preserved)
    final_df = final_df[['path', 'label', 'split']]

    print(f"\nWriting corrected manifest with {final_count} entries (preserving original order) to: {args.output_manifest}")
    try:
        final_df.to_csv(args.output_manifest, index=False)
        print("Manifest generation successful.")
    except Exception as e:
        print(f"Error writing final manifest: {e}")

    print("\n--- Manifest Generation Summary ---")
    print(f"Total files scanned across all splits: {processed_files}")
    print(f"RAVDESS files labelled from filename:  {ravdess_labelled}")
    print(f"Other files labelled from split CSV:   {other_labelled}")
    print(f"RAVDESS 'surprised' files omitted:     {omitted_surprised}")
    print(f"Files skipped due to label errors:     {label_errors}")
    print(f"Total unique entries in corrected manifest: {len(final_df)}")

if __name__ == "__main__":
    main()
