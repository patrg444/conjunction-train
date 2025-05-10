#!/usr/bin/env python3
import os
import glob
import numpy as np
from collections import Counter
import json

# Paths to check
RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"

# Maps from the script
emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

# Counters to track skipped reasons
ravdess_skipped_reasons = Counter()
cremad_skipped_reasons = Counter()
valid_counts = {'ravdess': 0, 'cremad': 0}

# Get the file lists
ravdess_files = glob.glob(os.path.join(RAVDESS_FACENET_DIR, "Actor_*", "*.npz"))
cremad_files = glob.glob(os.path.join(CREMA_D_FACENET_DIR, "*.npz"))

print(f"Found {len(ravdess_files)} RAVDESS files and {len(cremad_files)} CREMA-D files")

# Emotion code tracking
ravdess_emotion_codes = Counter()
cremad_emotion_codes = Counter()

# Process RAVDESS files
print("\nProcessing RAVDESS files...")
for i, file_path in enumerate(ravdess_files):
    # Only print the first 5
    if i < 5:
        print(f"Processing file: {file_path}")
    elif i == 5:
        print("...")
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if i < 5:
        print(f"  Base name: {base_name}")
    
    # Extract emotion code
    try:
        parts = base_name.split('-')
        if i < 5:
            print(f"  Parts: {parts}")
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            ravdess_emotion_codes[emotion_code] += 1
            
            # Check if in mapping
            mapped_emotion = ravdess_emotion_map.get(emotion_code, None)
            if mapped_emotion is None:
                ravdess_skipped_reasons[f"Unknown emotion code: {emotion_code}"] += 1
                if i < 5:
                    print(f"  SKIPPED: Unknown emotion code: {emotion_code}")
                continue
            
            # Check if mapped emotion is in final mapping
            if mapped_emotion not in emotion_map:
                ravdess_skipped_reasons[f"Mapped emotion not in emotion_map: {mapped_emotion}"] += 1
                if i < 5:
                    print(f"  SKIPPED: Mapped emotion not in emotion_map: {mapped_emotion}")
                continue
            
            # Check if file has video_features
            try:
                with np.load(file_path) as data:
                    if 'video_features' in data and data['video_features'].shape[0] > 0:
                        valid_counts['ravdess'] += 1
                        if i < 5:
                            print(f"  VALID: Has 'video_features' with shape {data['video_features'].shape}")
                    else:
                        keys = list(data.keys())
                        ravdess_skipped_reasons["Missing video_features key"] += 1
                        if i < 5:
                            print(f"  SKIPPED: Missing 'video_features' key. Available keys: {keys}")
                        continue
            except Exception as e:
                ravdess_skipped_reasons[f"Error loading npz: {str(e)}"] += 1
                if i < 5:
                    print(f"  SKIPPED: Error loading npz: {str(e)}")
                continue
                
        else:
            ravdess_skipped_reasons["Filename parts < 3"] += 1
            if i < 5:
                print(f"  SKIPPED: Filename doesn't have enough parts")
            continue
            
    except Exception as e:
        ravdess_skipped_reasons[f"Exception: {str(e)}"] += 1
        if i < 5:
            print(f"  SKIPPED: Exception: {str(e)}")
        continue

# Process CREMA-D files
print("\nProcessing CREMA-D files...")
for i, file_path in enumerate(cremad_files):
    # Only print the first 5
    if i < 5:
        print(f"Processing file: {file_path}")
    elif i == 5:
        print("...")
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if i < 5:
        print(f"  Base name: {base_name}")
    
    # Extract emotion code
    try:
        parts = base_name.split('_')
        if i < 5:
            print(f"  Parts: {parts}")
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            cremad_emotion_codes[emotion_code] += 1
            
            # Check if in mapping
            if emotion_code not in emotion_map:
                cremad_skipped_reasons[f"Unknown emotion code: {emotion_code}"] += 1
                if i < 5:
                    print(f"  SKIPPED: Unknown emotion code: {emotion_code}")
                continue
            
            # Check if file has video_features
            try:
                with np.load(file_path) as data:
                    if 'video_features' in data and data['video_features'].shape[0] > 0:
                        valid_counts['cremad'] += 1
                        if i < 5:
                            print(f"  VALID: Has 'video_features' with shape {data['video_features'].shape}")
                    else:
                        keys = list(data.keys())
                        cremad_skipped_reasons["Missing video_features key"] += 1
                        if i < 5:
                            print(f"  SKIPPED: Missing 'video_features' key. Available keys: {keys}")
                        continue
            except Exception as e:
                cremad_skipped_reasons[f"Error loading npz: {str(e)}"] += 1
                if i < 5:
                    print(f"  SKIPPED: Error loading npz: {str(e)}")
                continue
                
        else:
            cremad_skipped_reasons["Filename parts < 3"] += 1
            if i < 5:
                print(f"  SKIPPED: Filename doesn't have enough parts")
            continue
            
    except Exception as e:
        cremad_skipped_reasons[f"Exception: {str(e)}"] += 1
        if i < 5:
            print(f"  SKIPPED: Exception: {str(e)}")
        continue

# Report on RAVDESS emotion codes
print("\n--- RAVDESS Emotion Codes Statistics ---")
print(f"Total unique emotion codes: {len(ravdess_emotion_codes)}")
print("Emotion code counts:")
for code, count in ravdess_emotion_codes.most_common():
    in_map = code in ravdess_emotion_map
    mapped_to = ravdess_emotion_map.get(code, "N/A")
    print(f"  Code '{code}': {count} files, in map: {in_map}, maps to: {mapped_to}")

# Report on CREMA-D emotion codes
print("\n--- CREMA-D Emotion Codes Statistics ---")
print(f"Total unique emotion codes: {len(cremad_emotion_codes)}")
print("Emotion code counts:")
for code, count in cremad_emotion_codes.most_common():
    in_map = code in emotion_map
    print(f"  Code '{code}': {count} files, in emotion_map: {in_map}")

# Overall statistics
print("\n--- Overall Statistics ---")
print(f"Total RAVDESS files processed: {len(ravdess_files)}")
print(f"Valid RAVDESS files: {valid_counts['ravdess']}")
print(f"Total CREMA-D files processed: {len(cremad_files)}")
print(f"Valid CREMA-D files: {valid_counts['cremad']}")
print(f"Total valid files: {valid_counts['ravdess'] + valid_counts['cremad']}")

# Reasons for skipping
print("\n--- RAVDESS Skip Reasons ---")
for reason, count in ravdess_skipped_reasons.most_common():
    print(f"  {reason}: {count} files")

print("\n--- CREMA-D Skip Reasons ---")
for reason, count in cremad_skipped_reasons.most_common():
    print(f"  {reason}: {count} files")

# Potential fixes
print("\n--- Potential Fixes ---")
missing_ravdess_codes = set(ravdess_emotion_codes.keys()) - set(ravdess_emotion_map.keys())
if missing_ravdess_codes:
    print(f"Add these RAVDESS emotion codes to ravdess_emotion_map: {', '.join(missing_ravdess_codes)}")

missing_cremad_codes = set(cremad_emotion_codes.keys()) - set(emotion_map.keys())
if missing_cremad_codes:
    print(f"Add these CREMA-D emotion codes to emotion_map: {', '.join(missing_cremad_codes)}")
