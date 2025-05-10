#!/usr/bin/env python3
"""
Direct test of the load_data_paths_and_labels_video_only function copied from the training script
"""
import os
import glob
import numpy as np
from tqdm import tqdm

def load_data_paths_and_labels_video_only(facenet_dir_ravdess, facenet_dir_cremad):
    """Finds precomputed Facenet feature files (.npz) and extracts labels."""
    # Find precomputed Facenet feature files (assuming .npz format)
    print(f"RAVDESS glob pattern: {os.path.join(facenet_dir_ravdess, 'Actor_*', '*.npz')}")
    print(f"CREMA-D glob pattern: {os.path.join(facenet_dir_cremad, '*.npz')}")
    
    # Check if directories exist
    print(f"RAVDESS directory exists: {os.path.exists(facenet_dir_ravdess)}")
    print(f"CREMA-D directory exists: {os.path.exists(facenet_dir_cremad)}")
    
    # List directories to debug
    if os.path.exists(facenet_dir_ravdess):
        actor_dirs = glob.glob(os.path.join(facenet_dir_ravdess, "Actor_*"))
        print(f"Found {len(actor_dirs)} Actor directories")
        if actor_dirs:
            for d in actor_dirs[:3]:
                print(f"  {d}")
                npz_files = glob.glob(os.path.join(d, "*.npz"))
                print(f"    Contains {len(npz_files)} NPZ files")
    
    # Perform the actual glob
    ravdess_files = glob.glob(os.path.join(facenet_dir_ravdess, "Actor_*", "*.npz"))
    cremad_files = glob.glob(os.path.join(facenet_dir_cremad, "*.npz"))
    print(f"RAVDESS glob found {len(ravdess_files)} files")
    print(f"CREMA-D glob found {len(cremad_files)} files")
    
    # Combine files as in the training script
    facenet_files = ravdess_files + cremad_files
    print(f"Total combined files: {len(facenet_files)}")
    
    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}

    print(f"Found {len(facenet_files)} precomputed Facenet files. Extracting labels...")
    skipped_label = 0
    valid_facenet_files = []

    for facenet_file in tqdm(facenet_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(facenet_file))[0]
        # Remove potential suffixes like '_facenet_features' if present
        if base_name.endswith('_facenet_features'):
            base_name = base_name[:-len('_facenet_features')]

        label = None

        # Extract label
        try:
            if "Actor_" in facenet_file: # RAVDESS
                parts = base_name.split('-')
                if len(parts) >= 3:
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                    else:
                        print(f"Warning: RAVDESS emotion code '{parts[2]}' not mapped correctly")
                else:
                    print(f"Warning: RAVDESS filename doesn't have enough parts: {base_name}")
            else: # CREMA-D
                parts = base_name.split('_')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                    else:
                        print(f"Warning: CREMA-D emotion code '{emotion_code}' not in mapping")
                else:
                    print(f"Warning: CREMA-D filename doesn't have enough parts: {base_name}")
        except Exception as e:
            print(f"Label parsing error for {facenet_file}: {e}")
            label = None # Ensure label is None on error

        if label is not None:
            # Basic check: Does the file exist and is it non-empty?
            if os.path.exists(facenet_file) and os.path.getsize(facenet_file) > 0:
                # Check if the label is valid (not None)
                if label is not None:
                    # Check to load the npz and see if 'video_features' key exists
                    try:
                        with np.load(facenet_file) as data:
                            if 'video_features' in data and data['video_features'].shape[0] > 0:
                                valid_facenet_files.append(facenet_file)
                                labels.append(label)
                            else:
                                print(f"Warning: Skipping {facenet_file} - 'video_features' key missing or empty.")
                                skipped_label += 1
                    except Exception as load_e:
                        print(f"Warning: Skipping {facenet_file} - Error loading npz: {load_e}")
                        skipped_label += 1
                else:
                    print(f"Warning: Skipping {facenet_file} - Invalid label.")
                    skipped_label += 1
            else:
                print(f"Warning: Skipping {facenet_file} - File does not exist or is empty.")
                skipped_label += 1
        else:
            skipped_label += 1

    print(f"Found {len(valid_facenet_files)} Facenet files with valid labels and features.")
    print(f"Skipped {skipped_label} due to label parsing or feature issues.")

    if not valid_facenet_files:
        raise FileNotFoundError("No Facenet files with valid labels/features found. Ensure Facenet preprocessing ran and paths are correct.")

    return valid_facenet_files, np.array(labels)

if __name__ == "__main__":
    # Define feature directories (POINT TO FACENET FEATURES - ABSOLUTE PATHS)
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet" # Fixed absolute path
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet" # Fixed absolute path
    
    try:
        facenet_files, all_labels = load_data_paths_and_labels_video_only(
            RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
        )
        print(f"Successfully loaded {len(facenet_files)} Facenet files with labels")
        print(f"Labels shape: {all_labels.shape}")
        
        # Count by dataset
        ravdess_count = sum(1 for f in facenet_files if "Actor_" in f)
        cremad_count = len(facenet_files) - ravdess_count
        print(f"RAVDESS files: {ravdess_count}")
        print(f"CREMA-D files: {cremad_count}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
