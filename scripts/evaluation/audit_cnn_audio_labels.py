#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audits the labels and class balance for the precomputed CNN audio features.
"""

import os
import glob
import numpy as np
from collections import Counter
from tqdm import tqdm

# --- Copied from train_audio_only_cnn_lstm_v2.py ---
def load_data_paths_and_labels_audio_only(cnn_audio_dir_ravdess, cnn_audio_dir_cremad):
    """Finds precomputed CNN audio feature files and extracts labels."""
    # Find precomputed CNN audio feature files
    cnn_audio_files = glob.glob(os.path.join(cnn_audio_dir_ravdess, "Actor_*", "*.npy")) + \
                      glob.glob(os.path.join(cnn_audio_dir_cremad, "*.npy"))

    labels = []
    # Reverse map for printing labels
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    index_to_emotion = {v: k for k, v in emotion_map.items()}

    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []
    parsed_labels_list = [] # Store parsed labels for counting

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None
        emotion_code_str = "N/A" # For printing

        # Extract label
        try:
            if "Actor_" in cnn_audio_file: # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label_index = emotion_map[emotion_code]
                    label = np.zeros(len(emotion_map))
                    label[label_index] = 1
                    emotion_code_str = emotion_code
            else: # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label_index = emotion_map[emotion_code]
                    label = np.zeros(len(emotion_map))
                    label[label_index] = 1
                    emotion_code_str = emotion_code
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None # Ensure label is None on error

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
            parsed_labels_list.append(emotion_code_str) # Add the string label
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found. Ensure CNN audio preprocessing ran and paths are correct.")

    return valid_cnn_audio_files, np.array(labels), parsed_labels_list, index_to_emotion

# --- Main Audit Logic ---
if __name__ == '__main__':
    print("Starting CNN Audio Label Audit...")

    # Define feature directories (POINT TO FIXED CNN AUDIO FEATURES)
    RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed" # Use FIXED features
    CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed" # Use FIXED features

    try:
        cnn_audio_files, all_labels_one_hot, parsed_labels_str, index_to_emotion = load_data_paths_and_labels_audio_only(
            RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Class Balance ---
    print("\n--- Class Balance ---")
    class_counts = Counter(parsed_labels_str)
    total_samples = len(parsed_labels_str)
    print(f"Total valid samples: {total_samples}")
    for emotion, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"- {emotion}: {count} samples ({percentage:.2f}%)")

    # --- Label Parsing Verification ---
    print("\n--- Label Parsing Verification (First 15 samples) ---")
    print(f"{'Filename':<60} {'Parsed Label':<15} {'One-Hot Vector'}")
    print("-" * 90)
    for i in range(min(15, len(cnn_audio_files))):
        filename = os.path.basename(cnn_audio_files[i])
        label_str = parsed_labels_str[i]
        one_hot_vector = all_labels_one_hot[i]
        print(f"{filename:<60} {label_str:<15} {one_hot_vector}")

    print("\nAudit complete.")
