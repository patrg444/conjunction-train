#!/usr/bin/env python3
"""
Quick test to verify we can load the downloaded CNN audio features.
"""

import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from scripts.precomputed_cnn_audio_generator import PrecomputedCnnAudioGenerator

print("CNN Audio Feature Test")

# Define feature directories
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed"
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"

def load_data_paths_and_labels_audio_only(cnn_audio_dir_ravdess, cnn_audio_dir_cremad):
    """Finds precomputed CNN audio feature files and extracts labels."""
    # Find precomputed CNN audio feature files
    cnn_audio_files = glob.glob(os.path.join(cnn_audio_dir_ravdess, "*", "*.npy")) + \
                      glob.glob(os.path.join(cnn_audio_dir_cremad, "*.npy"))

    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None

        # Extract label
        try:
            if "Actor_" in cnn_audio_file: # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
            else: # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None # Ensure label is None on error

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found. Ensure CNN audio preprocessing ran and paths are correct.")

    return valid_cnn_audio_files, np.array(labels)

# Test loading files and parsing labels
try:
    cnn_audio_files, all_labels = load_data_paths_and_labels_audio_only(
        RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR
    )
    
    # Print label distribution
    label_counts = [np.sum(all_labels[:, i]) for i in range(all_labels.shape[1])]
    print("\nEmotion class distribution:")
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    for i, count in enumerate(label_counts):
        print(f"  {emotion_names[i]}: {int(count)} samples")

    # Test loading a few feature files
    print("\nLoading sample features to check shapes:")
    for i in range(min(5, len(cnn_audio_files))):
        features = np.load(cnn_audio_files[i])
        print(f"  {os.path.basename(cnn_audio_files[i])}: Shape {features.shape}")
    
    # Test the data generator
    print("\nTesting data generator:")
    batch_size = 8
    
    # Create a small subset for testing
    subset_size = min(20, len(cnn_audio_files))
    test_cnn_audio_files = cnn_audio_files[:subset_size]
    test_labels = all_labels[:subset_size]
    
    # Initialize the generator
    test_generator = PrecomputedCnnAudioGenerator(
        test_cnn_audio_files, test_labels, batch_size=batch_size, shuffle=True
    )
    
    print(f"Audio feature dimension from generator: {test_generator.cnn_audio_dim}")
    print(f"Number of batches in generator: {len(test_generator)}")
    
    # Get a batch
    print("\nFetching a batch from generator:")
    audio_batch, label_batch = test_generator[0]
    print(f"  Audio batch shape: {audio_batch.shape}")
    print(f"  Label batch shape: {label_batch.shape}")
    
    # Show a sample label
    sample_idx = 0
    label_idx = np.argmax(label_batch[sample_idx])
    print(f"\nSample label: {emotion_names[label_idx]} (Index {label_idx})")
    print(f"One-hot encoded: {label_batch[sample_idx]}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
