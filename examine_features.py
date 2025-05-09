#!/usr/bin/env python3
"""Examine the contents of a FaceNet feature file."""

import numpy as np
import sys
import glob

def examine_file(file_path):
    print(f"Examining: {file_path}")
    
    # Load the data
    data = np.load(file_path, allow_pickle=True)
    
    # Print the keys
    print(f"Keys in the file: {list(data.keys())}")
    
    # Print video features info if available
    if 'video_features' in data:
        video_features = data['video_features']
        print(f"\nVideo features shape: {video_features.shape}")
        print(f"Video features - first 5 elements of first frame:")
        print(video_features[0][:5])
        print(f"Non-zero percentage: {np.count_nonzero(video_features)/video_features.size*100:.2f}%")
        print(f"Min: {np.min(video_features):.6f}")
        print(f"Max: {np.max(video_features):.6f}")
        print(f"Mean: {np.mean(video_features):.6f}")
        print(f"Std: {np.std(video_features):.6f}")
    
    # Print audio features info if available
    if 'audio_features' in data:
        audio_features = data['audio_features']
        print(f"\nAudio features shape: {audio_features.shape}")
        print(f"Audio features - first 5 elements of first frame:")
        print(audio_features[0][:5])
        print(f"Non-zero percentage: {np.count_nonzero(audio_features)/audio_features.size*100:.2f}%")
        print(f"Min: {np.min(audio_features):.6f}")
        print(f"Max: {np.max(audio_features):.6f}")
        print(f"Mean: {np.mean(audio_features):.6f}")
        print(f"Std: {np.std(audio_features):.6f}")
    
    # Print emotion label if available
    if 'emotion_label' in data:
        print(f"\nEmotion label: {data['emotion_label'].item()}")

if __name__ == "__main__":
    # If a specific file was provided, examine it
    if len(sys.argv) > 1:
        examine_file(sys.argv[1])
    else:
        # Otherwise, try to find a file
        print("No file specified, looking for sample files...")
        
        # First try the standalone FaceNet features file
        facenet_files = glob.glob("*facenet_features.npz")
        if facenet_files:
            examine_file(facenet_files[0])
            exit(0)
        
        # Then try the RAVDESS features
        ravdess_files = glob.glob("ravdess_features/*.npz")
        if ravdess_files:
            examine_file(ravdess_files[0])
            exit(0)
        
        # Then try CREMA-D features
        cremad_files = glob.glob("crema_d_features/*.npz")
        if cremad_files:
            examine_file(cremad_files[0])
            exit(0)
        
        # If nothing found
        print("No feature files found.")
