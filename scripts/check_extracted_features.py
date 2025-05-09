#!/usr/bin/env python3
"""
Simple script to check if a random .npz file from the processed directories
contains both video and audio features with the correct dimensions.
"""

import os
import sys
import numpy as np
import glob
import random

def check_npz_file(file_path):
    """Check if the given npz file has both video and audio features with correct dimensions."""
    print(f"Checking file: {file_path}")
    
    try:
        data = np.load(file_path)
        
        # Check for video features
        if 'video_features' in data:
            video_features = data['video_features']
            video_dim = video_features.shape[1]
            print(f"✅ Video features: {video_features.shape[0]} frames with {video_dim} dimensions")
            
            # Check if dimensions match expected
            if video_dim == 4096:
                print("✅ Video feature dimensions are correct (4096)")
            else:
                print(f"❌ Video feature dimensions are incorrect: {video_dim} (expected 4096)")
        else:
            print("❌ No video features found in the file")
            
        # Check for audio features
        if 'audio_features' in data:
            audio_features = data['audio_features']
            audio_dim = audio_features.shape[1]
            print(f"✅ Audio features: {audio_features.shape[0]} frames with {audio_dim} dimensions")
            
            # Check if dimensions match expected
            if audio_dim >= 87:
                print(f"✅ Audio feature dimensions are correct ({audio_dim} ≥ 87)")
            else:
                print(f"❌ Audio feature dimensions are incorrect: {audio_dim} (expected ≥87)")
        else:
            print("❌ No audio features found in the file")
            
        # Check for emotion label
        if 'emotion_label' in data:
            emotion = data['emotion_label']
            print(f"✅ Emotion label: {emotion}")
        else:
            print("❌ No emotion label found in the file")
            
        return 'video_features' in data and 'audio_features' in data and data['video_features'].shape[1] == 4096 and data['audio_features'].shape[1] >= 87
            
    except Exception as e:
        print(f"❌ Error checking file: {str(e)}")
        return False

def main():
    """Check random files from the processed directories."""
    ravdess_dir = "processed_ravdess_fixed"
    crema_d_dir = "processed_crema_d_fixed"
    
    ravdess_files = glob.glob(os.path.join(ravdess_dir, "*.npz"))
    crema_d_files = glob.glob(os.path.join(crema_d_dir, "*.npz"))
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(crema_d_files)} CREMA-D files")
    
    all_ok = True
    
    if ravdess_files:
        print("\n=== Checking RAVDESS ===")
        ravdess_sample = random.choice(ravdess_files)
        if not check_npz_file(ravdess_sample):
            all_ok = False
    
    if crema_d_files:
        print("\n=== Checking CREMA-D ===")
        crema_d_sample = random.choice(crema_d_files)
        if not check_npz_file(crema_d_sample):
            all_ok = False
    
    if all_ok:
        print("\n✅ SUCCESS: Both datasets have files with correct video and audio features!")
        return 0
    else:
        print("\n❌ FAILURE: Issues found with at least one dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main())
