#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the emotion recognition dataset utilities.
This script verifies that we can correctly:
1. Load the RAVDESS and CREMA-D datasets
2. Analyze their class distribution
3. Create train/val/test splits
"""

import os
import argparse
import matplotlib.pyplot as plt
from emotion_comparison.common.dataset_utils import EmotionDatasetManager, EMOTION_NAMES

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test emotion recognition dataset utilities')
    parser.add_argument('--ravdess_dir', type=str, default='/home/ubuntu/datasets/ravdess_videos', 
                        help='Path to RAVDESS dataset directory')
    parser.add_argument('--cremad_dir', type=str, default='/home/ubuntu/datasets/crema_d_videos', 
                        help='Path to CREMA-D dataset directory')
    parser.add_argument('--output_dir', type=str, default='./processed_data', 
                        help='Directory to save processed data')
    args = parser.parse_args()
    
    print("\n===== Testing Emotion Recognition Dataset Utilities =====\n")
    
    # Initialize dataset manager
    print(f"Initializing dataset manager with:")
    print(f"  RAVDESS directory: {args.ravdess_dir}")
    print(f"  CREMA-D directory: {args.cremad_dir}")
    print(f"  Output directory: {args.output_dir}")
    
    manager = EmotionDatasetManager(
        ravdess_dir=args.ravdess_dir,
        cremad_dir=args.cremad_dir,
        output_dir=args.output_dir
    )
    
    # Test loading datasets
    print("\n----- Testing Dataset Loading -----")
    
    # Check if we need to create data directories
    if not os.path.exists(args.ravdess_dir) or not os.path.exists(args.cremad_dir):
        print(f"Warning: One or both dataset directories don't exist.")
        print(f"  RAVDESS directory exists: {os.path.exists(args.ravdess_dir)}")
        print(f"  CREMA-D directory exists: {os.path.exists(args.cremad_dir)}")
        print("Please check your paths or create the directories.")
        return
    
    # Print file counts
    print(f"RAVDESS files found: {len(manager.ravdess_files)}")
    print(f"CREMA-D files found: {len(manager.cremad_files)}")
    
    if len(manager.ravdess_files) == 0 or len(manager.cremad_files) == 0:
        print("Warning: One or both datasets have no files. Check the directories.")
        return
    
    # Test some file paths
    if len(manager.ravdess_files) > 0:
        print(f"\nExample RAVDESS file: {os.path.basename(manager.ravdess_files[0])}")
        emotion_idx, emotion_name = manager.get_emotion_from_path(manager.ravdess_files[0])
        print(f"  Extracted emotion: {emotion_name} (index {emotion_idx})")
    
    if len(manager.cremad_files) > 0:
        print(f"\nExample CREMA-D file: {os.path.basename(manager.cremad_files[0])}")
        emotion_idx, emotion_name = manager.get_emotion_from_path(manager.cremad_files[0])
        print(f"  Extracted emotion: {emotion_name} (index {emotion_idx})")
    
    # Test class distribution analysis
    print("\n----- Testing Class Distribution Analysis -----")
    ravdess_counts, cremad_counts, combined_counts, class_weights = manager.analyze_class_distribution()
    
    # Visualize the distributions
    manager.visualize_distributions(ravdess_counts, cremad_counts, combined_counts)
    
    # Test dataset splitting
    print("\n----- Testing Dataset Splitting -----")
    train_files, val_files, test_files = manager.create_dataset_splits()
    
    # Print a few examples from each split
    print("\nExample training files:")
    for i in range(min(3, len(train_files))):
        print(f"  {os.path.basename(train_files[i])}")
    
    print("\nExample validation files:")
    for i in range(min(3, len(val_files))):
        print(f"  {os.path.basename(val_files[i])}")
    
    print("\nExample test files:")
    for i in range(min(3, len(test_files))):
        print(f"  {os.path.basename(test_files[i])}")
    
    # Test frame extraction
    print("\n----- Testing Frame Extraction -----")
    if len(manager.ravdess_files) > 0:
        test_video = manager.ravdess_files[0]
        frame_count = EmotionDatasetManager.get_frame_count(test_video)
        print(f"Frame count in {os.path.basename(test_video)}: {frame_count}")
        
        # Extract a few frames
        frames = EmotionDatasetManager.extract_frames(test_video, max_frames=3)
        print(f"Extracted {len(frames)} frames")
        print(f"Frame shape: {frames[0].shape if frames else 'N/A'}")
        
        # Display a frame
        if frames:
            plt.figure(figsize=(8, 6))
            plt.imshow(frames[0])
            plt.title(f"Sample Frame from {os.path.basename(test_video)}")
            plt.axis('off')
            plt.savefig(os.path.join(args.output_dir, 'sample_frame.png'))
            print(f"Sample frame saved to {os.path.join(args.output_dir, 'sample_frame.png')}")
    
    print("\n===== Testing Completed Successfully =====")

if __name__ == "__main__":
    main()
