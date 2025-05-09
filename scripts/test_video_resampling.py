#!/usr/bin/env python3
"""
Test script to verify the new video resampling functionality for feature extraction.
This script tests the resampling and compares feature extraction with and without it.
"""

import os
import sys
import time
import numpy as np
import cv2
from multimodal_preprocess_fixed import (
    resample_video,
    extract_frame_level_video_features,
    process_video_for_single_segment
)

def test_video_resampling(video_path, fps=15):
    """Test the video resampling function."""
    print(f"\n=== Testing video resampling for: {video_path} ===")
    
    # Get original video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Original Video: {width}x{height} @ {original_fps:.2f} fps, {frame_count} frames, {duration:.2f} seconds")
    
    # Expected frame count after resampling
    expected_frame_count = int(duration * fps)
    print(f"Expected frames at {fps} fps: ~{expected_frame_count}")
    
    # Resample the video
    start_time = time.time()
    resampled_path = resample_video(video_path, fps=fps)
    resample_time = time.time() - start_time
    print(f"Resampling took {resample_time:.2f} seconds")
    print(f"Resampled video: {resampled_path}")
    
    # Check resampled video properties
    cap = cv2.VideoCapture(resampled_path)
    if not cap.isOpened():
        print(f"Error: Could not open resampled video {resampled_path}")
        return
    
    resampled_fps = cap.get(cv2.CAP_PROP_FPS)
    resampled_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resampled_duration = resampled_frame_count / resampled_fps
    cap.release()
    
    print(f"Resampled Video: {resampled_fps:.2f} fps, {resampled_frame_count} frames, {resampled_duration:.2f} seconds")
    print(f"Actual vs Expected: {resampled_frame_count} vs {expected_frame_count} frames")
    
    return resampled_path

def compare_feature_extraction(video_path, fps=15):
    """Compare feature extraction with and without resampling."""
    print(f"\n=== Comparing feature extraction methods for: {video_path} ===")
    
    # Method 1: Original approach - extract features with on-the-fly downsampling
    print("\nMethod 1: Original approach (on-the-fly downsampling)")
    start_time = time.time()
    features1, timestamps1 = extract_frame_level_video_features(video_path, fps=fps)
    execution_time1 = time.time() - start_time
    
    if features1 is None:
        print("Error: Feature extraction failed with Method 1")
        return
    
    print(f"Features shape: {features1.shape}")
    print(f"Timestamps shape: {timestamps1.shape}")
    print(f"Execution time: {execution_time1:.2f} seconds")
    
    # Method 2: New approach - first resample video, then extract features
    print("\nMethod 2: New approach (pre-resampling)")
    # Resample first
    resampled_path = resample_video(video_path, fps=fps)
    
    # Then extract features (without fps parameter since video is already resampled)
    start_time = time.time()
    features2, timestamps2 = extract_frame_level_video_features(resampled_path)
    execution_time2 = time.time() - start_time
    
    if features2 is None:
        print("Error: Feature extraction failed with Method 2")
        return
    
    print(f"Features shape: {features2.shape}")
    print(f"Timestamps shape: {timestamps2.shape}")
    print(f"Execution time: {execution_time2:.2f} seconds")
    
    # Compare results
    print("\nComparison:")
    print(f"Feature count: {len(features1)} vs {len(features2)}")
    print(f"Speed improvement: {(execution_time1 - execution_time2) / execution_time1 * 100:.1f}%")
    
    # Check if feature counts are similar (may not be exact due to rounding differences)
    count_diff = abs(len(features1) - len(features2))
    if count_diff <= max(2, int(0.1 * len(features1))):  # Allow 10% or 2 frame difference
        print("✓ Feature counts match within acceptable margin")
    else:
        print(f"✗ Feature counts differ significantly: {count_diff} frames difference")
    
    return features1, features2, execution_time1, execution_time2

def test_full_processing(video_path, output_dir="test_output_15fps"):
    """Test the full video processing pipeline."""
    print(f"\n=== Testing full processing for: {video_path} ===")
    
    # Process the video using our new implementation
    start_time = time.time()
    output_file = process_video_for_single_segment(
        video_path=video_path,
        output_dir=output_dir
    )
    execution_time = time.time() - start_time
    
    if output_file:
        print(f"Successfully processed video to: {output_file}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Load and check the saved features
        data = np.load(output_file, allow_pickle=True)
        print("\nSaved data contents:")
        for key in data.files:
            try:
                if isinstance(data[key], np.ndarray):
                    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                else:
                    print(f"  {key}: {data[key]}")
            except ValueError:
                print(f"  {key}: [Object array - requires allow_pickle=True to view details]")
                
        return output_file
    else:
        print("Error: Processing failed")
        return None

if __name__ == "__main__":
    # Use provided video path or default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Try to find a test video
        default_paths = [
            "../data/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4",
            "data/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4",
            "/Users/patrickgloria/conjunction-train/data/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4"
        ]
        
        video_path = None
        for path in default_paths:
            if os.path.exists(path):
                video_path = path
                break
                
        if video_path is None:
            print("Error: Could not find a test video. Please provide a video path.")
            sys.exit(1)
    
    # Test the resampling
    resampled_path = test_video_resampling(video_path)
    
    # Compare feature extraction methods
    features1, features2, time1, time2 = compare_feature_extraction(video_path)
    
    # Test the full processing
    output_file = test_full_processing(video_path)
    
    # Summary
    print("\n=== Summary ===")
    print(f"Input video: {video_path}")
    print(f"Resampled video: {resampled_path}")
    print(f"Original method features: {len(features1)} frames in {time1:.2f} seconds")
    print(f"New method features: {len(features2)} frames in {time2:.2f} seconds")
    print(f"Speed improvement: {(time1 - time2) / time1 * 100:.1f}%")
    print(f"Processed output: {output_file}")
