#!/usr/bin/env python3
"""Test script to verify FPS resampling in the video feature extraction."""

import cv2
import numpy as np
import sys
import os

# Import the function from our module
from multimodal_preprocess_fixed import extract_frame_level_video_features

def analyze_fps_resampling(video_path, target_fps=15):
    """Analyze how fps resampling works for a video."""
    print(f"Analyzing FPS resampling for: {video_path}")
    
    # Get original video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    cap.release()
    
    print(f"\nOriginal Video:")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Frame count: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Calculate expected frame count after resampling
    expected_frame_count = int(duration * target_fps)
    sample_interval = int(original_fps / target_fps)
    print(f"\nResampling to {target_fps} FPS:")
    print(f"  Sample interval: Every {sample_interval} frames")
    print(f"  Expected frame count: ~{expected_frame_count}")
    
    # Extract features with resampling
    print(f"\nExtracting features with fps={target_fps}...")
    features, timestamps = extract_frame_level_video_features(video_path, fps=target_fps)
    
    if features is not None:
        print(f"\nExtracted Features:")
        print(f"  Feature count: {len(features)}")
        print(f"  Feature dimensions: {features.shape}")
        print(f"  Timestamps count: {len(timestamps)}")
        
        # Calculate actual FPS
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            actual_fps = (len(timestamps) - 1) / time_span if time_span > 0 else 0
            print(f"  Time span: {time_span:.2f} seconds")
            print(f"  Actual FPS: {actual_fps:.2f}")
    else:
        print("Error: Failed to extract features")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "../data/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4"
    
    fps = 15
    if len(sys.argv) > 2:
        try:
            fps = float(sys.argv[2])
        except ValueError:
            print(f"Invalid FPS value: {sys.argv[2]}, using default {fps}")
    
    analyze_fps_resampling(video_path, fps)
