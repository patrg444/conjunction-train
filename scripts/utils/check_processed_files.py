#!/usr/bin/env python3
"""Check processed RAVDESS NPZ files and their dimensions."""

import os
import sys
import numpy as np
import cv2

def check_processed_files(processed_dir, video_dir=None, num_files=5):
    """Check the processed NPZ files and their dimensions."""
    files = os.listdir(processed_dir)
    if len(files) == 0:
        print(f"No files found in {processed_dir}")
        return
    
    print(f"Checking {min(num_files, len(files))} files in {processed_dir}:")
    
    for i, filename in enumerate(files[:num_files]):
        if not filename.endswith('.npz'):
            continue
            
        npz_path = os.path.join(processed_dir, filename)
        data = np.load(npz_path)
        
        # Print basic info
        print(f"\nFile {i+1}: {filename}")
        print(f"  Keys: {list(data.keys())}")
        print(f"  Video features shape: {data['video_features'].shape}")
        print(f"  Audio features shape: {data['audio_features'].shape}")
        print(f"  Emotion label: {data['emotion_label']}")
        
        # Check original video if path provided
        if video_dir:
            try:
                # For RAVDESS actor directories
                # RAVDESS filename format: 01-01-03-02-01-01-01.npz where last number before extension is actor ID
                if "-" in filename:
                    parts = filename[:-4].split("-")  # Remove .npz extension and split
                    if len(parts) >= 7:  # RAVDESS format has 7 parts
                        actor_id = parts[6]  # Last part is actor ID
                    else:
                        actor_id = "01"  # Default if can't parse
                else:
                    actor_id = "01"  # Default
                
                video_path = os.path.join(video_dir, f"Actor_{actor_id}", f"{filename[:-4]}.mp4")
                
                if not os.path.exists(video_path):
                    print(f"  Original video not found at: {video_path}")
                    continue
                    
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  Failed to open video: {video_path}")
                    continue
                
                # Get video info
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps
                cap.release()
                
                print(f"  Original video: {frame_count} frames, {fps:.2f} fps, {duration:.2f} seconds")
                
                # Calculate expected frames at 15fps
                expected_frames = int(duration * 15)
                print(f"  Expected frames at 15fps: ~{expected_frames}")
                print(f"  Actual extracted frames: {data['video_features'].shape[0]}")
                print(f"  Frame reduction ratio: {data['video_features'].shape[0] / frame_count:.2f}")
                
            except Exception as e:
                print(f"  Error analyzing original video: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processed_dir = sys.argv[1]
    else:
        processed_dir = "../processed_ravdess_fixed"
    
    video_dir = None
    if len(sys.argv) > 2:
        video_dir = sys.argv[2]
    else:
        video_dir = "../data/RAVDESS"
    
    num_files = 5
    if len(sys.argv) > 3:
        try:
            num_files = int(sys.argv[3])
        except ValueError:
            pass
    
    check_processed_files(processed_dir, video_dir, num_files)
