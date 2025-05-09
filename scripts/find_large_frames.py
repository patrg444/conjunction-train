#!/usr/bin/env python3
"""
Script to identify .npz files with specific video frame counts in crema_d_features
"""

import os
import sys
import numpy as np
from tqdm import tqdm

def main():
    # Directory containing the features
    features_dir = "crema_d_features"
    
    # Specific frame count to look for
    target_frame_count = 928
    
    # To store results
    matching_files = []
    largest_files = []
    
    print(f"Scanning {features_dir} for files with {target_frame_count} video frames...")
    
    # Get list of all .npz files
    npz_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
    
    # Examine each file
    for filename in tqdm(npz_files):
        filepath = os.path.join(features_dir, filename)
        try:
            # Load the .npz file
            data = np.load(filepath, allow_pickle=True)
            
            # Check if video_features exists
            if 'video_features' in data:
                video_features = data['video_features']
                frame_count = video_features.shape[0]
                
                # Check if this file matches the target frame count
                if frame_count == target_frame_count:
                    audio_frame_count = data['audio_features'].shape[0] if 'audio_features' in data else "N/A"
                    matching_files.append((filename, frame_count, audio_frame_count))
                
                # Keep track of largest files too
                if frame_count > 200:  # Only track unusually large files
                    audio_frame_count = data['audio_features'].shape[0] if 'audio_features' in data else "N/A"
                    largest_files.append((filename, frame_count, audio_frame_count))
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Print results for exact matches
    if matching_files:
        print(f"\nFound {len(matching_files)} file(s) with exactly {target_frame_count} video frames:")
        for filename, video_frames, audio_frames in matching_files:
            print(f"  - {filename} (Video: {video_frames}, Audio: {audio_frames})")
            print(f"    Path: {os.path.join(features_dir, filename)}")
    else:
        print(f"\nNo files found with exactly {target_frame_count} video frames.")
    
    # Also show largest files by frame count
    if largest_files:
        # Sort by frame count, largest first
        largest_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\nLargest files by video frame count:")
        for i, (filename, video_frames, audio_frames) in enumerate(largest_files[:10]):  # Show top 10
            print(f"  {i+1}. {filename} (Video: {video_frames}, Audio: {audio_frames})")
            print(f"     Path: {os.path.join(features_dir, filename)}")
    
    return matching_files

if __name__ == "__main__":
    main()
