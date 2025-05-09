#!/usr/bin/env python3
"""
Analyze CREMA-D frame lengths to determine distribution and identify outliers.
This script helps identify how many samples exceed a certain length threshold.
"""

import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_cremad_frame_lengths():
    # Path to processed CREMA-D files
    cremad_dir = "processed_crema_d_single_segment"
    
    # Collect statistics
    video_lengths = []
    audio_lengths = []
    files_over_75_frames = []
    total_files = 0
    
    # Process each file
    for filename in os.listdir(cremad_dir):
        if filename.endswith('.npz'):
            total_files += 1
            filepath = os.path.join(cremad_dir, filename)
            
            try:
                data = np.load(filepath, allow_pickle=True)
                video_frames = data['video_features'].shape[0]
                audio_frames = data['audio_features'].shape[0]
                
                video_lengths.append(video_frames)
                audio_lengths.append(audio_frames)
                
                # Calculate ratio for this file
                ratio = audio_frames / video_frames if video_frames > 0 else 0
                
                if video_frames > 75:
                    files_over_75_frames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Calculate statistics
    count_over_75 = len(files_over_75_frames)
    percent_over_75 = (count_over_75 / total_files) * 100 if total_files > 0 else 0
    
    # Calculate audio to video ratio
    avg_video_length = sum(video_lengths) / len(video_lengths) if video_lengths else 0
    avg_audio_length = sum(audio_lengths) / len(audio_lengths) if audio_lengths else 0
    avg_ratio = avg_audio_length / avg_video_length if avg_video_length > 0 else 0
    
    # Report results
    print(f"Total CREMA-D files analyzed: {total_files}")
    print(f"Files with over 75 video frames: {count_over_75} ({percent_over_75:.2f}%)")
    print(f"Video frame statistics: min={min(video_lengths)}, max={max(video_lengths)}, avg={sum(video_lengths)/len(video_lengths):.1f}")
    print(f"Audio frame statistics: min={min(audio_lengths)}, max={max(audio_lengths)}, avg={sum(audio_lengths)/len(audio_lengths):.1f}")
    print(f"Average audio-to-video ratio: {avg_ratio:.2f}")
    
    # Show distribution
    length_bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    counts = [0] * (len(length_bins))
    
    for length in video_lengths:
        bin_found = False
        for i in range(len(length_bins)-1):
            if length <= length_bins[i+1]:
                counts[i] += 1
                bin_found = True
                break
        if not bin_found:
            counts[-1] += 1
    
    for i in range(len(length_bins)-1):
        print(f"Files with {length_bins[i]}-{length_bins[i+1]} frames: {counts[i]} ({counts[i]/total_files*100:.2f}%)")
    print(f"Files with >{length_bins[-1]} frames: {counts[-1]} ({counts[-1]/total_files*100:.2f}%)")
    
    # Create a histogram of video frame lengths
    plt.figure(figsize=(10, 6))
    plt.hist(video_lengths, bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of CREMA-D Video Frame Lengths')
    plt.xlabel('Number of Video Frames')
    plt.ylabel('Number of Files')
    plt.axvline(x=75, color='red', linestyle='--', linewidth=2, label='5 second threshold (75 frames)')
    plt.legend()
    plt.savefig('cremad_frame_distribution.png')
    print(f"Histogram saved to cremad_frame_distribution.png")
    
    # Print the first 10 files that exceed 75 frames (as examples)
    if files_over_75_frames:
        print("\nExample files exceeding 75 frames (first 10):")
        for i, filename in enumerate(files_over_75_frames[:10]):
            print(f"  {filename}")

if __name__ == "__main__":
    analyze_cremad_frame_lengths()
