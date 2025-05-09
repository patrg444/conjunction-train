#!/usr/bin/env python3
"""
Dataset Balance Verification

This script analyzes RAVDESS and CREMA-D feature archives to count samples per emotion class.
It helps identify potential class imbalance issues before training.
"""

import os
import glob
import argparse
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Emotion mappings from the training script
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
RAVDESS_EMOTION_MAP = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', 
                       '05': 'ANG', '06': 'FEA', '07': 'DIS'}

def count_npz_files_by_class(path_pattern, dataset_name):
    """
    Count NPZ files by emotion class.
    
    Args:
        path_pattern: Glob pattern for NPZ files
        dataset_name: Name of dataset for reporting
        
    Returns:
        Counter object with class counts
    """
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"Error: No files found matching pattern: {path_pattern}")
        return Counter()
        
    print(f"Processing {len(files)} files from {dataset_name}")
    class_counter = Counter()
    
    for file_path in files:
        try:
            # Extract emotion code from filename using same logic as training script
            filename = os.path.basename(file_path)
            if '-' in filename:  # RAVDESS format
                parts = filename.split('-')
                emotion_code = RAVDESS_EMOTION_MAP.get(parts[2], None) if len(parts) >= 3 else None
            else:  # CREMA-D format
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None
                
            # Skip files with unknown emotion codes
            if emotion_code not in EMOTION_MAP:
                continue
                
            # Check if file is valid by ensuring it has the required arrays
            data = np.load(file_path)
            if 'audio_features' not in data or 'video_features' not in data:
                continue
                
            # Basic check for minimal length
            audio_features = data['audio_features']
            video_features = data['video_features']
            if len(audio_features) < 5 or len(video_features) < 5:
                continue
                
            # Count the valid file
            class_counter[emotion_code] += 1
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            
    return class_counter

def plot_class_distribution(class_counts, title="Emotion Class Distribution"):
    """Generate a bar chart of class distribution."""
    # Convert to ordered list based on emotion map
    counts = [class_counts.get(emotion, 0) for emotion in EMOTION_MAP.keys()]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(EMOTION_NAMES, counts, color='skyblue')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Emotion Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('class_distribution.png')
    plt.close()
    print(f"Plot saved to class_distribution.png")

def main():
    parser = argparse.ArgumentParser(
        description="Verify emotion class balance in RAVDESS and CREMA-D feature datasets")
    parser.add_argument("--ravdess", type=str, 
                      default="ravdess_features_facenet/*/*.npz",
                      help="Glob pattern for RAVDESS npz files")
    parser.add_argument("--cremad", type=str, 
                      default="crema_d_features_facenet/*.npz",
                      help="Glob pattern for CREMA-D npz files")
    parser.add_argument("--no-plot", action="store_true",
                      help="Disable plotting (text output only)")
    args = parser.parse_args()
    
    # Process each dataset
    ravdess_counts = count_npz_files_by_class(args.ravdess, "RAVDESS")
    cremad_counts = count_npz_files_by_class(args.cremad, "CREMA-D")
    
    # Combine the counts
    total_counts = Counter()
    for emotion in EMOTION_MAP.keys():
        total_counts[emotion] = ravdess_counts.get(emotion, 0) + cremad_counts.get(emotion, 0)
    
    # Print results
    print("\nEmotion Class Counts:")
    print("=" * 40)
    print(f"{'Emotion':<10} {'RAVDESS':>10} {'CREMA-D':>10} {'TOTAL':>10}")
    print("-" * 40)
    
    for emotion, code in EMOTION_MAP.items():
        emotion_name = EMOTION_NAMES[code]
        print(f"{emotion_name:<10} {ravdess_counts.get(emotion, 0):>10} "
              f"{cremad_counts.get(emotion, 0):>10} {total_counts.get(emotion, 0):>10}")
    
    print("-" * 40)
    print(f"{'TOTAL':<10} {sum(ravdess_counts.values()):>10} "
          f"{sum(cremad_counts.values()):>10} {sum(total_counts.values()):>10}")
    
    # Calculate class distribution percentages
    total_samples = sum(total_counts.values())
    if total_samples > 0:
        print("\nClass Distribution (%):")
        print("=" * 40)
        for emotion, code in EMOTION_MAP.items():
            emotion_name = EMOTION_NAMES[code]
            percentage = (total_counts.get(emotion, 0) / total_samples) * 100
            print(f"{emotion_name:<10}: {percentage:.2f}%")
    
    # Generate plot unless disabled
    if not args.no_plot and sum(total_counts.values()) > 0:
        plot_class_distribution(total_counts)

if __name__ == "__main__":
    main()
