#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check the class distribution across the datasets used for training the emotion recognition model.
This helps identify data imbalance issues that might affect model training.
"""

import os
import sys
import numpy as np
import glob
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Constants
EMOTION_NAMES = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']  # Order used in training
RAVDESS_EMOTION_MAP = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}  # Index mapping

def analyze_dataset_distribution(file_paths, dataset_name="Dataset"):
    """Analyze the class distribution of a specific dataset"""
    emotion_counts = Counter()
    valid_files = 0
    skipped_files = 0
    
    print(f"\n=== Analyzing {dataset_name} Distribution ===")
    
    for file_path in tqdm(file_paths, desc=f"Processing {dataset_name} files"):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.endswith('_facenet_features'):
            base_name = base_name[:-len('_facenet_features')]
            
        try:
            # RAVDESS format: 01-01-03-02-01-01-XX.npz (3rd position is emotion)
            if "Actor_" in file_path:
                parts = base_name.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_name = RAVDESS_EMOTION_MAP.get(emotion_code)
                    if emotion_name in EMOTION_MAP:
                        emotion_counts[emotion_name] += 1
                        valid_files += 1
                    else:
                        skipped_files += 1
                else:
                    skipped_files += 1
            
            # CREMA-D format: 1076_IEO_ANG_XX.npz
            else:
                parts = base_name.split('_')
                if len(parts) >= 3:
                    emotion_name = parts[2]
                    if emotion_name in EMOTION_MAP:
                        emotion_counts[emotion_name] += 1
                        valid_files += 1
                    else:
                        skipped_files += 1
                else:
                    skipped_files += 1
                    
        except Exception as e:
            skipped_files += 1
    
    # Print results
    print(f"\n{dataset_name} Statistics:")
    print(f"Total files: {len(file_paths)}")
    print(f"Valid files: {valid_files}")
    print(f"Skipped files: {skipped_files}")
    print("\nEmotion Distribution:")
    
    total = sum(emotion_counts.values())
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: EMOTION_MAP.get(x[0], 999)):
        print(f"{emotion}: {count} ({count/total*100:.1f}%)")
    
    return emotion_counts

def visualize_distributions(ravdess_counts, cremad_counts, combined_counts=None):
    """Create a bar chart comparing the distributions"""
    # Prepare data
    emotions = sorted(EMOTION_MAP.keys(), key=lambda x: EMOTION_MAP[x])
    ravdess_values = [ravdess_counts.get(e, 0) for e in emotions]
    cremad_values = [cremad_counts.get(e, 0) for e in emotions]
    
    if combined_counts:
        combined_values = [combined_counts.get(e, 0) for e in emotions]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the width of the bars
    bar_width = 0.3
    index = np.arange(len(emotions))
    
    # Create the bars
    ravdess_bars = ax.bar(index - bar_width/2, ravdess_values, bar_width, 
                          label='RAVDESS', color='skyblue', edgecolor='black')
    cremad_bars = ax.bar(index + bar_width/2, cremad_values, bar_width,
                         label='CREMA-D', color='lightgreen', edgecolor='black')
    
    if combined_counts:
        combined_bars = ax.bar(index + 1.5*bar_width, combined_values, bar_width,
                              label='Combined', color='salmon', edgecolor='black')
    
    # Add labels, title, etc.
    ax.set_xlabel('Emotion', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Emotion Distribution Across Datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(emotions)
    ax.legend()
    
    # Add count labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    add_labels(ravdess_bars)
    add_labels(cremad_bars)
    if combined_counts:
        add_labels(combined_bars)
    
    # Calculate and add class imbalance ratio
    if combined_counts:
        counts = list(combined_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        ax.text(0.5, 0.95, f'Class Imbalance Ratio: {imbalance_ratio:.2f}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('emotion_distribution_comparison.png')
    print("\nVisualization saved to 'emotion_distribution_comparison.png'")
    
    return fig

def calculate_class_weights(combined_counts):
    """Calculate class weights to balance training"""
    # Get total samples per class
    counts = np.array([combined_counts.get(emotion, 0) for emotion in sorted(EMOTION_MAP.keys(), key=lambda x: EMOTION_MAP[x])])
    
    # Compute class weights: n_samples / (n_classes * np.bincount(y))
    n_samples = np.sum(counts)
    n_classes = len(EMOTION_MAP)
    
    class_weights = {}
    for emotion, idx in sorted(EMOTION_MAP.items(), key=lambda x: x[1]):
        count = combined_counts.get(emotion, 0)
        if count > 0:
            weight = n_samples / (n_classes * count)
            class_weights[idx] = weight
        else:
            class_weights[idx] = 1.0
    
    print("\n=== Class Weights for Training ===")
    for idx, weight in sorted(class_weights.items()):
        emotion = [e for e, i in EMOTION_MAP.items() if i == idx][0]
        print(f"{emotion} (class {idx}): {weight:.4f}")
    
    print("\nAdd these weights to your training script as:")
    print("class_weight = {", end="")
    for idx, weight in sorted(class_weights.items()):
        print(f"{idx}: {weight:.4f}", end=", " if idx < max(class_weights.keys()) else "")
    print("}")
    
    return class_weights

def main():
    parser = argparse.ArgumentParser(description='Analyze emotion class distribution across datasets')
    parser.add_argument('--ravdess_dir', type=str, default='/home/ubuntu/emotion-recognition/ravdess_features_facenet',
                        help='Directory containing RAVDESS Facenet features')
    parser.add_argument('--cremad_dir', type=str, default='/home/ubuntu/emotion-recognition/crema_d_features_facenet',
                        help='Directory containing CREMA-D Facenet features')
    parser.add_argument('--local', action='store_true', 
                        help='Use local paths for testing')
    
    args = parser.parse_args()
    
    if args.local:
        ravdess_dir = './ravdess_features_facenet'
        cremad_dir = './crema_d_features_facenet'
    else:
        ravdess_dir = args.ravdess_dir
        cremad_dir = args.cremad_dir
    
    print(f"RAVDESS directory: {ravdess_dir}")
    print(f"CREMA-D directory: {cremad_dir}")
    
    # Find all feature files
    ravdess_files = glob.glob(os.path.join(ravdess_dir, "Actor_*", "*.npz"))
    cremad_files = glob.glob(os.path.join(cremad_dir, "*.npz"))
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(cremad_files)} CREMA-D files")
    
    # Analyze each dataset
    ravdess_counts = analyze_dataset_distribution(ravdess_files, "RAVDESS")
    cremad_counts = analyze_dataset_distribution(cremad_files, "CREMA-D")
    
    # Combine counts
    combined_counts = Counter()
    for emotion, count in ravdess_counts.items():
        combined_counts[emotion] += count
    for emotion, count in cremad_counts.items():
        combined_counts[emotion] += count
    
    print("\n=== Combined Dataset Statistics ===")
    total = sum(combined_counts.values())
    for emotion, count in sorted(combined_counts.items(), key=lambda x: EMOTION_MAP.get(x[0], 999)):
        print(f"{emotion}: {count} ({count/total*100:.1f}%)")
    
    # Visualize the distributions
    visualize_distributions(ravdess_counts, cremad_counts, combined_counts)
    
    # Calculate class weights for training
    class_weights = calculate_class_weights(combined_counts)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
