#!/usr/bin/env python3
"""
Compare feature distributions between RAVDESS and CREMA-D datasets.
This script loads samples from both datasets and generates comparative visualizations
of their audio and video features.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm

def load_random_samples(ravdess_dir, crema_d_dir, num_samples=5, seed=42):
    """
    Load random samples from both datasets.
    
    Args:
        ravdess_dir: Directory containing RAVDESS .npz files
        crema_d_dir: Directory containing CREMA-D .npz files
        num_samples: Number of random samples to load from each dataset
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with loaded samples
    """
    random.seed(seed)
    
    # Find all .npz files in both directories
    ravdess_files = glob.glob(os.path.join(ravdess_dir, "*.npz"))
    crema_d_files = glob.glob(os.path.join(crema_d_dir, "*.npz"))
    
    if not ravdess_files or not crema_d_files:
        print(f"Error: Could not find .npz files in one or both directories")
        return None
    
    # Sample files
    if len(ravdess_files) > num_samples:
        ravdess_files = random.sample(ravdess_files, num_samples)
    
    if len(crema_d_files) > num_samples:
        crema_d_files = random.sample(crema_d_files, num_samples)
    
    # Load the data
    ravdess_samples = []
    crema_d_samples = []
    
    print(f"Loading {len(ravdess_files)} RAVDESS samples...")
    for file_path in ravdess_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            if 'video_features' in data and 'audio_features' in data:
                sample = {
                    'filename': os.path.basename(file_path),
                    'video_features': data['video_features'],
                    'audio_features': data['audio_features'],
                    'emotion_label': data['emotion_label'].item() if 'emotion_label' in data else None
                }
                ravdess_samples.append(sample)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loading {len(crema_d_files)} CREMA-D samples...")
    for file_path in crema_d_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            if 'video_features' in data and 'audio_features' in data:
                sample = {
                    'filename': os.path.basename(file_path),
                    'video_features': data['video_features'],
                    'audio_features': data['audio_features'],
                    'emotion_label': data['emotion_label'].item() if 'emotion_label' in data else None
                }
                crema_d_samples.append(sample)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return {
        'ravdess': ravdess_samples,
        'crema_d': crema_d_samples
    }

def plot_sequence_length_distributions(samples, output_dir):
    """
    Plot histograms of sequence lengths for both datasets.
    
    Args:
        samples: Dictionary with loaded samples
        output_dir: Directory to save output visualizations
    """
    ravdess_video_lengths = [s['video_features'].shape[0] for s in samples['ravdess']]
    ravdess_audio_lengths = [s['audio_features'].shape[0] for s in samples['ravdess']]
    crema_d_video_lengths = [s['video_features'].shape[0] for s in samples['crema_d']]
    crema_d_audio_lengths = [s['audio_features'].shape[0] for s in samples['crema_d']]
    
    plt.figure(figsize=(15, 10))
    
    # Video sequence lengths
    plt.subplot(2, 1, 1)
    plt.hist(ravdess_video_lengths, bins=20, alpha=0.5, label='RAVDESS')
    plt.hist(crema_d_video_lengths, bins=20, alpha=0.5, label='CREMA-D')
    plt.title('Video Sequence Lengths')
    plt.xlabel('Number of Frames')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Audio sequence lengths
    plt.subplot(2, 1, 2)
    plt.hist(ravdess_audio_lengths, bins=20, alpha=0.5, label='RAVDESS')
    plt.hist(crema_d_audio_lengths, bins=20, alpha=0.5, label='CREMA-D')
    plt.title('Audio Sequence Lengths')
    plt.xlabel('Number of Frames')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_length_distribution.png'))
    plt.close()
    
    print(f"Sequence length distributions saved to {output_dir}")

def plot_feature_value_distributions(samples, output_dir):
    """
    Plot histograms of feature values for both datasets.
    
    Args:
        samples: Dictionary with loaded samples
        output_dir: Directory to save output visualizations
    """
    # Flatten all features for each dataset
    ravdess_video_values = np.concatenate([s['video_features'].flatten() for s in samples['ravdess']])
    ravdess_audio_values = np.concatenate([s['audio_features'].flatten() for s in samples['ravdess']])
    crema_d_video_values = np.concatenate([s['video_features'].flatten() for s in samples['crema_d']])
    crema_d_audio_values = np.concatenate([s['audio_features'].flatten() for s in samples['crema_d']])
    
    # Filter out zeros for clarity in the histograms (they dominate otherwise)
    ravdess_video_values_nonzero = ravdess_video_values[ravdess_video_values > 0]
    crema_d_video_values_nonzero = crema_d_video_values[crema_d_video_values > 0]
    
    # Create figure with 4 subplots
    plt.figure(figsize=(15, 12))
    
    # Video feature distributions - All values
    plt.subplot(2, 2, 1)
    plt.hist(ravdess_video_values, bins=100, alpha=0.5, label='RAVDESS')
    plt.hist(crema_d_video_values, bins=100, alpha=0.5, label='CREMA-D')
    plt.title('Video Feature Values (All)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Video feature distributions - Non-zero values
    plt.subplot(2, 2, 2)
    plt.hist(ravdess_video_values_nonzero, bins=100, alpha=0.5, label='RAVDESS')
    plt.hist(crema_d_video_values_nonzero, bins=100, alpha=0.5, label='CREMA-D')
    plt.title('Video Feature Values (Non-zero)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Audio feature distributions - Full range
    plt.subplot(2, 2, 3)
    plt.hist(ravdess_audio_values, bins=100, alpha=0.5, label='RAVDESS', range=(-100, 100))
    plt.hist(crema_d_audio_values, bins=100, alpha=0.5, label='CREMA-D', range=(-100, 100))
    plt.title('Audio Feature Values (Central Range)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Audio feature distributions - Zoomed to most common range
    plt.subplot(2, 2, 4)
    plt.hist(ravdess_audio_values, bins=100, alpha=0.5, label='RAVDESS', range=(-10, 10))
    plt.hist(crema_d_audio_values, bins=100, alpha=0.5, label='CREMA-D', range=(-10, 10))
    plt.title('Audio Feature Values (Zoomed)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_value_distributions.png'))
    plt.close()
    
    print(f"Feature value distributions saved to {output_dir}")

def plot_sparsity_comparison(samples, output_dir):
    """
    Plot histograms of feature sparsity for both datasets.
    
    Args:
        samples: Dictionary with loaded samples
        output_dir: Directory to save output visualizations
    """
    # Calculate sparsity (percentage of zero values) for each sample
    ravdess_video_sparsity = [np.mean(s['video_features'] == 0) * 100 for s in samples['ravdess']]
    ravdess_audio_sparsity = [np.mean(s['audio_features'] == 0) * 100 for s in samples['ravdess']]
    crema_d_video_sparsity = [np.mean(s['video_features'] == 0) * 100 for s in samples['crema_d']]
    crema_d_audio_sparsity = [np.mean(s['audio_features'] == 0) * 100 for s in samples['crema_d']]
    
    plt.figure(figsize=(15, 6))
    
    # Video sparsity
    plt.subplot(1, 2, 1)
    plt.boxplot([ravdess_video_sparsity, crema_d_video_sparsity], labels=['RAVDESS', 'CREMA-D'])
    plt.title('Video Feature Sparsity')
    plt.ylabel('Zero Values (%)')
    plt.grid(True, alpha=0.3)
    
    # Audio sparsity
    plt.subplot(1, 2, 2)
    plt.boxplot([ravdess_audio_sparsity, crema_d_audio_sparsity], labels=['RAVDESS', 'CREMA-D'])
    plt.title('Audio Feature Sparsity')
    plt.ylabel('Zero Values (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_sparsity.png'))
    plt.close()
    
    print(f"Feature sparsity comparison saved to {output_dir}")

def plot_dataset_comparison(samples, output_dir):
    """
    Create a comprehensive comparison visualization of the two datasets.
    
    Args:
        samples: Dictionary with loaded samples
        output_dir: Directory to save output visualizations
    """
    # Statistics to compare
    ravdess_video_shapes = [s['video_features'].shape for s in samples['ravdess']]
    ravdess_audio_shapes = [s['audio_features'].shape for s in samples['ravdess']]
    crema_d_video_shapes = [s['video_features'].shape for s in samples['crema_d']]
    crema_d_audio_shapes = [s['audio_features'].shape for s in samples['crema_d']]
    
    ravdess_avg_video_len = np.mean([s[0] for s in ravdess_video_shapes])
    ravdess_avg_audio_len = np.mean([s[0] for s in ravdess_audio_shapes])
    crema_d_avg_video_len = np.mean([s[0] for s in crema_d_video_shapes])
    crema_d_avg_audio_len = np.mean([s[0] for s in crema_d_audio_shapes])
    
    ravdess_video_nonzero = np.mean([np.mean(s['video_features'] != 0) * 100 for s in samples['ravdess']])
    ravdess_audio_nonzero = np.mean([np.mean(s['audio_features'] != 0) * 100 for s in samples['ravdess']])
    crema_d_video_nonzero = np.mean([np.mean(s['video_features'] != 0) * 100 for s in samples['crema_d']])
    crema_d_audio_nonzero = np.mean([np.mean(s['audio_features'] != 0) * 100 for s in samples['crema_d']])
    
    # Create bar chart comparison
    categories = ['Video Length', 'Audio Length', 'Video Non-Zero %', 'Audio Non-Zero %']
    ravdess_values = [ravdess_avg_video_len, ravdess_avg_audio_len, ravdess_video_nonzero, ravdess_audio_nonzero]
    crema_d_values = [crema_d_avg_video_len, crema_d_avg_audio_len, crema_d_video_nonzero, crema_d_audio_nonzero]
    
    # Set width of bars
    barWidth = 0.35
    
    # Set position of bar on X axis
    r1 = np.arange(len(categories))
    r2 = [x + barWidth for x in r1]
    
    plt.figure(figsize=(15, 8))
    
    # Make the plot
    plt.bar(r1, ravdess_values, width=barWidth, edgecolor='white', label='RAVDESS')
    plt.bar(r2, crema_d_values, width=barWidth, edgecolor='white', label='CREMA-D')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Metric', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(categories))], categories)
    
    # Create custom y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Set y-axes scales
    ax1.set_ylabel('Length (frames)')
    ax1.set_ylim([0, 500])
    ax2.set_ylabel('Non-Zero %')
    ax2.set_ylim([0, 100])
    
    # Add legend and title
    plt.title('RAVDESS vs CREMA-D Dataset Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'))
    plt.close()
    
    print(f"Dataset comparison saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare RAVDESS and CREMA-D feature distributions")
    parser.add_argument("--ravdess-dir", default="ravdess_features", help="Directory containing RAVDESS .npz files")
    parser.add_argument("--crema-d-dir", default="crema_d_features", help="Directory containing CREMA-D .npz files")
    parser.add_argument("--output-dir", default=".", help="Directory to save output visualizations")
    parser.add_argument("--samples", type=int, default=20, help="Number of random samples to load from each dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load samples from both datasets
    samples = load_random_samples(args.ravdess_dir, args.crema_d_dir, args.samples, args.seed)
    if not samples:
        print("Error: Failed to load samples. Exiting.")
        return
    
    # Generate visualizations
    plot_sequence_length_distributions(samples, args.output_dir)
    plot_feature_value_distributions(samples, args.output_dir)
    plot_sparsity_comparison(samples, args.output_dir)
    plot_dataset_comparison(samples, args.output_dir)
    
    print("All visualizations completed successfully.")

if __name__ == "__main__":
    main()
