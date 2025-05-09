#!/usr/bin/env python3
"""
Examine the contents of NPZ files in the RAVDESS features directory
to understand their structure, feature dimensions, and other properties.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def examine_npz_file(file_path, verbose=True):
    """
    Examine the contents of a single NPZ file and return a summary of its structure.
    
    Args:
        file_path: Path to the NPZ file
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary containing summary information about the file
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Initialize summary dictionary
        summary = {
            'filename': base_filename,
            'keys': list(data.keys()),
            'has_video_features': False,
            'has_audio_features': False,
            'video_shape': None,
            'audio_shape': None,
            'video_stats': None,
            'audio_stats': None,
            'params': None,
            'emotion_label': None,
            'valid_frames': None
        }
        
        # Check for video features
        if 'video_features' in data:
            video_features = data['video_features']
            summary['has_video_features'] = True
            summary['video_shape'] = video_features.shape
            summary['video_stats'] = {
                'min': float(np.min(video_features)),
                'max': float(np.max(video_features)),
                'mean': float(np.mean(video_features)),
                'std': float(np.std(video_features)),
                'non_zero': float(np.mean(video_features != 0))
            }
        
        # Check for audio features
        if 'audio_features' in data:
            audio_features = data['audio_features']
            summary['has_audio_features'] = True
            summary['audio_shape'] = audio_features.shape
            summary['audio_stats'] = {
                'min': float(np.min(audio_features)),
                'max': float(np.max(audio_features)),
                'mean': float(np.mean(audio_features)),
                'std': float(np.std(audio_features)),
                'non_zero': float(np.mean(audio_features != 0))
            }
        
        # Check for params
        if 'params' in data:
            params = data['params'].item()
            if isinstance(params, dict):
                summary['params'] = params
        
        # Check for emotion label
        if 'emotion_label' in data:
            summary['emotion_label'] = data['emotion_label'].item()
        
        # Check for valid frames
        if 'valid_frames' in data:
            valid_frames = data['valid_frames']
            summary['valid_frames'] = {
                'shape': valid_frames.shape,
                'valid_count': int(np.sum(valid_frames)),
                'total_frames': int(len(valid_frames)),
                'percent_valid': float(np.mean(valid_frames) * 100)
            }
        
        if verbose:
            print(f"\nExamining: {base_filename}")
            print(f"  Keys: {', '.join(summary['keys'])}")
            
            if summary['has_video_features']:
                print(f"  Video features: shape={summary['video_shape']}")
                print(f"    Min: {summary['video_stats']['min']:.4f}, Max: {summary['video_stats']['max']:.4f}")
                print(f"    Mean: {summary['video_stats']['mean']:.4f}, Std: {summary['video_stats']['std']:.4f}")
                print(f"    Non-zero elements: {summary['video_stats']['non_zero'] * 100:.1f}%")
            
            if summary['has_audio_features']:
                print(f"  Audio features: shape={summary['audio_shape']}")
                print(f"    Min: {summary['audio_stats']['min']:.4f}, Max: {summary['audio_stats']['max']:.4f}")
                print(f"    Mean: {summary['audio_stats']['mean']:.4f}, Std: {summary['audio_stats']['std']:.4f}")
                print(f"    Non-zero elements: {summary['audio_stats']['non_zero'] * 100:.1f}%")
            
            if summary['params'] is not None:
                print(f"  Parameters: {summary['params']}")
            
            if summary['emotion_label'] is not None:
                print(f"  Emotion label: {summary['emotion_label']}")
            
            if summary['valid_frames'] is not None:
                print(f"  Valid frames: {summary['valid_frames']['valid_count']}/{summary['valid_frames']['total_frames']} "
                      f"({summary['valid_frames']['percent_valid']:.1f}%)")
        
        return summary
    
    except Exception as e:
        print(f"Error examining {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_dataset(file_paths, output_dir=None):
    """
    Analyze a collection of NPZ files to gather dataset statistics
    
    Args:
        file_paths: List of paths to NPZ files
        output_dir: Directory to save analysis results (plots, etc.)
    """
    if not file_paths:
        print("No files to analyze")
        return
    
    print(f"Analyzing {len(file_paths)} files...")
    
    # Initialize statistics
    summaries = []
    video_shapes = []
    audio_shapes = []
    video_lengths = []
    audio_lengths = []
    video_feature_dims = []
    audio_feature_dims = []
    emotion_labels = []
    
    # Create output directory if necessary
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        summary = examine_npz_file(file_path, verbose=False)
        if summary is not None:
            summaries.append(summary)
            
            if summary['has_video_features']:
                video_shapes.append(summary['video_shape'])
                video_lengths.append(summary['video_shape'][0])
                if len(summary['video_shape']) > 1:
                    video_feature_dims.append(summary['video_shape'][1])
            
            if summary['has_audio_features']:
                audio_shapes.append(summary['audio_shape'])
                audio_lengths.append(summary['audio_shape'][0])
                if len(summary['audio_shape']) > 1:
                    audio_feature_dims.append(summary['audio_shape'][1])
            
            if summary['emotion_label'] is not None:
                emotion_labels.append(summary['emotion_label'])
    
    # Print overall statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total files analyzed: {len(summaries)}")
    
    # Video statistics
    if video_lengths:
        print("\nVideo features:")
        print(f"  Sequence lengths: min={min(video_lengths)}, max={max(video_lengths)}, "
              f"mean={np.mean(video_lengths):.1f}, median={np.median(video_lengths)}")
        
        if video_feature_dims:
            print(f"  Feature dimensions: min={min(video_feature_dims)}, max={max(video_feature_dims)}, "
                  f"most common={max(set(video_feature_dims), key=video_feature_dims.count)}")
    
    # Audio statistics
    if audio_lengths:
        print("\nAudio features:")
        print(f"  Sequence lengths: min={min(audio_lengths)}, max={max(audio_lengths)}, "
              f"mean={np.mean(audio_lengths):.1f}, median={np.median(audio_lengths)}")
        
        if audio_feature_dims:
            print(f"  Feature dimensions: min={min(audio_feature_dims)}, max={max(audio_feature_dims)}, "
                  f"most common={max(set(audio_feature_dims), key=audio_feature_dims.count)}")
    
    # Emotion distribution
    if emotion_labels:
        print("\nEmotion distribution:")
        emotion_counts = {}
        for label in emotion_labels:
            emotion_counts[label] = emotion_counts.get(label, 0) + 1
        
        for label in sorted(emotion_counts.keys()):
            count = emotion_counts[label]
            percent = count / len(emotion_labels) * 100
            print(f"  Emotion {label}: {count} ({percent:.1f}%)")
    
    # Generate plots if output directory is provided
    if output_dir:
        # Plot sequence length histograms
        if video_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(video_lengths, bins=30, alpha=0.7, color='blue')
            plt.title('Distribution of Video Sequence Lengths')
            plt.xlabel('Number of Frames')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'video_length_distribution.png'))
            plt.close()
        
        if audio_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(audio_lengths, bins=30, alpha=0.7, color='green')
            plt.title('Distribution of Audio Sequence Lengths')
            plt.xlabel('Number of Frames')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'audio_length_distribution.png'))
            plt.close()
        
        # Plot emotion distribution
        if emotion_labels:
            emotion_counts = {}
            for label in emotion_labels:
                emotion_counts[label] = emotion_counts.get(label, 0) + 1
            
            # Sort by emotion label
            labels = sorted(emotion_counts.keys())
            counts = [emotion_counts[label] for label in labels]
            
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, color='purple')
            plt.title('Emotion Distribution')
            plt.xlabel('Emotion Label')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Examine NPZ files in a directory")
    parser.add_argument("directory", help="Directory containing NPZ files")
    parser.add_argument("--sample", type=int, default=None, help="Number of random files to examine")
    parser.add_argument("--output", default=None, help="Directory to save analysis results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information for each file")
    
    args = parser.parse_args()
    
    # Find all NPZ files in the directory
    file_pattern = os.path.join(args.directory, "*.npz")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No NPZ files found in {args.directory}")
        return
    
    print(f"Found {len(files)} NPZ files in {args.directory}")
    
    # Sample a subset of files if requested
    if args.sample and args.sample < len(files):
        import random
        files = random.sample(files, args.sample)
        print(f"Sampled {len(files)} files for examination")
    
    if args.verbose:
        # Examine each file individually
        for file_path in files:
            examine_npz_file(file_path)
    else:
        # Perform dataset-level analysis
        analyze_dataset(files, args.output)

if __name__ == "__main__":
    main()
