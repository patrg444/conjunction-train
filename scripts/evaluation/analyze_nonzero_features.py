#!/usr/bin/env python3
"""
Analyze the distribution of non-zero values in the FaceNet features across
RAVDESS and CREMA-D datasets. This provides insight into how much useful
facial embedding information is present in each file.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random

def analyze_nonzero_data(file_path):
    """
    Analyze a single NPZ file for non-zero data statistics.
    
    Args:
        file_path: Path to the NPZ file
    
    Returns:
        Dictionary with non-zero statistics or None if error
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        filename = os.path.basename(file_path)
        
        result = {
            'filename': filename,
            'video_frames': 0,
            'video_nonzero_percent': 0,
            'video_feature_dim': 0,
            'audio_frames': 0,
            'audio_nonzero_percent': 0,
            'audio_feature_dim': 0,
            'emotion_label': None
        }
        
        # Analyze video features
        if 'video_features' in data:
            video_features = data['video_features']
            
            # Calculate non-zero percentage
            nonzero_percentage = np.mean(video_features != 0) * 100
            
            result['video_frames'] = video_features.shape[0]
            result['video_nonzero_percent'] = nonzero_percentage
            
            if len(video_features.shape) > 1:
                result['video_feature_dim'] = video_features.shape[1]
                
            # Calculate percentage of frames with at least one non-zero value
            frames_with_data = np.sum(np.any(video_features != 0, axis=1))
            result['video_frames_with_data_percent'] = (frames_with_data / video_features.shape[0]) * 100
        
        # Analyze audio features
        if 'audio_features' in data:
            audio_features = data['audio_features']
            
            # Calculate non-zero percentage
            nonzero_percentage = np.mean(audio_features != 0) * 100
            
            result['audio_frames'] = audio_features.shape[0]
            result['audio_nonzero_percent'] = nonzero_percentage
            
            if len(audio_features.shape) > 1:
                result['audio_feature_dim'] = audio_features.shape[1]
                
            # Calculate percentage of frames with at least one non-zero value
            frames_with_data = np.sum(np.any(audio_features != 0, axis=1))
            result['audio_frames_with_data_percent'] = (frames_with_data / audio_features.shape[0]) * 100
        
        # Get emotion label if available
        if 'emotion_label' in data:
            result['emotion_label'] = data['emotion_label'].item()
        
        return result
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def analyze_dataset(directory, max_files=None, recursive=False):
    """
    Analyze all NPZ files in a directory for non-zero data statistics.
    
    Args:
        directory: Directory containing NPZ files
        max_files: Maximum number of files to analyze (randomly sampled)
        recursive: Whether to search recursively for NPZ files
    
    Returns:
        Pandas DataFrame with non-zero statistics for all files
    """
    # Find all NPZ files
    if recursive:
        file_pattern = os.path.join(directory, "**", "*.npz")
        files = glob.glob(file_pattern, recursive=True)
    else:
        file_pattern = os.path.join(directory, "*.npz")
        files = glob.glob(file_pattern)
    
    if not files:
        print(f"No NPZ files found in {directory}")
        return None
    
    print(f"Found {len(files)} NPZ files in {directory}")
    
    # Sample files if requested
    if max_files and max_files < len(files):
        print(f"Sampling {max_files} files for analysis")
        files = random.sample(files, max_files)
    
    # Analyze each file
    results = []
    for file_path in tqdm(files, desc="Analyzing files"):
        result = analyze_nonzero_data(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def generate_statistics(df, output_prefix):
    """
    Generate statistics and visualizations from analysis results.
    
    Args:
        df: Pandas DataFrame with analysis results
        output_prefix: Prefix for output files
    """
    # Calculate overall statistics
    stats = {
        'total_files': len(df),
        'video_frames_mean': df['video_frames'].mean(),
        'video_frames_min': df['video_frames'].min(),
        'video_frames_max': df['video_frames'].max(),
        'video_nonzero_mean': df['video_nonzero_percent'].mean(),
        'video_nonzero_min': df['video_nonzero_percent'].min(),
        'video_nonzero_max': df['video_nonzero_percent'].max(),
        'video_frames_with_data_mean': df['video_frames_with_data_percent'].mean(),
        'audio_frames_mean': df['audio_frames'].mean(),
        'audio_frames_min': df['audio_frames'].min(),
        'audio_frames_max': df['audio_frames'].max(),
        'audio_nonzero_mean': df['audio_nonzero_percent'].mean(),
        'audio_nonzero_min': df['audio_nonzero_percent'].min(),
        'audio_nonzero_max': df['audio_nonzero_percent'].max(),
        'audio_frames_with_data_mean': df['audio_frames_with_data_percent'].mean(),
    }
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Video frames: mean={stats['video_frames_mean']:.1f}, min={stats['video_frames_min']}, max={stats['video_frames_max']}")
    print(f"Video non-zero percentage: mean={stats['video_nonzero_mean']:.2f}%, min={stats['video_nonzero_min']:.2f}%, max={stats['video_nonzero_max']:.2f}%")
    print(f"Video frames with data: mean={stats['video_frames_with_data_mean']:.2f}%")
    print(f"Audio frames: mean={stats['audio_frames_mean']:.1f}, min={stats['audio_frames_min']}, max={stats['audio_frames_max']}")
    print(f"Audio non-zero percentage: mean={stats['audio_nonzero_mean']:.2f}%, min={stats['audio_nonzero_min']:.2f}%, max={stats['audio_nonzero_max']:.2f}%")
    print(f"Audio frames with data: mean={stats['audio_frames_with_data_mean']:.2f}%")
    
    # Generate histograms
    
    # Video non-zero percentage histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['video_nonzero_percent'], bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Non-Zero Values in Video Features')
    plt.xlabel('Percentage of Non-Zero Values')
    plt.ylabel('Number of Files')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_video_nonzero_dist.png")
    plt.close()
    
    # Audio non-zero percentage histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['audio_nonzero_percent'], bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Non-Zero Values in Audio Features')
    plt.xlabel('Percentage of Non-Zero Values')
    plt.ylabel('Number of Files')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_audio_nonzero_dist.png")
    plt.close()
    
    # Video frames with data histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['video_frames_with_data_percent'], bins=50, color='purple', alpha=0.7)
    plt.title('Percentage of Video Frames with at Least One Non-Zero Value')
    plt.xlabel('Percentage of Frames')
    plt.ylabel('Number of Files')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_video_frames_with_data_dist.png")
    plt.close()
    
    # Create bins for ranges of non-zero percentages
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                 '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    df['video_nonzero_bin'] = pd.cut(df['video_nonzero_percent'], bins=bin_edges, labels=bin_labels)
    video_bin_counts = df['video_nonzero_bin'].value_counts().sort_index()
    
    # Bar chart of video non-zero percentage bins
    plt.figure(figsize=(12, 6))
    video_bin_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Distribution of Files by Non-Zero Values in Video Features')
    plt.xlabel('Percentage of Non-Zero Values')
    plt.ylabel('Number of Files')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_video_nonzero_bins.png")
    plt.close()
    
    # Save raw data to CSV
    df.to_csv(f"{output_prefix}_analysis.csv", index=False)
    
    # Save detailed statistics
    with open(f"{output_prefix}_statistics.txt", 'w') as f:
        f.write("=== Dataset Statistics ===\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Add binned statistics
        f.write("\n=== Video Non-Zero Value Distribution ===\n")
        for bin_label, count in video_bin_counts.items():
            percent = (count / len(df)) * 100
            f.write(f"{bin_label}: {count} files ({percent:.2f}%)\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze non-zero values in feature files")
    parser.add_argument("--ravdess-dir", type=str, default="ravdess_features_facenet",
                        help="Directory containing RAVDESS NPZ files")
    parser.add_argument("--cremad-dir", type=str, default="crema_d_features_facenet",
                        help="Directory containing CREMA-D NPZ files")
    parser.add_argument("--output-dir", type=str, default="analysis_output/nonzero_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Maximum number of files to analyze from each dataset")
    parser.add_argument("--recursive", action="store_true",
                        help="Search recursively for NPZ files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze RAVDESS dataset
    print("\n=== Analyzing RAVDESS Dataset ===")
    ravdess_df = analyze_dataset(
        args.ravdess_dir,
        max_files=args.max_files,
        recursive=args.recursive or "ravdess" in args.ravdess_dir.lower()  # Always recursive for RAVDESS
    )
    
    if ravdess_df is not None:
        generate_statistics(ravdess_df, os.path.join(args.output_dir, "ravdess"))
    
    # Analyze CREMA-D dataset
    print("\n=== Analyzing CREMA-D Dataset ===")
    cremad_df = analyze_dataset(
        args.cremad_dir,
        max_files=args.max_files,
        recursive=args.recursive
    )
    
    if cremad_df is not None:
        generate_statistics(cremad_df, os.path.join(args.output_dir, "cremad"))
    
    # Generate combined statistics if both datasets were analyzed
    if ravdess_df is not None and cremad_df is not None:
        # Add dataset column to each dataframe
        ravdess_df['dataset'] = 'RAVDESS'
        cremad_df['dataset'] = 'CREMA-D'
        
        # Combine dataframes
        combined_df = pd.concat([ravdess_df, cremad_df], ignore_index=True)
        
        # Generate comparison visualizations
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='video_nonzero_percent', data=combined_df)
        plt.title('Comparison of Non-Zero Values in Video Features')
        plt.xlabel('Dataset')
        plt.ylabel('Percentage of Non-Zero Values')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_video_nonzero.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='audio_nonzero_percent', data=combined_df)
        plt.title('Comparison of Non-Zero Values in Audio Features')
        plt.xlabel('Dataset')
        plt.ylabel('Percentage of Non-Zero Values')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_audio_nonzero.png"))
        plt.close()
        
        # Save combined data
        combined_df.to_csv(os.path.join(args.output_dir, "combined_analysis.csv"), index=False)

if __name__ == "__main__":
    try:
        import seaborn as sns
        main()
    except ImportError:
        print("Please install seaborn: pip install seaborn")
        # Continue without seaborn
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')  # Use a nice style as fallback
        main()
