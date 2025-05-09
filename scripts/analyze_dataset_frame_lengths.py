#!/usr/bin/env python3
"""
Analyze the distribution of frame lengths (both video and audio) in RAVDESS and CREMA-D datasets.
Groups frame counts into brackets and generates visualizations showing the distribution.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def find_npz_files(directory):
    """Find all NPZ files in the given directory (recursively)."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
        
    # For RAVDESS with Actor subdirectories
    if "ravdess" in directory.lower():
        files = []
        actor_dirs = glob.glob(os.path.join(directory, "Actor_*"))
        
        if actor_dirs:
            for actor_dir in actor_dirs:
                actor_files = glob.glob(os.path.join(actor_dir, "*.npz"))
                files.extend(actor_files)
        else:
            # Fallback to searching all subdirectories
            files = glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True)
    else:
        # For CREMA-D with flat structure
        files = glob.glob(os.path.join(directory, "*.npz"))
    
    return sorted(files)

def extract_frame_lengths(npz_files):
    """Extract video and audio frame lengths from NPZ files."""
    video_lengths = []
    audio_lengths = []
    filenames = []
    
    for file_path in tqdm(npz_files, desc="Analyzing files"):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if 'video_features' in data and 'audio_features' in data:
                video_feat = data['video_features']
                audio_feat = data['audio_features']
                
                video_lengths.append(video_feat.shape[0])
                audio_lengths.append(audio_feat.shape[0])
                filenames.append(os.path.basename(file_path))
            else:
                print(f"File {file_path} does not contain expected features.")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return video_lengths, audio_lengths, filenames

def create_brackets(lengths, bracket_edges=None):
    """Group lengths into predefined brackets."""
    if bracket_edges is None:
        # Define default brackets if none provided
        bracket_edges = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, float('inf')]
    
    brackets = {}
    for i in range(len(bracket_edges) - 1):
        lower = bracket_edges[i]
        upper = bracket_edges[i] if bracket_edges[i+1] == float('inf') else bracket_edges[i+1]
        
        if bracket_edges[i+1] == float('inf'):
            bracket_name = f"{lower}+"
        else:
            bracket_name = f"{lower}-{upper-1}"
            
        brackets[bracket_name] = 0
    
    for length in lengths:
        for i in range(len(bracket_edges) - 1):
            if bracket_edges[i] <= length < bracket_edges[i+1]:
                if bracket_edges[i+1] == float('inf'):
                    bracket_name = f"{bracket_edges[i]}+"
                else:
                    bracket_name = f"{bracket_edges[i]}-{bracket_edges[i+1]-1}"
                brackets[bracket_name] += 1
                break
    
    return brackets

def plot_brackets(ravdess_brackets, cremad_brackets, title, output_file):
    """Plot brackets comparison between RAVDESS and CREMA-D."""
    # Combine data for plotting
    brackets = sorted(list(set(ravdess_brackets.keys()) | set(cremad_brackets.keys())),
                    key=lambda x: int(x.split('-')[0].replace('+', '')))
    
    ravdess_counts = [ravdess_brackets.get(b, 0) for b in brackets]
    cremad_counts = [cremad_brackets.get(b, 0) for b in brackets]
    
    ravdess_pct = [count / sum(ravdess_counts) * 100 if sum(ravdess_counts) > 0 else 0 for count in ravdess_counts]
    cremad_pct = [count / sum(cremad_counts) * 100 if sum(cremad_counts) > 0 else 0 for count in cremad_counts]
    
    # Create DataFrame for seaborn
    data = {
        'Bracket': brackets * 2,
        'Count': ravdess_counts + cremad_counts,
        'Percentage': ravdess_pct + cremad_pct,
        'Dataset': ['RAVDESS'] * len(brackets) + ['CREMA-D'] * len(brackets)
    }
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Raw counts plot
    sns.barplot(x='Bracket', y='Count', hue='Dataset', data=df, ax=ax1)
    ax1.set_title(f"{title} - Raw Counts")
    ax1.set_ylabel("Number of Files")
    
    # Percentage plot
    sns.barplot(x='Bracket', y='Percentage', hue='Dataset', data=df, ax=ax2)
    ax2.set_title(f"{title} - Percentage Distribution")
    ax2.set_xlabel("Frame Length Bracket")
    ax2.set_ylabel("Percentage of Files (%)")
    
    # Rotate x-tick labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add count values on the bars
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved visualization to {output_file}")
    
    # Save data to CSV
    csv_file = output_file.replace('.png', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved data to {csv_file}")
    
    return df

def create_frame_length_table(video_lengths, audio_lengths, filenames, output_file):
    """Create a table of frame lengths for each file."""
    data = {
        'Filename': filenames,
        'Video Frames': video_lengths,
        'Audio Frames': audio_lengths
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved frame length table to {output_file}")
    return df

def plot_histograms(ravdess_lengths, cremad_lengths, title, output_file, bins=20):
    """Create histograms of frame lengths for each dataset."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # RAVDESS histogram
    ax1.hist(ravdess_lengths, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f"RAVDESS {title} Distribution")
    ax1.set_ylabel("Number of Files")
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if ravdess_lengths:
        ax1.axvline(np.mean(ravdess_lengths), color='red', linestyle='dashed', linewidth=1)
        ax1.text(np.mean(ravdess_lengths) * 1.1, ax1.get_ylim()[1] * 0.9, 
                f'Mean: {np.mean(ravdess_lengths):.1f}', color='red')
    
    # CREMA-D histogram
    ax2.hist(cremad_lengths, bins=bins, alpha=0.7, color='salmon', edgecolor='black')
    ax2.set_title(f"CREMA-D {title} Distribution")
    ax2.set_xlabel("Number of Frames")
    ax2.set_ylabel("Number of Files")
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if cremad_lengths:
        ax2.axvline(np.mean(cremad_lengths), color='red', linestyle='dashed', linewidth=1)
        ax2.text(np.mean(cremad_lengths) * 1.1, ax2.get_ylim()[1] * 0.9, 
                f'Mean: {np.mean(cremad_lengths):.1f}', color='red')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved histogram to {output_file}")

def analyze_datasets(ravdess_dir, cremad_dir, output_dir):
    """Analyze frame length distributions in RAVDESS and CREMA-D datasets."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find NPZ files
    ravdess_files = find_npz_files(ravdess_dir)
    cremad_files = find_npz_files(cremad_dir)
    
    print(f"Found {len(ravdess_files)} RAVDESS files and {len(cremad_files)} CREMA-D files")
    
    # Extract frame lengths
    ravdess_video_lengths, ravdess_audio_lengths, ravdess_filenames = extract_frame_lengths(ravdess_files)
    cremad_video_lengths, cremad_audio_lengths, cremad_filenames = extract_frame_lengths(cremad_files)
    
    # Define bracket edges
    video_bracket_edges = [0, 25, 50, 75, 100, 150, 200, 300, 500, float('inf')]
    audio_bracket_edges = [0, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, float('inf')]
    
    # Create brackets
    ravdess_video_brackets = create_brackets(ravdess_video_lengths, video_bracket_edges)
    cremad_video_brackets = create_brackets(cremad_video_lengths, video_bracket_edges)
    
    ravdess_audio_brackets = create_brackets(ravdess_audio_lengths, audio_bracket_edges)
    cremad_audio_brackets = create_brackets(cremad_audio_lengths, audio_bracket_edges)
    
    # Plot brackets
    video_df = plot_brackets(ravdess_video_brackets, cremad_video_brackets, 
                          "Video Frame Length Distribution", 
                          os.path.join(output_dir, "video_frame_length_brackets.png"))
    
    audio_df = plot_brackets(ravdess_audio_brackets, cremad_audio_brackets, 
                          "Audio Frame Length Distribution", 
                          os.path.join(output_dir, "audio_frame_length_brackets.png"))
    
    # Create frame length tables
    ravdess_table = create_frame_length_table(
        ravdess_video_lengths, ravdess_audio_lengths, ravdess_filenames,
        os.path.join(output_dir, "ravdess_frame_lengths.csv")
    )
    
    cremad_table = create_frame_length_table(
        cremad_video_lengths, cremad_audio_lengths, cremad_filenames,
        os.path.join(output_dir, "cremad_frame_lengths.csv")
    )
    
    # Plot histograms
    plot_histograms(
        ravdess_video_lengths, cremad_video_lengths, 
        "Video Frame Length", 
        os.path.join(output_dir, "video_frame_length_histogram.png"),
        bins=20
    )
    
    plot_histograms(
        ravdess_audio_lengths, cremad_audio_lengths, 
        "Audio Frame Length", 
        os.path.join(output_dir, "audio_frame_length_histogram.png"),
        bins=20
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("RAVDESS:")
    print(f"  Video frames: min={min(ravdess_video_lengths) if ravdess_video_lengths else 'N/A'}, "
          f"max={max(ravdess_video_lengths) if ravdess_video_lengths else 'N/A'}, "
          f"mean={np.mean(ravdess_video_lengths) if ravdess_video_lengths else 'N/A':.1f}")
    print(f"  Audio frames: min={min(ravdess_audio_lengths) if ravdess_audio_lengths else 'N/A'}, "
          f"max={max(ravdess_audio_lengths) if ravdess_audio_lengths else 'N/A'}, "
          f"mean={np.mean(ravdess_audio_lengths) if ravdess_audio_lengths else 'N/A':.1f}")
    
    print("CREMA-D:")
    print(f"  Video frames: min={min(cremad_video_lengths) if cremad_video_lengths else 'N/A'}, "
          f"max={max(cremad_video_lengths) if cremad_video_lengths else 'N/A'}, "
          f"mean={np.mean(cremad_video_lengths) if cremad_video_lengths else 'N/A':.1f}")
    print(f"  Audio frames: min={min(cremad_audio_lengths) if cremad_audio_lengths else 'N/A'}, "
          f"max={max(cremad_audio_lengths) if cremad_audio_lengths else 'N/A'}, "
          f"mean={np.mean(cremad_audio_lengths) if cremad_audio_lengths else 'N/A':.1f}")
    
    return {
        'ravdess_video': ravdess_video_lengths,
        'ravdess_audio': ravdess_audio_lengths,
        'cremad_video': cremad_video_lengths,
        'cremad_audio': cremad_audio_lengths,
        'ravdess_video_brackets': ravdess_video_brackets,
        'cremad_video_brackets': cremad_video_brackets,
        'ravdess_audio_brackets': ravdess_audio_brackets,
        'cremad_audio_brackets': cremad_audio_brackets
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze frame length distributions in RAVDESS and CREMA-D datasets')
    parser.add_argument('--ravdess-dir', type=str, default='ravdess_features_facenet',
                      help='Directory containing RAVDESS features')
    parser.add_argument('--cremad-dir', type=str, default='crema_d_features_facenet',
                      help='Directory containing CREMA-D features')
    parser.add_argument('--output-dir', type=str, default='analysis_output/frame_lengths',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_datasets(args.ravdess_dir, args.cremad_dir, args.output_dir)
