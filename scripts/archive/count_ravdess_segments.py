#!/usr/bin/env python3
"""
Script to count the number of 3.5-second segments for each type of emotion classification
in the processed RAVDESS features directory.

Usage:
    python count_ravdess_segments.py --features path/to/processed_features
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# RAVDESS emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def count_segments_by_emotion(features_dir="processed_features_3_5s"):
    """
    Count the number of 3.5-second segments for each emotion in the processed features.
    
    Args:
        features_dir: Directory containing processed features (.npz files)
        
    Returns:
        Dictionary with counts per emotion
    """
    # Find all NPZ files in the features directory
    npz_pattern = os.path.join(features_dir, "*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        print(f"No .npz files found in {features_dir}")
        return {}, {}, {}
    
    print(f"Found {len(npz_files)} processed feature files in {features_dir}")
    
    # Count segments by emotion
    emotion_segment_counts = Counter()
    
    # Count segments by file
    file_segment_counts = {}
    
    # Count by actor and emotion
    actor_emotion_segments = defaultdict(Counter)
    
    for npz_file in npz_files:
        # Extract emotion code from filename
        filename = os.path.basename(npz_file)
        parts = filename.split('-')
        
        # Check if filename matches expected format
        if len(parts) < 3:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        
        # Extract metadata
        try:
            actor_id = parts[0]
            emotion_code = parts[2]
            
            # Load the NPZ file to count segments
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                if 'video_sequences' in data:
                    # Count the number of sequences
                    video_seqs = data['video_sequences']
                    
                    if isinstance(video_seqs, np.ndarray):
                        segment_count = len(video_seqs)
                        
                        # Increment emotion counter
                        emotion_segment_counts[emotion_code] += segment_count
                        
                        # Store file segment count
                        file_segment_counts[filename] = segment_count
                        
                        # Increment actor-emotion counter
                        actor_emotion_segments[actor_id][emotion_code] += segment_count
                    else:
                        print(f"Warning: video_sequences not an array in {npz_file}")
                else:
                    print(f"Warning: No video_sequences found in {npz_file}")
                    
            except Exception as e:
                print(f"Error loading {npz_file}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error parsing metadata from {filename}: {str(e)}")
            continue
    
    return emotion_segment_counts, file_segment_counts, actor_emotion_segments

def display_segment_statistics(emotion_counts, file_counts, actor_emotion_counts):
    """
    Display statistics about segment counts.
    
    Args:
        emotion_counts: Counter with emotion segment counts
        file_counts: Dict with file segment counts
        actor_emotion_counts: Dict of Counters with actor-emotion segment counts
    """
    # Display emotion counts
    print("\n" + "=" * 60)
    print(f"SEGMENT DISTRIBUTION BY EMOTION")
    print("=" * 60)
    
    # Create a dataframe for better display
    emotion_df = pd.DataFrame({
        'Emotion Code': list(emotion_counts.keys()),
        'Emotion Name': [EMOTION_MAP.get(code, f"Unknown-{code}") for code in emotion_counts.keys()],
        'Segment Count': list(emotion_counts.values())
    })
    
    # Sort by emotion code
    emotion_df = emotion_df.sort_values('Emotion Code')
    
    # Add total
    total_segments = emotion_df['Segment Count'].sum()
    emotion_df.loc[len(emotion_df)] = ['Total', 'All Emotions', total_segments]
    
    # Add percentage
    emotion_df['Percentage'] = emotion_df['Segment Count'].apply(
        lambda x: f"{x / total_segments * 100:.1f}%" if x != total_segments else "100.0%"
    )
    
    print(emotion_df.to_string(index=False))
    
    # Display segments per file statistics
    print("\n" + "=" * 60)
    print(f"SEGMENTS PER FILE STATISTICS")
    print("=" * 60)
    
    segments_per_file = list(file_counts.values())
    
    if segments_per_file:
        min_segments = min(segments_per_file)
        max_segments = max(segments_per_file)
        avg_segments = sum(segments_per_file) / len(segments_per_file)
        
        print(f"Minimum segments per file: {min_segments}")
        print(f"Maximum segments per file: {max_segments}")
        print(f"Average segments per file: {avg_segments:.2f}")
        print(f"Total files: {len(file_counts)}")
        print(f"Total segments: {sum(segments_per_file)}")
    else:
        print("No segment data available.")
    
    # Display actor-emotion breakdown
    print("\n" + "=" * 60)
    print(f"SEGMENT DISTRIBUTION BY ACTOR AND EMOTION")
    print("=" * 60)
    
    # Create a matrix of actors and emotions
    actors = sorted(actor_emotion_counts.keys())
    emotions = sorted(EMOTION_MAP.keys())
    
    # Initialize the DataFrame with zeros
    actor_emotion_df = pd.DataFrame(0, index=actors, columns=[EMOTION_MAP[e] for e in emotions])
    
    # Fill in the data
    for actor, emotion_counter in actor_emotion_counts.items():
        for emotion_code, count in emotion_counter.items():
            if emotion_code in EMOTION_MAP:
                actor_emotion_df.loc[actor, EMOTION_MAP[emotion_code]] = count
    
    # Add totals row and column
    actor_emotion_df['Total'] = actor_emotion_df.sum(axis=1)
    actor_emotion_df.loc['Total'] = actor_emotion_df.sum()
    
    # Rename index
    actor_emotion_df.index.name = 'Actor ID'
    
    print(actor_emotion_df.to_string())

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Count 3.5-second segments for each emotion in processed RAVDESS features.')
    
    parser.add_argument('--features', type=str, default='processed_features_3_5s',
                        help='Directory containing processed feature files (.npz)')
    
    return parser.parse_args()

def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print(f"COUNTING RAVDESS SEGMENTS BY EMOTION")
    print("=" * 60)
    print(f"Features directory: {args.features}")
    
    # Count segments
    emotion_counts, file_counts, actor_emotion_counts = count_segments_by_emotion(args.features)
    
    if not emotion_counts:
        print("\nNo valid segment data found to analyze!")
        return
    
    # Display statistics
    display_segment_statistics(emotion_counts, file_counts, actor_emotion_counts)
    
    print("\n" + "=" * 60)
    print(f"SEGMENT ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
