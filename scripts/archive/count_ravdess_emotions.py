#!/usr/bin/env python3
"""
Script to count the number of videos for each type of emotion classification
in the RAVDESS dataset for all actors.

Usage:
    python count_ravdess_emotions.py --dataset path/to/RAVDESS
"""

import os
import glob
import argparse
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

def count_emotion_videos(dataset_dir="data/RAVDESS"):
    """
    Count the number of videos for each emotion in the RAVDESS dataset.
    
    Args:
        dataset_dir: Directory containing RAVDESS dataset
        
    Returns:
        Dictionary with counts per emotion
    """
    # Find all MP4 files in the dataset (RAVDESS uses MP4)
    video_pattern = os.path.join(dataset_dir, "**", "*.mp4")
    video_paths = glob.glob(video_pattern, recursive=True)
    
    if not video_paths:
        print(f"No videos found matching pattern {video_pattern}")
        return {}, {}, {}
    
    print(f"Found {len(video_paths)} total videos in {dataset_dir}")
    
    # Count by emotion
    emotion_counts = Counter()
    
    # Count by actor and emotion
    actor_emotion_counts = defaultdict(Counter)
    
    # Count intensity by emotion
    emotion_intensity_counts = defaultdict(Counter)
    
    for path in video_paths:
        filename = os.path.basename(path)
        parts = filename.split('-')
        
        # Check if filename matches expected format
        if len(parts) < 3:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        
        # Extract metadata
        try:
            actor_id = parts[0]
            emotion_code = parts[2]
            intensity_code = parts[3] if len(parts) > 3 else None
            
            # Increment emotion counter
            emotion_counts[emotion_code] += 1
            
            # Increment actor-emotion counter
            actor_emotion_counts[actor_id][emotion_code] += 1
            
            # Increment intensity counter
            if intensity_code:
                emotion_intensity_counts[emotion_code][intensity_code] += 1
                
        except Exception as e:
            print(f"Error parsing metadata from {filename}: {str(e)}")
            continue
    
    return emotion_counts, actor_emotion_counts, emotion_intensity_counts

def display_emotion_statistics(emotion_counts, actor_emotion_counts, emotion_intensity_counts):
    """
    Display statistics about emotions in the dataset.
    
    Args:
        emotion_counts: Counter with emotion counts
        actor_emotion_counts: Dict of Counters with actor-emotion counts
        emotion_intensity_counts: Dict of Counters with emotion-intensity counts
    """
    # Display emotion counts
    print("\n" + "=" * 50)
    print(f"EMOTION DISTRIBUTION")
    print("=" * 50)
    
    # Create a dataframe for better display
    emotion_df = pd.DataFrame({
        'Emotion Code': list(emotion_counts.keys()),
        'Emotion Name': [EMOTION_MAP.get(code, f"Unknown-{code}") for code in emotion_counts.keys()],
        'Count': list(emotion_counts.values())
    })
    
    # Sort by emotion code
    emotion_df = emotion_df.sort_values('Emotion Code')
    
    # Add total
    total_videos = emotion_df['Count'].sum()
    emotion_df.loc[len(emotion_df)] = ['Total', 'All Emotions', total_videos]
    
    # Add percentage
    emotion_df['Percentage'] = emotion_df['Count'].apply(
        lambda x: f"{x / total_videos * 100:.1f}%" if x != total_videos else "100.0%"
    )
    
    print(emotion_df.to_string(index=False))
    
    # Display actor-emotion breakdown
    print("\n" + "=" * 50)
    print(f"ACTOR-EMOTION BREAKDOWN")
    print("=" * 50)
    
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
    
    # Display intensity breakdown if available
    if emotion_intensity_counts:
        print("\n" + "=" * 50)
        print(f"EMOTION-INTENSITY BREAKDOWN")
        print("=" * 50)
        
        # Intensity mapping
        INTENSITY_MAP = {
            "01": "normal",
            "02": "strong"
        }
        
        # Create a matrix of emotions and intensities
        emotions = sorted(emotion_intensity_counts.keys())
        intensities = sorted(set(intensity for counters in emotion_intensity_counts.values() 
                                for intensity in counters.keys()))
        
        # Initialize the DataFrame with zeros
        intensity_df = pd.DataFrame(0, 
                                   index=[EMOTION_MAP.get(e, f"Unknown-{e}") for e in emotions],
                                   columns=[INTENSITY_MAP.get(i, f"Unknown-{i}") for i in intensities])
        
        # Fill in the data
        for emotion_code, intensity_counter in emotion_intensity_counts.items():
            emotion_name = EMOTION_MAP.get(emotion_code, f"Unknown-{emotion_code}")
            for intensity_code, count in intensity_counter.items():
                intensity_name = INTENSITY_MAP.get(intensity_code, f"Unknown-{intensity_code}")
                intensity_df.loc[emotion_name, intensity_name] = count
        
        # Add totals row and column
        intensity_df['Total'] = intensity_df.sum(axis=1)
        intensity_df.loc['Total'] = intensity_df.sum()
        
        # Rename index
        intensity_df.index.name = 'Emotion'
        
        print(intensity_df.to_string())

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Count videos for each emotion in RAVDESS dataset.')
    
    parser.add_argument('--dataset', type=str, default='data/RAVDESS',
                        help='Directory containing RAVDESS dataset')
    
    return parser.parse_args()

def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "=" * 50)
    print(f"COUNTING RAVDESS EMOTION VIDEOS")
    print("=" * 50)
    print(f"Dataset directory: {args.dataset}")
    
    # Count videos
    emotion_counts, actor_emotion_counts, emotion_intensity_counts = count_emotion_videos(args.dataset)
    
    if not emotion_counts:
        print("No valid videos found to analyze!")
        return
    
    # Display statistics
    display_emotion_statistics(emotion_counts, actor_emotion_counts, emotion_intensity_counts)
    
    print("\n" + "=" * 50)
    print(f"ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
