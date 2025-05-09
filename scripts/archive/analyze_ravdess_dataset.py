#!/usr/bin/env python3
"""
Script to analyze the RAVDESS dataset, providing comprehensive statistics on:
- Number of videos per emotion category
- Number of videos per actor
- Number of videos per emotion and intensity
- Number of videos per gender
- Number of processed segments (if available)

Usage:
    python analyze_ravdess_dataset.py --dataset path/to/RAVDESS [--processed path/to/processed_features]
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# RAVDESS intensity mapping
INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

# RAVDESS actor gender mapping (based on documentation)
ACTOR_GENDER = {
    "01": "female", "02": "male", "03": "female", "04": "male",
    "05": "female", "06": "male", "07": "female", "08": "male",
    "09": "female", "10": "male", "11": "female", "12": "male",
    "13": "female", "14": "male", "15": "female", "16": "male",
    "17": "female", "18": "male", "19": "female", "20": "male",
    "21": "female", "22": "male", "23": "female", "24": "male"
}

def count_emotion_videos(dataset_dir="data/RAVDESS"):
    """
    Count the number of videos for each emotion in the RAVDESS dataset.
    
    Args:
        dataset_dir: Directory containing RAVDESS dataset
        
    Returns:
        Tuple of (emotion_counts, actor_emotion_counts, emotion_intensity_counts, actor_counts, gender_counts)
    """
    # Find all MP4 files in the dataset (RAVDESS uses MP4)
    video_pattern = os.path.join(dataset_dir, "**", "*.mp4")
    video_paths = glob.glob(video_pattern, recursive=True)
    
    if not video_paths:
        print(f"No videos found matching pattern {video_pattern}")
        return {}, {}, {}, {}, {}
    
    print(f"Found {len(video_paths)} total videos in {dataset_dir}")
    
    # Count by emotion
    emotion_counts = Counter()
    
    # Count by actor and emotion
    actor_emotion_counts = defaultdict(Counter)
    
    # Count intensity by emotion
    emotion_intensity_counts = defaultdict(Counter)
    
    # Count videos by actor
    actor_counts = Counter()
    
    # Count videos by gender
    gender_counts = Counter()
    
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
            
            # Increment actor counter
            actor_counts[actor_id] += 1
            
            # Increment actor-emotion counter
            actor_emotion_counts[actor_id][emotion_code] += 1
            
            # Increment intensity counter
            if intensity_code:
                emotion_intensity_counts[emotion_code][intensity_code] += 1
                
            # Increment gender counter
            if actor_id in ACTOR_GENDER:
                gender_counts[ACTOR_GENDER[actor_id]] += 1
            
        except Exception as e:
            print(f"Error parsing metadata from {filename}: {str(e)}")
            continue
    
    return emotion_counts, actor_emotion_counts, emotion_intensity_counts, actor_counts, gender_counts

def count_processed_segments(processed_dir):
    """
    Count the number of segments in processed feature files.
    
    Args:
        processed_dir: Directory containing processed features (.npz files)
        
    Returns:
        Tuple of (emotion_segment_counts, file_segment_counts, actor_emotion_segments)
    """
    # Find all NPZ files in the features directory
    npz_pattern = os.path.join(processed_dir, "*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        print(f"No .npz files found in {processed_dir}")
        return {}, {}, {}
    
    print(f"Found {len(npz_files)} processed feature files in {processed_dir}")
    
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

def display_statistics(emotion_counts, actor_emotion_counts, emotion_intensity_counts, 
                       actor_counts, gender_counts, segment_data=None):
    """
    Display comprehensive statistics about the dataset.
    
    Args:
        emotion_counts: Counter with emotion counts
        actor_emotion_counts: Dict of Counters with actor-emotion counts
        emotion_intensity_counts: Dict of Counters with emotion-intensity counts
        actor_counts: Counter with actor counts
        gender_counts: Counter with gender counts
        segment_data: Optional tuple of segment statistics
    """
    # Display emotion counts
    print("\n" + "=" * 60)
    print(f"EMOTION DISTRIBUTION")
    print("=" * 60)
    
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
    
    # Display gender breakdown
    print("\n" + "=" * 60)
    print(f"GENDER DISTRIBUTION")
    print("=" * 60)
    
    # Create gender dataframe
    gender_df = pd.DataFrame({
        'Gender': list(gender_counts.keys()),
        'Count': list(gender_counts.values())
    })
    
    # Add total
    total_gender = gender_df['Count'].sum()
    gender_df.loc[len(gender_df)] = ['Total', total_gender]
    
    # Add percentage
    gender_df['Percentage'] = gender_df['Count'].apply(
        lambda x: f"{x / total_gender * 100:.1f}%" if x != total_gender else "100.0%"
    )
    
    print(gender_df.to_string(index=False))
    
    # Display actor counts
    print("\n" + "=" * 60)
    print(f"VIDEOS PER ACTOR")
    print("=" * 60)
    
    # Create actor dataframe
    actor_df = pd.DataFrame({
        'Actor ID': list(actor_counts.keys()),
        'Gender': [ACTOR_GENDER.get(actor, "Unknown") for actor in actor_counts.keys()],
        'Count': list(actor_counts.values())
    })
    
    # Sort by actor ID
    actor_df = actor_df.sort_values('Actor ID')
    
    # Add total
    total_actor_vids = actor_df['Count'].sum()
    actor_df.loc[len(actor_df)] = ['Total', 'All', total_actor_vids]
    
    # Add percentage
    actor_df['Percentage'] = actor_df['Count'].apply(
        lambda x: f"{x / total_actor_vids * 100:.1f}%" if x != total_actor_vids else "100.0%"
    )
    
    print(actor_df.to_string(index=False))
    
    # Display actor-emotion breakdown
    print("\n" + "=" * 60)
    print(f"ACTOR-EMOTION BREAKDOWN")
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
    
    # Display intensity breakdown if available
    if emotion_intensity_counts:
        print("\n" + "=" * 60)
        print(f"EMOTION-INTENSITY BREAKDOWN")
        print("=" * 60)
        
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
    
    # Display segment information if available
    if segment_data:
        emotion_segment_counts, file_segment_counts, actor_emotion_segments = segment_data
        
        print("\n" + "=" * 60)
        print(f"PROCESSED SEGMENTS DISTRIBUTION")
        print("=" * 60)
        
        # Create a dataframe for segment counts by emotion
        segment_df = pd.DataFrame({
            'Emotion Code': list(emotion_segment_counts.keys()),
            'Emotion Name': [EMOTION_MAP.get(code, f"Unknown-{code}") for code in emotion_segment_counts.keys()],
            'Segment Count': list(emotion_segment_counts.values())
        })
        
        # Sort by emotion code
        segment_df = segment_df.sort_values('Emotion Code')
        
        # Add total
        total_segments = segment_df['Segment Count'].sum()
        segment_df.loc[len(segment_df)] = ['Total', 'All Emotions', total_segments]
        
        # Add percentage
        segment_df['Percentage'] = segment_df['Segment Count'].apply(
            lambda x: f"{x / total_segments * 100:.1f}%" if x != total_segments else "100.0%"
        )
        
        print(segment_df.to_string(index=False))
        
        # Display segments per file statistics
        print("\n" + "=" * 60)
        print(f"SEGMENTS PER FILE STATISTICS")
        print("=" * 60)
        
        segments_per_file = list(file_segment_counts.values())
        
        if segments_per_file:
            min_segments = min(segments_per_file)
            max_segments = max(segments_per_file)
            avg_segments = sum(segments_per_file) / len(segments_per_file)
            
            print(f"Minimum segments per file: {min_segments}")
            print(f"Maximum segments per file: {max_segments}")
            print(f"Average segments per file: {avg_segments:.2f}")
            print(f"Total files: {len(file_segment_counts)}")
            print(f"Total segments: {sum(segments_per_file)}")
        else:
            print("No segment data available.")

def generate_plots(emotion_counts, actor_counts, gender_counts, segment_data=None, output_dir="visualization"):
    """
    Generate visualization plots of the dataset statistics.
    
    Args:
        emotion_counts: Counter with emotion counts
        actor_counts: Counter with actor counts
        gender_counts: Counter with gender counts
        segment_data: Optional tuple of segment statistics
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Emotion distribution plot
    plt.figure(figsize=(12, 6))
    emotions = [EMOTION_MAP.get(code, f"Unknown-{code}") for code in sorted(emotion_counts.keys())]
    counts = [emotion_counts[code] for code in sorted(emotion_counts.keys())]
    
    plt.bar(emotions, counts, color='skyblue')
    plt.title('Distribution of Emotions in RAVDESS Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    
    # 2. Gender distribution pie chart
    plt.figure(figsize=(8, 8))
    genders = list(gender_counts.keys())
    gender_values = list(gender_counts.values())
    
    plt.pie(gender_values, labels=genders, autopct='%1.1f%%', colors=['lightpink', 'lightblue'])
    plt.title('Gender Distribution in RAVDESS Dataset')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    
    # 3. Videos per actor bar chart
    plt.figure(figsize=(14, 6))
    actors = sorted(actor_counts.keys())
    actor_values = [actor_counts[actor] for actor in actors]
    actor_colors = ['lightpink' if ACTOR_GENDER.get(actor) == 'female' else 'lightblue' for actor in actors]
    
    plt.bar(actors, actor_values, color=actor_colors)
    plt.title('Videos per Actor in RAVDESS Dataset')
    plt.xlabel('Actor ID')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'videos_per_actor.png'))
    
    # 4. Segments distribution (if available)
    if segment_data:
        emotion_segment_counts, _, _ = segment_data
        plt.figure(figsize=(12, 6))
        segment_emotions = [EMOTION_MAP.get(code, f"Unknown-{code}") for code in sorted(emotion_segment_counts.keys())]
        segment_counts = [emotion_segment_counts[code] for code in sorted(emotion_segment_counts.keys())]
        
        plt.bar(segment_emotions, segment_counts, color='lightgreen')
        plt.title('Distribution of Processed Segments by Emotion')
        plt.xlabel('Emotion')
        plt.ylabel('Number of Segments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'segment_distribution.png'))
    
    print(f"\nPlots saved to {output_dir} directory")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze the RAVDESS dataset for emotion recognition research.')
    
    parser.add_argument('--dataset', type=str, default='data/RAVDESS',
                        help='Directory containing RAVDESS dataset')
    
    parser.add_argument('--processed', type=str, default=None,
                        help='Optional: Directory containing processed feature files (.npz)')
    
    parser.add_argument('--plots', action='store_true',
                        help='Generate visualization plots')
    
    parser.add_argument('--output-dir', type=str, default='visualization',
                        help='Directory to save visualization plots')
    
    return parser.parse_args()

def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print(f"ANALYZING RAVDESS DATASET")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset}")
    if args.processed:
        print(f"Processed features: {args.processed}")
    
    # Count videos
    emotion_counts, actor_emotion_counts, emotion_intensity_counts, actor_counts, gender_counts = count_emotion_videos(args.dataset)
    
    if not emotion_counts:
        print("No valid videos found to analyze!")
        return
    
    # Count processed segments if directory provided
    segment_data = None
    if args.processed:
        segment_data = count_processed_segments(args.processed)
    
    # Display statistics
    display_statistics(emotion_counts, actor_emotion_counts, emotion_intensity_counts, 
                       actor_counts, gender_counts, segment_data)
    
    # Generate plots if requested
    if args.plots:
        generate_plots(emotion_counts, actor_counts, gender_counts, segment_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
