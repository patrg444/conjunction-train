#!/usr/bin/env python3
"""
Analyze the processed RAVDESS dataset to get statistics about emotions, actors, and sequence counts.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse

def analyze_dataset(data_dir):
    """Analyzes the npz files in data_dir and prints statistics."""
    file_pattern = os.path.join(data_dir, "*.npz")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No .npz files found in {data_dir}")
        return
    
    print(f"Found {len(files)} .npz files in {data_dir}")
    
    total_sequences = 0
    emotion_labels = []
    actor_ids = []
    sequence_counts = []
    file_sizes = []
    
    # RAVDESS filename format: 01-01-06-01-02-01-16.npz
    # [actor]-[modality]-[emotion]-[intensity]-[statement]-[repetition]-[take]
    
    # Analyze files
    for file_path in files:
        # Get actor and emotion from filename
        basename = os.path.basename(file_path)
        parts = basename.split('-')
        
        if len(parts) >= 3:
            actor_id = parts[0]
            emotion_code = parts[2]
            
            actor_ids.append(actor_id)
            emotion_labels.append(emotion_code)
        
        # Get file size
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        file_sizes.append(file_size)
        
        try:
            # Load the npz file
            data = np.load(file_path, allow_pickle=True)
            
            # Count sequences
            if 'video_sequences' in data:
                video_seqs = data['video_sequences']
                num_sequences = len(video_seqs)
                sequence_counts.append(num_sequences)
                total_sequences += num_sequences
                
                # Print the shape of the first sequence
                if num_sequences > 0:
                    first_video_seq = video_seqs[0]
                    first_audio_seq = data['audio_sequences'][0] if 'audio_sequences' in data else None
                    
                    print(f"File: {basename}")
                    print(f"  Video sequence shape: {first_video_seq.shape}")
                    print(f"  Audio sequence shape: {first_audio_seq.shape if first_audio_seq is not None else 'N/A'}")
                    print(f"  Emotion label: {data['emotion_label'] if 'emotion_label' in data else 'N/A'}")
                    print(f"  Number of sequences: {num_sequences}")
                    print("")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Print statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"Total files: {len(files)}")
    print(f"Total sequences: {total_sequences}")
    print(f"Average sequences per file: {total_sequences/len(files) if files else 0:.2f}")
    
    # Emotion distribution
    emotion_counter = Counter(emotion_labels)
    print("\nEmotion Distribution:")
    for emotion, count in sorted(emotion_counter.items()):
        emotion_name = get_emotion_name(emotion)
        print(f"  {emotion} ({emotion_name}): {count} files")
    
    # Actor distribution
    actor_counter = Counter(actor_ids)
    print("\nActor Distribution:")
    for actor, count in sorted(actor_counter.items()):
        print(f"  Actor {actor}: {count} files")
    
    # Sequence count distribution
    if sequence_counts:
        print("\nSequence Count Distribution:")
        seq_counter = Counter(sequence_counts)
        for count, frequency in sorted(seq_counter.items()):
            print(f"  {count} sequences: {frequency} files")
    
    # Plot distributions if matplotlib is available
    try:
        # Plot emotion distribution
        plt.figure(figsize=(12, 6))
        emotions = [f"{e} ({get_emotion_name(e)})" for e in sorted(emotion_counter.keys())]
        counts = [emotion_counter[e] for e in sorted(emotion_counter.keys())]
        
        plt.subplot(1, 2, 1)
        plt.bar(emotions, counts)
        plt.title('Emotion Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Files')
        
        # Plot sequence count distribution
        plt.subplot(1, 2, 2)
        seq_counts = sorted(seq_counter.keys())
        seq_freqs = [seq_counter[c] for c in seq_counts]
        
        plt.bar([str(c) for c in seq_counts], seq_freqs)
        plt.title('Sequences per File')
        plt.xlabel('Number of Sequences')
        plt.ylabel('Number of Files')
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(data_dir, 'dataset_statistics.png')
        plt.savefig(output_file)
        print(f"\nStatistics visualization saved to {output_file}")
        
        plt.close()
    except Exception as e:
        print(f"Could not create visualizations: {str(e)}")

def get_emotion_name(emotion_code):
    """Maps RAVDESS emotion codes to names."""
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    return emotion_map.get(emotion_code, "unknown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze processed RAVDESS dataset')
    parser.add_argument('--data_dir', type=str, default='processed_features_large',
                        help='Directory containing processed features')
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir)
