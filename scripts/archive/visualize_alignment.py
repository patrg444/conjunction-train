#!/usr/bin/env python3
"""
Visualization tool for audio-visual alignment in the multimodal emotion recognition system.
This script helps verify alignment between audio and video features.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def visualize_alignment(audio_features, video_features, aligned_audio, aligned_video, 
                       sr=16000, hop_length_audio=512, frame_rate=30, hop_length_video=1,
                       title="Alignment Check", save_path=None, feature_dims=3):
    """Visualizes the alignment of audio and video features.

    Args:
        audio_features: Original audio features (before alignment).
        video_features: Original video features (before alignment).
        aligned_audio: Aligned audio features.
        aligned_video: Aligned video features.
        sr: Audio sample rate.
        hop_length_audio: Audio hop length.
        frame_rate: Video frame rate.
        hop_length_video: Video hop length.
        title: Plot title.
        save_path: If provided, save visualization to this path.
        feature_dims: Number of feature dimensions to visualize.
    """
    # Calculate time bases
    time_audio_original = np.arange(audio_features.shape[0]) * hop_length_audio / sr
    time_video_original = np.arange(video_features.shape[0]) * hop_length_video / frame_rate
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot original features
    plt.subplot(2, 1, 1)
    
    # Plot multiple audio feature dimensions
    for i in range(min(feature_dims, audio_features.shape[1])):
        plt.plot(time_audio_original, audio_features[:, i], 
                 label=f"Audio Feature {i}", 
                 alpha=0.7, 
                 linestyle='-')
    
    # Plot multiple video feature dimensions
    for i in range(min(feature_dims, video_features.shape[1])):
        plt.plot(time_video_original, video_features[:, i], 
                 label=f"Video Feature {i}", 
                 alpha=0.7, 
                 linestyle='--')
    
    plt.title("Original Features (Before Alignment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot aligned features
    plt.subplot(2, 1, 2)
    
    # Handle the case where audio and video have different lengths
    # This indicates they're not actually aligned in the same array
    if aligned_audio.shape[0] != aligned_video.shape[0]:
        # Create separate time axes for each
        time_aligned_audio = np.arange(aligned_audio.shape[0])
        time_aligned_video = np.arange(aligned_video.shape[0])
        
        # Plot multiple aligned audio feature dimensions
        for i in range(min(feature_dims, aligned_audio.shape[1])):
            plt.plot(time_aligned_audio, aligned_audio[:, i], 
                    label=f"Aligned Audio Feature {i}", 
                    alpha=0.7, 
                    linestyle='-')
        
        # Plot multiple aligned video feature dimensions
        for i in range(min(feature_dims, aligned_video.shape[1])):
            # Normalize the video time axis to match the audio range
            # This helps visualize them together even though they're different lengths
            normalized_time = time_aligned_video * (aligned_audio.shape[0] / aligned_video.shape[0])
            plt.plot(normalized_time, aligned_video[:, i], 
                    label=f"Aligned Video Feature {i}", 
                    alpha=0.7, 
                    linestyle='--')
            
        plt.title(f"'Aligned' Features (WARNING: Different lengths - A:{aligned_audio.shape[0]}, V:{aligned_video.shape[0]})")
    else:
        # Normal case where audio and video are the same length (truly aligned)
        time_aligned = np.arange(aligned_audio.shape[0])
        
        # Plot multiple aligned audio feature dimensions
        for i in range(min(feature_dims, aligned_audio.shape[1])):
            plt.plot(time_aligned, aligned_audio[:, i], 
                    label=f"Aligned Audio Feature {i}", 
                    alpha=0.7, 
                    linestyle='-')
        
        # Plot multiple aligned video feature dimensions
        for i in range(min(feature_dims, aligned_video.shape[1])):
            plt.plot(time_aligned, aligned_video[:, i], 
                    label=f"Aligned Video Feature {i}", 
                    alpha=0.7, 
                    linestyle='--')
        
        plt.title("Aligned Features (Same Length)")
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def visualize_processed_file(npz_file, output_dir="alignment_visualizations"):
    """Visualizes alignment for a processed NPZ file."""
    try:
        # Load data
        data = np.load(npz_file, allow_pickle=True)
        
        # Verify this is a valid processed file
        required_keys = ['video_sequences', 'audio_sequences', 'params']
        if not all(k in data for k in required_keys):
            print(f"Error: {npz_file} is missing required keys. Not a valid processed file.")
            return None
        
        # Extract parameters
        params = data['params'].item() if 'params' in data else {}
        sr = params.get('sr', 16000)
        hop_length_audio = params.get('hop_length_audio', 512)
        frame_rate = params.get('frame_rate', 30)
        hop_length_video = params.get('hop_length_video', 1)
        
        # Get the aligned sequences
        video_sequences = data['video_sequences']
        audio_sequences = data['audio_sequences']
        
        if len(video_sequences) == 0 or len(audio_sequences) == 0:
            print(f"Error: {npz_file} contains empty sequences.")
            return None
        
        # Visualize alignment for first sequence 
        # (we assume all sequences in the file are aligned the same way)
        first_video_seq = video_sequences[0]
        first_audio_seq = audio_sequences[0]
        
        # For demo purposes, we don't have the original unaligned features
        # So we'll simulate them by stretching the aligned features
        # This is just for visualization purposes
        simulated_orig_audio = np.repeat(first_audio_seq, 2, axis=0)[:500]  # Longer audio
        simulated_orig_video = first_video_seq  # Video at original rate
        
        # Create output filename
        basename = os.path.splitext(os.path.basename(npz_file))[0]
        output_path = os.path.join(output_dir, f"{basename}_alignment.png")
        
        # Visualize alignment
        title = f"Alignment Visualization for {basename}\nEmotion: {params.get('emotion_name', 'Unknown')}"
        return visualize_alignment(
            simulated_orig_audio, simulated_orig_video,
            first_audio_seq, first_video_seq,
            sr=sr, hop_length_audio=hop_length_audio,
            frame_rate=frame_rate, hop_length_video=hop_length_video,
            title=title, save_path=output_path
        )
    except Exception as e:
        print(f"Error visualizing alignment for {npz_file}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def visualize_feature_distributions(npz_files, output_dir="alignment_visualizations"):
    """Visualizes distributions of feature values by emotion."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Container for features by emotion
    audio_features_by_emotion = {}
    video_features_by_emotion = {}
    
    print("Loading features for distribution analysis...")
    for npz_file in tqdm(npz_files):
        try:
            # Load data
            data = np.load(npz_file, allow_pickle=True)
            
            # Skip if missing required data
            if 'video_sequences' not in data or 'audio_sequences' not in data or 'emotion_label' not in data:
                continue
            
            # Get emotion label
            emotion_label = data['emotion_label']
            if np.isscalar(emotion_label) or (isinstance(emotion_label, np.ndarray) and emotion_label.size == 1):
                emotion_label = emotion_label.item() if isinstance(emotion_label, np.ndarray) else emotion_label
            else:
                # If there are multiple labels, use the first one
                emotion_label = emotion_label[0] if hasattr(emotion_label, '__iter__') else emotion_label
            
            # Map the numeric label to a name
            emotion_name = {
                0: "neutral", 1: "calm", 2: "happy", 3: "sad",
                4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
            }.get(emotion_label, f"unknown-{emotion_label}")
            
            # Get sequences
            video_seqs = data['video_sequences']
            audio_seqs = data['audio_sequences']
            
            # Skip if empty
            if len(video_seqs) == 0 or len(audio_seqs) == 0:
                continue
            
            # Add to containers
            if emotion_name not in audio_features_by_emotion:
                audio_features_by_emotion[emotion_name] = []
                video_features_by_emotion[emotion_name] = []
            
            # Add features (we'll flatten sequences for distribution analysis)
            for video_seq in video_seqs:
                video_features_by_emotion[emotion_name].append(video_seq.flatten())
            
            for audio_seq in audio_seqs:
                audio_features_by_emotion[emotion_name].append(audio_seq.flatten())
        
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
    
    # Create feature distribution visualizations
    print("Creating feature distribution visualizations...")
    
    # 1. Audio features boxplot by emotion
    plt.figure(figsize=(14, 8))
    audio_data = []
    audio_labels = []
    
    for emotion, features in audio_features_by_emotion.items():
        # Sample values to avoid overwhelming plots
        sampled_values = []
        for feat in features[:10]:  # Limit to 10 sequences per emotion
            sampled_values.extend(feat[:1000])  # Limit to 1000 values per sequence
        
        if sampled_values:
            audio_data.append(sampled_values)
            audio_labels.append(f"{emotion} (n={len(features)})")
    
    plt.boxplot(audio_data, labels=audio_labels)
    plt.title("Audio Feature Distributions by Emotion")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save audio distribution
    audio_dist_path = os.path.join(output_dir, "audio_feature_distributions.png")
    plt.savefig(audio_dist_path, dpi=150)
    plt.close()
    
    # 2. Video features boxplot by emotion
    plt.figure(figsize=(14, 8))
    video_data = []
    video_labels = []
    
    for emotion, features in video_features_by_emotion.items():
        # Sample values to avoid overwhelming plots
        sampled_values = []
        for feat in features[:10]:  # Limit to 10 sequences per emotion
            sampled_values.extend(feat[:1000])  # Limit to 1000 values per sequence
        
        if sampled_values:
            video_data.append(sampled_values)
            video_labels.append(f"{emotion} (n={len(features)})")
    
    plt.boxplot(video_data, labels=video_labels)
    plt.title("Video Feature Distributions by Emotion")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save video distribution
    video_dist_path = os.path.join(output_dir, "video_feature_distributions.png")
    plt.savefig(video_dist_path, dpi=150)
    plt.close()
    
    print(f"Feature distribution visualizations saved to {output_dir}")
    return audio_dist_path, video_dist_path

def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description='Visualize audio-visual alignment in processed data.')
    parser.add_argument('--data-dir', type=str, default='processed_features_2s',
                       help='Directory containing processed features (default: processed_features_2s)')
    parser.add_argument('--output-dir', type=str, default='alignment_visualizations',
                       help='Directory to save visualizations (default: alignment_visualizations)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of files to visualize (default: 5)')
    parser.add_argument('--analyze-distributions', action='store_true',
                       help='Analyze feature distributions by emotion')
    
    args = parser.parse_args()
    
    # Get list of NPZ files
    import glob
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files in {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize alignment for random samples
    import random
    random.seed(42)  # For reproducibility
    
    if len(npz_files) > args.num_samples:
        files_to_visualize = random.sample(npz_files, args.num_samples)
    else:
        files_to_visualize = npz_files
    
    print(f"Visualizing alignment for {len(files_to_visualize)} files...")
    for file_path in files_to_visualize:
        output_path = visualize_processed_file(file_path, args.output_dir)
        if output_path:
            print(f"Created visualization: {output_path}")
    
    # Analyze feature distributions if requested
    if args.analyze_distributions:
        print("Analyzing feature distributions by emotion...")
        visualize_feature_distributions(npz_files, args.output_dir)

if __name__ == "__main__":
    main()
