#!/usr/bin/env python3
"""
Test script for complete facial feature extraction using FaceNetExtractor.
This script demonstrates extracting and saving facial embeddings from a RAVDESS video.
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from facenet_extractor import FaceNetExtractor

def extract_and_save_features(video_path, output_path=None, sample_interval=1):
    """
    Extract facial features from a video and save them to a NPZ file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the NPZ file (if None, uses video filename with .npz extension)
        sample_interval: Process every Nth frame
    
    Returns:
        Path to the saved NPZ file
    """
    print(f"Extracting features from: {os.path.basename(video_path)}")
    
    # Initialize extractor
    extractor = FaceNetExtractor()
    
    # Extract features
    features, timestamps = extractor.process_video(video_path, sample_interval)
    
    if features is None or len(features) == 0:
        print("No features were extracted!")
        return None
    
    # Calculate statistics
    num_frames = len(features)
    num_nonzero_frames = np.sum(np.any(features != 0, axis=1))
    avg_value = np.mean(features[np.any(features != 0, axis=1)])
    std_value = np.std(features[np.any(features != 0, axis=1)])
    
    print(f"Feature extraction summary:")
    print(f"  Total frames processed: {num_frames}")
    print(f"  Frames with detected faces: {num_nonzero_frames} ({num_nonzero_frames/num_frames*100:.1f}%)")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Average feature value: {avg_value:.4f}")
    print(f"  Feature standard deviation: {std_value:.4f}")
    
    # Save to NPZ file
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_facenet_features.npz"
    
    # Create a dictionary with all the data we want to save
    data_dict = {
        'video_features': features,
        'timestamps': timestamps,
        'video_path': video_path,
        'sample_interval': sample_interval,
    }
    
    np.savez(output_path, **data_dict)
    print(f"Features saved to: {output_path}")
    
    return output_path

def visualize_features(npz_path):
    """
    Visualize the extracted features from an NPZ file.
    
    Args:
        npz_path: Path to the NPZ file containing features
    """
    print(f"Visualizing features from: {npz_path}")
    
    # Load the NPZ file
    data = np.load(npz_path)
    features = data['video_features']
    timestamps = data['timestamps']
    
    # Plot a heatmap of the first few frames' features
    n_frames = min(5, len(features))
    n_features = min(50, features.shape[1])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.imshow(features[:n_frames, :n_features], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.ylabel('Frame')
    plt.xlabel('Feature Dimension (first 50)')
    plt.title('Facial Embeddings Heatmap (First 5 Frames)')
    
    # Plot average feature value over time
    plt.subplot(212)
    mean_values = np.mean(features, axis=1)
    plt.plot(timestamps, mean_values)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Feature Value')
    plt.title('Average Feature Value Over Time')
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.splitext(npz_path)[0] + '_visualization.png'
    plt.savefig(fig_path)
    print(f"Visualization saved to: {fig_path}")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Extract facial features from a video using FaceNet")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--output", help="Path to save the NPZ file (optional)")
    parser.add_argument("--interval", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the extracted features")
    parser.add_argument("--ravdess-sample", action="store_true", help="Use a sample RAVDESS video")
    
    args = parser.parse_args()
    
    video_path = args.video
    
    if args.ravdess_sample:
        import glob
        ravdess_videos = glob.glob("downsampled_videos/RAVDESS/*/*.mp4")
        if ravdess_videos:
            video_path = ravdess_videos[0]
            print(f"Selected RAVDESS sample: {video_path}")
        else:
            print("No RAVDESS videos found in 'downsampled_videos/RAVDESS/'")
            return
    
    if not video_path:
        print("Please provide a video path with --video or use --ravdess-sample")
        return
    
    # Extract and save features
    npz_path = extract_and_save_features(video_path, args.output, args.interval)
    
    if npz_path and args.visualize:
        visualize_features(npz_path)

if __name__ == "__main__":
    main()
