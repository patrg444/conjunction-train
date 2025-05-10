#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract 3D CNN features from RAVDESS and CREMA-D datasets.
This script batch processes videos and saves extracted features.
"""

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# Add the parent directory to the path so we can import from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import EmotionDatasetManager
from cnn3d.feature_extractor import CNN3DFeatureExtractor

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Extract 3D CNN features from video datasets')
    
    parser.add_argument('--ravdess_dir', type=str, default='./downsampled_videos/RAVDESS',
                      help='Path to RAVDESS dataset directory')
    parser.add_argument('--cremad_dir', type=str, default='./downsampled_videos/CREMA-D-audio-complete',
                      help='Path to CREMA-D dataset directory')
    parser.add_argument('--output_dir', type=str, default='./processed_data/cnn3d',
                      help='Directory to save extracted features')
    parser.add_argument('--face_detector', type=str, default=None,
                      help='Path to face detector model (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--visualize', action='store_true',
                      help='Save visualizations of processed frames')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from previous extraction (skip existing files)')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Process only a sample of videos (for testing)')
    
    return parser.parse_args()

def setup_gpu():
    """Configure GPU settings for TensorFlow."""
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {len(gpus)} GPU(s) detected")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPU found. Using CPU.")

def create_dataset_manager(args):
    """Create and initialize the dataset manager."""
    manager = EmotionDatasetManager(
        ravdess_dir=args.ravdess_dir,
        cremad_dir=args.cremad_dir,
        output_dir=args.output_dir
    )
    
    # Load or create dataset splits
    if not manager.load_splits():
        print("Creating new dataset splits...")
        manager.analyze_class_distribution()
        manager.create_dataset_splits()
    
    return manager

def create_feature_extractor(args):
    """Create the 3D CNN feature extractor."""
    return CNN3DFeatureExtractor(
        input_shape=(16, 112, 112, 3),
        feature_dim=256,
        face_detector_path=args.face_detector
    )

def extract_features(extractor, video_paths, output_dir, visualize=False, resume=False, batch_size=32):
    """
    Extract features from video files and save to disk.
    
    Args:
        extractor: Feature extractor model
        video_paths: List of video paths
        output_dir: Output directory for features
        visualize: Whether to save visualizations
        resume: Whether to skip existing files
        batch_size: Batch size for processing
    """
    features_dir = os.path.join(output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    if visualize:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    # Track successes, failures, and skips
    results = {'success': 0, 'failed': 0, 'skipped': 0}
    failures = []
    
    # Process videos
    for i, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
        try:
            # Create output filename
            base_name = os.path.basename(video_path)
            base_name = os.path.splitext(base_name)[0]
            output_path = os.path.join(features_dir, f"{base_name}.npy")
            
            # Skip if file exists and resume is True
            if resume and os.path.exists(output_path):
                results['skipped'] += 1
                continue
            
            # Extract features
            features = extractor.extract_features(
                video_path, 
                face_crop=(args.face_detector is not None),
                normalize_frames=True,
                save_visualization=(visualize and i % 50 == 0),  # Visualize every 50th sample
                output_dir=viz_dir if visualize else None
            )
            
            if features is not None:
                # Save features
                np.save(output_path, features)
                results['success'] += 1
                
                # Print progress every 100 files
                if (i + 1) % 100 == 0:
                    print(f"Progress: {i+1}/{len(video_paths)} - "
                         f"Success: {results['success']}, "
                         f"Failed: {results['failed']}, "
                         f"Skipped: {results['skipped']}")
            else:
                results['failed'] += 1
                failures.append(video_path)
                
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results['failed'] += 1
            failures.append(f"{video_path}: {str(e)}")
    
    # Save extraction summary
    summary_path = os.path.join(output_dir, "extraction_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Extraction Summary:\n")
        f.write(f"-----------------\n")
        f.write(f"Total files: {len(video_paths)}\n")
        f.write(f"Successful extractions: {results['success']}\n")
        f.write(f"Failed extractions: {results['failed']}\n")
        f.write(f"Skipped files: {results['skipped']}\n\n")
        
        if failures:
            f.write("Failed Files:\n")
            f.write("------------\n")
            for failure in failures:
                f.write(f"{failure}\n")
    
    print(f"\nExtraction complete!")
    print(f"Total: {len(video_paths)}, Success: {results['success']}, "
         f"Failed: {results['failed']}, Skipped: {results['skipped']}")
    print(f"Summary saved to {summary_path}")
    
    return results

def main(args):
    """Main function to run the feature extraction process."""
    setup_gpu()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset manager and feature extractor
    dataset_manager = create_dataset_manager(args)
    feature_extractor = create_feature_extractor(args)
    
    # Get video paths
    train_files = dataset_manager.train_files
    val_files = dataset_manager.val_files
    test_files = dataset_manager.test_files
    
    # Take a sample for testing if requested
    if args.sample_size:
        all_files = train_files + val_files + test_files
        if args.sample_size < len(all_files):
            np.random.seed(42)  # For reproducibility
            sample_idx = np.random.choice(len(all_files), args.sample_size, replace=False)
            video_paths = [all_files[i] for i in sample_idx]
            print(f"Using a sample of {args.sample_size} videos for testing")
        else:
            video_paths = all_files
            print(f"Sample size {args.sample_size} >= total videos {len(all_files)}, using all videos")
    else:
        video_paths = train_files + val_files + test_files
    
    # Extract features
    print(f"Starting feature extraction for {len(video_paths)} videos")
    print(f"Output directory: {args.output_dir}")
    
    results = extract_features(
        feature_extractor, 
        video_paths, 
        args.output_dir,
        visualize=args.visualize,
        resume=args.resume,
        batch_size=args.batch_size
    )
    
    # Create metadata about the extraction
    metadata = {
        'feature_dim': feature_extractor.feature_dim,
        'input_shape': feature_extractor.input_shape,
        'num_train': len(train_files),
        'num_val': len(val_files),
        'num_test': len(test_files),
        'extraction_results': results
    }
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("Feature Extraction Metadata:\n")
        f.write("--------------------------\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Feature extraction complete. Metadata saved to {metadata_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
