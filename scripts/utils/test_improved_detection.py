#!/usr/bin/env python3
"""
Test script to verify the improved face detection using different detector backends.
This script processes a single file from each dataset and compares the results.
"""

import os
import sys
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from multimodal_preprocess_fixed import process_video_for_single_segment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_improved_detection.log"),
        logging.StreamHandler()
    ]
)

def process_with_detector(video_path, output_dir, detector_backend="mtcnn"):
    """Process a video file using a specific detector backend."""
    detector_output_dir = os.path.join(output_dir, detector_backend)
    os.makedirs(detector_output_dir, exist_ok=True)
    
    output_file = process_video_for_single_segment(
        video_path=video_path,
        output_dir=detector_output_dir,
        model_name="VGG-Face",
        detector_backend=detector_backend
    )
    
    return output_file

def analyze_results(output_files, output_dir):
    """Analyze and compare results from different detector backends."""
    logging.info("Analyzing results from different detector backends...")
    
    # Load the NPZ files
    results = {}
    for detector, file_path in output_files.items():
        if file_path and os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                video_features = data['video_features']
                
                # Calculate sparsity (percentage of zeros)
                sparsity = np.mean(video_features == 0) * 100
                
                results[detector] = {
                    'file': file_path,
                    'shape': video_features.shape,
                    'sparsity': sparsity,
                    'nonzero_count': np.count_nonzero(video_features),
                    'frame_count': video_features.shape[0]
                }
                
                logging.info(f"{detector}: Processed {video_features.shape[0]} frames, {sparsity:.2f}% zeros")
            except Exception as e:
                logging.error(f"Error analyzing {file_path}: {str(e)}")
        else:
            logging.warning(f"No output file for {detector}")
    
    # Create comparison plots
    create_comparison_plots(results, output_dir)
    
    return results

def create_comparison_plots(results, output_dir):
    """Create plots comparing the results from different detector backends."""
    if not results:
        logging.error("No results to plot")
        return
    
    # Plot sparsity comparison
    plt.figure(figsize=(10, 6))
    detectors = list(results.keys())
    sparsities = [results[d]['sparsity'] for d in detectors]
    
    # Sort by sparsity
    sorted_indices = np.argsort(sparsities)
    detectors = [detectors[i] for i in sorted_indices]
    sparsities = [sparsities[i] for i in sorted_indices]
    
    plt.bar(detectors, sparsities)
    plt.title('Feature Sparsity by Detector Backend')
    plt.xlabel('Detector')
    plt.ylabel('Sparsity (% zeros)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'detector_sparsity_comparison.png')
    plt.savefig(plot_path)
    logging.info(f"Saved sparsity comparison plot to {plot_path}")
    
    # Plot frame count comparison
    plt.figure(figsize=(10, 6))
    frame_counts = [results[d]['frame_count'] for d in detectors]
    nonzero_counts = [results[d]['nonzero_count'] // results[d]['frame_count'] for d in detectors]
    
    # Create a side-by-side bar chart
    x = np.arange(len(detectors))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, frame_counts, width, label='Total Frames')
    ax.bar(x + width/2, nonzero_counts, width, label='Non-zero Features per Frame')
    
    ax.set_title('Face Detection Effectiveness by Detector Backend')
    ax.set_xlabel('Detector')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'detector_frame_comparison.png')
    plt.savefig(plot_path)
    logging.info(f"Saved frame count comparison plot to {plot_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test face detection with different detector backends")
    parser.add_argument("--ravdess-video", 
                      help="RAVDESS video file for testing")
    parser.add_argument("--crema-d-video", 
                      help="CREMA-D video file for testing")
    parser.add_argument("--output-dir", default="test_detector_results",
                      help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detectors to test
    detectors = ["opencv", "mtcnn", "retinaface"]
    
    # Process RAVDESS video if provided
    if args.ravdess_video:
        if not os.path.exists(args.ravdess_video):
            logging.error(f"RAVDESS video not found: {args.ravdess_video}")
        else:
            logging.info(f"Processing RAVDESS video: {args.ravdess_video}")
            ravdess_output_dir = os.path.join(args.output_dir, "ravdess")
            os.makedirs(ravdess_output_dir, exist_ok=True)
            
            ravdess_results = {}
            for detector in detectors:
                logging.info(f"Testing detector: {detector}")
                output_file = process_with_detector(
                    args.ravdess_video, 
                    ravdess_output_dir, 
                    detector_backend=detector
                )
                ravdess_results[detector] = output_file
            
            # Analyze results
            analyze_results(ravdess_results, ravdess_output_dir)
    
    # Process CREMA-D video if provided
    if args.crema_d_video:
        if not os.path.exists(args.crema_d_video):
            logging.error(f"CREMA-D video not found: {args.crema_d_video}")
        else:
            logging.info(f"Processing CREMA-D video: {args.crema_d_video}")
            crema_d_output_dir = os.path.join(args.output_dir, "crema_d")
            os.makedirs(crema_d_output_dir, exist_ok=True)
            
            crema_d_results = {}
            for detector in detectors:
                logging.info(f"Testing detector: {detector}")
                output_file = process_with_detector(
                    args.crema_d_video, 
                    crema_d_output_dir, 
                    detector_backend=detector
                )
                crema_d_results[detector] = output_file
            
            # Analyze results
            analyze_results(crema_d_results, crema_d_output_dir)
    
    if not args.ravdess_video and not args.crema_d_video:
        logging.error("No video files provided. Please specify at least one video file.")
        parser.print_help()

if __name__ == "__main__":
    main()
