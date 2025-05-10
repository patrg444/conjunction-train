#!/usr/bin/env python3
"""
Main script for processing the CREMA-D dataset using synchronized audio-video feature extraction.
This script extracts audio directly from video files, processes both modalities in parallel,
and creates synchronized multimodal features for LSTM training.
"""

import os
import sys
import argparse
import logging
import glob
from tqdm import tqdm
import numpy as np

# Import our multimodal preprocessing functions
from multimodal_preprocess import (
    process_video_for_multimodal_lstm,
    process_dataset_videos
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_crema_d.log"),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process CREMA-D dataset with synchronized audio-video features')
    
    parser.add_argument('--video-dir', type=str, default='data/CREMA-D/VideoFlash',
                        help='Directory containing CREMA-D video files')
    parser.add_argument('--config-file', type=str, default='opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf',
                        help='Path to openSMILE configuration file')
    parser.add_argument('--output-dir', type=str, default='processed_features_crema_d',
                        help='Directory to save processed features')
    parser.add_argument('--model-name', type=str, default='VGG-Face',
                        help='DeepFace model to use (VGG-Face, Facenet, Facenet512, OpenFace, ArcFace)')
    parser.add_argument('--window-size', type=float, default=1.0,
                        help='Time window size in seconds for feature alignment')
    parser.add_argument('--hop-size', type=float, default=0.5,
                        help='Time window hop size in seconds for feature alignment')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Number of frames in each LSTM sequence')
    parser.add_argument('--sequence-overlap', type=int, default=15,
                        help='Number of frames to overlap between sequences')
    parser.add_argument('--opensmile-path', type=str, default=None,
                        help='Path to openSMILE executable (optional)')
    parser.add_argument('--sample-count', type=int, default=None,
                        help='Number of videos to process (for testing, None=all)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes (1=sequential)')
    
    return parser.parse_args()

def extract_emotion_labels(video_paths):
    """Extract emotion labels from CREMA-D video filenames.
    
    Args:
        video_paths: List of paths to CREMA-D video files
        
    Returns:
        Dictionary mapping video filenames to emotion labels
    """
    # CREMA-D emotion code to label mapping
    emotion_mapping = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    # Numeric mapping for labels (for models)
    numeric_mapping = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5
    }
    
    labels = {}
    
    for video_path in video_paths:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # CREMA-D filename format: 1076_MTI_SAD_XX.flv
        parts = filename.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_mapping:
                emotion_label = emotion_mapping[emotion_code]
                labels[filename] = {
                    'text': emotion_label,
                    'numeric': numeric_mapping[emotion_label]
                }
            else:
                logging.warning(f"Unknown emotion code in filename: {filename}")
        else:
            logging.warning(f"Filename does not match expected format: {filename}")
    
    return labels

def save_metadata(video_paths, labels, output_dir):
    """Save metadata about processed videos and their labels.
    
    Args:
        video_paths: List of paths to processed video files
        labels: Dictionary of labels for each video
        output_dir: Directory to save metadata
    """
    metadata = []
    
    for video_path in video_paths:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        feature_file = os.path.join(output_dir, f"{filename}.npz")
        
        if os.path.exists(feature_file) and filename in labels:
            metadata.append({
                'video_path': video_path,
                'feature_file': feature_file,
                'filename': filename,
                'emotion': labels[filename]['text'],
                'emotion_code': labels[filename]['numeric']
            })
    
    # Convert to numpy structured array for easier loading
    dtype = [
        ('video_path', 'U200'),
        ('feature_file', 'U200'),
        ('filename', 'U50'),
        ('emotion', 'U10'),
        ('emotion_code', 'i4')
    ]
    
    metadata_array = np.array(
        [(item['video_path'], item['feature_file'], item['filename'], 
          item['emotion'], item['emotion_code']) for item in metadata],
        dtype=dtype
    )
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.npz")
    np.savez_compressed(metadata_file, metadata=metadata_array)
    
    logging.info(f"Saved metadata for {len(metadata)} videos to {metadata_file}")
    
    # Also save a simple CSV for easy inspection
    csv_file = os.path.join(output_dir, "metadata.csv")
    with open(csv_file, 'w') as f:
        f.write("filename,emotion,emotion_code,feature_file\n")
        for item in metadata:
            f.write(f"{item['filename']},{item['emotion']},{item['emotion_code']},{item['feature_file']}\n")
    
    logging.info(f"Saved CSV metadata to {csv_file}")
    
    # Print summary of emotions
    emotion_counts = {}
    for item in metadata:
        emotion = item['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    logging.info("Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        logging.info(f"  {emotion}: {count} videos ({count/len(metadata)*100:.1f}%)")

def process_crema_d(args):
    """Process the CREMA-D dataset with synchronized audio-video features.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of paths to processed feature files
    """
    # Verify that the video directory exists
    if not os.path.exists(args.video_dir):
        logging.error(f"Video directory not found: {args.video_dir}")
        return []
    
    # Find video files
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.flv")))
    if not video_paths:
        logging.error(f"No .flv video files found in {args.video_dir}")
        return []
    
    # Limit sample count if specified
    if args.sample_count is not None:
        video_paths = video_paths[:args.sample_count]
    
    logging.info(f"Processing {len(video_paths)} videos from {args.video_dir}")
    
    # Extract emotion labels from filenames
    labels = extract_emotion_labels(video_paths)
    logging.info(f"Extracted {len(labels)} emotion labels")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process videos
    if args.workers <= 1:
        # Process sequentially
        output_files = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            output_file = process_video_for_multimodal_lstm(
                video_path=video_path,
                config_file=args.config_file,
                output_dir=args.output_dir,
                model_name=args.model_name,
                window_size=args.window_size,
                hop_size=args.hop_size,
                sequence_length=args.sequence_length,
                sequence_overlap=args.sequence_overlap,
                opensmile_path=args.opensmile_path
            )
            if output_file:
                output_files.append(output_file)
    else:
        # Process in parallel
        output_files = process_dataset_videos(
            video_dir=args.video_dir,
            pattern="*.flv",
            config_file=args.config_file,
            output_dir=args.output_dir,
            n_workers=args.workers,
            model_name=args.model_name,
            window_size=args.window_size,
            hop_size=args.hop_size,
            sequence_length=args.sequence_length,
            sequence_overlap=args.sequence_overlap,
            opensmile_path=args.opensmile_path
        )
    
    # Save metadata about processed videos and labels
    save_metadata(video_paths, labels, args.output_dir)
    
    # Print summary
    logging.info(f"Successfully processed {len(output_files)} out of {len(video_paths)} videos")
    logging.info(f"Output directory: {args.output_dir}")
    
    return output_files

def main():
    """Main entry point."""
    args = parse_arguments()
    
    logging.info("=== CREMA-D Multimodal Processing ===")
    logging.info(f"Video directory: {args.video_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Window size: {args.window_size}s, Hop size: {args.hop_size}s")
    logging.info(f"Sequence length: {args.sequence_length}, Overlap: {args.sequence_overlap}")
    
    output_files = process_crema_d(args)
    
    if output_files:
        logging.info(f"Processing completed successfully. Processed {len(output_files)} videos.")
        return 0
    else:
        logging.error("Processing failed or no videos were processed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
