#!/usr/bin/env python3
"""
Test script for multimodal feature extraction with audio-video synchronization.
This script demonstrates how to use the multimodal_preprocess module to extract
synchronized features from video files.
"""

import os
import sys
import glob
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_preprocess import (
    extract_audio_from_video,
    extract_frame_level_video_features,
    extract_frame_level_audio_features,
    align_audio_video_features,
    create_sequences,
    process_video_for_multimodal_lstm,
    process_dataset_videos
)

def parse_ravdess_filename(video_path):
    """Parses a RAVDESS filename and extracts emotion information.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        A dictionary containing the emotion label (text and numeric) or None if parsing fails.
    """
    filename = os.path.basename(video_path)
    parts = filename.split('-')
    
    if len(parts) != 7:
        return None
    
    try:
        modality = int(parts[0])
        vocal_channel = int(parts[1])
        emotion = int(parts[2])
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor = int(parts[6].split('.')[0])  # Remove extension

        emotion_mapping = {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fearful',
            7: 'disgust',
            8: 'surprised'
        }
        
        numeric_mapping = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5,
            'disgust': 6,
            'surprised': 7
        }

        emotion_label = emotion_mapping.get(emotion)
        if emotion_label:
            return {
                'text': emotion_label,
                'numeric': numeric_mapping[emotion_label]
            }
        else:
            return None
            
    except ValueError:
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("synchronize_test.log"),
        logging.StreamHandler()
    ]
)

def test_audio_extraction(video_path, output_dir="temp_extracted_audio"):
    """Test audio extraction from a video file."""
    logging.info(f"Testing audio extraction from {video_path}")
    
    audio_path = extract_audio_from_video(video_path, output_dir)
    
    if audio_path and os.path.exists(audio_path):
        logging.info(f"Successfully extracted audio to {audio_path}")
        return True, audio_path
    else:
        logging.error(f"Failed to extract audio from {video_path}")
        return False, None

def test_feature_extraction(video_path, model_name="VGG-Face"):
    """Test feature extraction from video and audio."""
    logging.info(f"Testing feature extraction with model {model_name}")
    
    # 1. Extract audio
    success, audio_path = test_audio_extraction(video_path)
    if not success:
        return False
    
    # 2. Extract video features
    video_features, video_timestamps = extract_frame_level_video_features(
        video_path, model_name=model_name
    )
    
    if video_features is None:
        logging.error(f"Failed to extract video features from {video_path}")
        return False
    
    logging.info(f"Extracted {len(video_features)} video frames with features of dimension {len(video_features[0])}")
    
    # 3. Extract audio features
    audio_features, audio_timestamps = extract_frame_level_audio_features(audio_path)
    
    if audio_features is None:
        logging.error(f"Failed to extract audio features from {audio_path}")
        return False
    
    logging.info(f"Extracted {len(audio_features)} audio frames with features of dimension {audio_features.shape[1]}")
    
    # 4. Align features
    aligned_features = align_audio_video_features(
        video_features, video_timestamps,
        audio_features, audio_timestamps
    )
    
    if aligned_features is None or len(aligned_features) == 0:
        logging.error(f"Failed to align features")
        return False
    
    logging.info(f"Created {len(aligned_features)} aligned feature vectors of dimension {aligned_features.shape[1]}")
    
    # 5. Create sequences
    sequences, sequence_lengths = create_sequences(aligned_features)
    
    if sequences is None:
        logging.error(f"Failed to create sequences")
        return False
    
    logging.info(f"Created {len(sequences)} sequences of length {sequences.shape[1]}")
    
    return True

def visualize_alignment(npz_file):
    """Visualize the alignment of audio and video features."""
    data = np.load(npz_file)
    sequences = data['sequences']
    
    # Assuming first part is video features and second part is audio features
    # We'll plot the first few dimensions of each to visualize synchronization
    n_sequences = min(5, sequences.shape[0])
    
    # Determine feature split point (video vs audio)
    feature_dim = sequences.shape[2]
    split_point = feature_dim // 2  # Estimate, adapt as needed
    
    plt.figure(figsize=(15, 10))
    
    for i in range(n_sequences):
        sequence = sequences[i]
        
        # Plot a few dimensions of video features
        plt.subplot(n_sequences, 2, i*2 + 1)
        video_features = sequence[:, :3]  # First 3 dimensions of video features
        plt.plot(video_features)
        plt.title(f"Sequence {i+1}: Video Features (first 3 dims)")
        plt.xlabel("Time")
        plt.ylabel("Feature Value")
        
        # Plot a few dimensions of audio features
        plt.subplot(n_sequences, 2, i*2 + 2)
        audio_features = sequence[:, split_point:split_point+3]  # First 3 dimensions of audio features
        plt.plot(audio_features)
        plt.title(f"Sequence {i+1}: Audio Features (first 3 dims)")
        plt.xlabel("Time")
        plt.ylabel("Feature Value")
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.dirname(npz_file)
    fig_path = os.path.join(output_dir, f"{os.path.basename(npz_file).replace('.npz', '_visualization.png')}")
    plt.savefig(fig_path)
    logging.info(f"Saved visualization to {fig_path}")
    
    plt.close()
    
    return fig_path

def test_process_single_video(video_path, output_dir="processed_features_test"):
    """Test the complete processing pipeline on a single video."""
    logging.info(f"Testing complete pipeline on {video_path}")

    output_file = process_video_for_multimodal_lstm(
        video_path=video_path,
        output_dir=output_dir
    )
    
    if output_file and os.path.exists(output_file):
        logging.info(f"Successfully processed video to {output_file}")
        
        # Load the processed data to verify
        data = np.load(output_file)
        sequences = data['sequences']
        sequence_lengths = data['sequence_lengths']
        
        logging.info(f"Processed data shape: {sequences.shape}")
        logging.info(f"Sequence lengths: min={sequence_lengths.min()}, max={sequence_lengths.max()}")
        
        # Visualize the alignment
        try:
            fig_path = visualize_alignment(output_file)
            logging.info(f"Created visualization at {fig_path}")
        except Exception as e:
            logging.error(f"Failed to create visualization: {str(e)}")
        
        return True, output_file
    else:
        logging.error(f"Failed to process video {video_path}")
        return False, None

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        video_dir = sys.argv[1]
    else:
        video_dir = "data/RAVDESS"
    

    # Make sure the video directory exists
    if not os.path.exists(video_dir):
        logging.error(f"Video directory not found: {video_dir}")
        return False

    # Get all RAVDESS video files
    video_paths = glob.glob(os.path.join(video_dir, "Actor_*", "*.mp4"))
    
    if not video_paths:
        logging.error(f"No videos found in {video_dir}")
        return False

    logging.info(f"Found {len(video_paths)} videos for testing")
    
    # Create output directory
    output_dir = "processed_features_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test the complete pipeline on each video
    results = []
    for video_path in video_paths:
        # Parse filename to get emotion label
        emotion_info = parse_ravdess_filename(video_path)
        
        if emotion_info:
            logging.info(f"Processing: {video_path} - Emotion: {emotion_info['text']}")
            success, output_file = test_process_single_video(video_path, output_dir)
            results.append((video_path, success, output_file, emotion_info['text']))
        else:
            logging.warning(f"Skipping {video_path} due to filename parsing error.")
            results.append((video_path, False, None, None))

    # Print summary
    logging.info("=== Test Summary ===")
    for video_path, success, output_file, emotion in results:
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"{status}: {video_path} - Emotion: {emotion}")
        if success:
            logging.info(f"  Output: {output_file}")

    # Count successes
    success_count = sum(1 for _, success, _, _ in results if success)
    logging.info(f"Successfully processed {success_count} out of {len(results)} videos")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
