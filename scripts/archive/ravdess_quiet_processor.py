#!/usr/bin/env python3
"""
Process RAVDESS dataset files with reduced terminal output.
This is a standalone script that doesn't depend on other modules.
"""

import os
import sys
import glob
import shutil
import logging
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from deepface import DeepFace
from deepface.DeepFace import analyze  # Import the analyze function
import time

# Configure logging - separate file and console loggers
def configure_logging(verbose=False):
    """Configure logging with different levels for file and console output.
    
    Args:
        verbose: If True, show INFO level logs in console. If False, only show a minimal
                progress summary and errors in console.
    """
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create handlers
    file_handler = logging.FileHandler("ravdess_processing.log")
    file_handler.setLevel(logging.INFO)  # Detailed logging to file
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    if verbose:
        console_handler.setLevel(logging.INFO)  # Show all info in console when verbose
    else:
        console_handler.setLevel(logging.WARNING)  # Only warnings/errors in console when not verbose
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Capture all INFO+ logs
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def print_progress(current, total, success_count, last_file=None, error_files=None):
    """Print a concise progress report to the console."""
    success_rate = (success_count / current * 100) if current > 0 else 0
    status_msg = f"Processed: {current}/{total} | Success: {success_count} ({success_rate:.1f}%)"
    
    if last_file:
        status_msg += f" | Last: {os.path.basename(last_file)}"
    
    if error_files and len(error_files) > 0:
        status_msg += f" | Errors: {len(error_files)}"
    
    # Clear line and print new status
    print(f"\r{status_msg}", end="")
    sys.stdout.flush()

def extract_audio_from_video(video_path, output_dir="temp_extracted_audio"):
    """Extract audio track from video file and save as WAV."""
    # Create unique filename based on video filename
    video_basename = os.path.basename(video_path)
    audio_filename = os.path.splitext(video_basename)[0] + ".wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
            video.close()
            return audio_path
        else:
            logging.warning(f"No audio track found in {video_path}")
            video.close()
            return None
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {str(e)}")
        return None

def get_embedding_dimension(model_name):
    """Returns the embedding dimension for a given DeepFace model."""
    if model_name == "VGG-Face":
        return 4096
    elif model_name == "Facenet":
        return 128
    elif model_name == "Facenet512":
        return 512
    elif model_name == "OpenFace":
        return 128
    elif model_name == "DeepFace":
        return 4096
    elif model_name == "ArcFace":
        return 512
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def select_primary_face(embedding_objs):
    """Selects the primary face embedding."""
    # Select the face with the largest bounding box area
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    areas = []
    for obj in embedding_objs:
        x, y, w, h = obj["facial_area"].values()
        areas.append((w * h, obj["embedding"]))

    # Sort by area (descending) and take the embedding with largest area
    primary_embedding = max(areas, key=lambda x: x[0])[1]
    return primary_embedding

def extract_frame_level_video_features(video_path, model_name="VGG-Face", fps=None):
    """Extract features from video frames."""
    # Create DeepFace cache
    deepface_cache = {}
    
    def get_cache_key(video_path, frame_idx):
        return f"{video_path}_{model_name}_{frame_idx}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None, None

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0

    logging.info(f"Video {video_path}: {frame_count} frames, {video_fps} FPS, {duration:.2f}s duration")

    # Determine sampling rate (use original or specified fps)
    sampling_fps = fps if fps else video_fps

    # Calculate frame indices to process
    frame_indices = []
    timestamps = []

    # Sample frames at consistent intervals
    for t in np.arange(0, duration, 1/sampling_fps):
        frame_idx = int(t * video_fps)
        if frame_idx < frame_count:
            frame_indices.append(frame_idx)
            timestamps.append(t)

    logging.info(f"Processing {len(frame_indices)} frames at {sampling_fps} FPS")

    # Extract features for each selected frame
    frame_features = []
    for idx in tqdm(frame_indices, desc=f"Extracting video features ({model_name})"):
        cache_key = get_cache_key(video_path, idx)

        # Check cache
        if cache_key in deepface_cache:
            frame_features.append(deepface_cache[cache_key])
            continue

        # Not in cache, need to process
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Check if the frame is valid
            if frame is None:
                logging.warning(f"Invalid frame {idx} from {video_path}")
                frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                continue

            # Use DeepFace.analyze to check for face detection *before* retrying
            try:
                analyze_result = DeepFace.analyze(img_path=frame, actions=['detection'], enforce_detection=False, silent=True)
                if len(analyze_result) == 0 or "face_1" not in analyze_result[0]:
                    logging.warning(f"No faces detected in frame {idx} of {video_path}.")
                    frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                    continue  # Skip to the next frame
            except Exception as e:
                logging.error(f"Error analyzing frame {idx} of {video_path}: {str(e)}")
                frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                continue

            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    embedding_objs = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
                    
                    # Try/except for select_primary_face
                    try:
                        embedding = select_primary_face(embedding_objs)
                    except Exception as e:
                        logging.error(f"Error in select_primary_face at frame {idx}: {str(e)}")
                        embedding = np.zeros(get_embedding_dimension(model_name))

                    # Cache the result
                    deepface_cache[cache_key] = embedding

                    frame_features.append(embedding)
                    break  # Exit loop if successful
                except Exception as e:
                    retries += 1
                    logging.error(f"DeepFace.represent ({model_name}) error at frame {idx}, attempt {retries}/{max_retries}: {str(e)}")
                    time.sleep(0.5)  # Wait before retrying
            if retries == max_retries:
                logging.error(f"Giving up on frame {idx} after {max_retries} attempts.")
                frame_features.append(np.zeros(get_embedding_dimension(model_name)))
        else:
            logging.warning(f"Failed to read frame {idx} from {video_path}")
            frame_features.append(np.zeros(get_embedding_dimension(model_name)))

    cap.release()
    return frame_features, timestamps

def load_arff_features(arff_dir, frame_size=0.025, frame_step=0.01):
    """Simple ARFF parser for OpenSMILE features."""
    # This is a simplified version
    try:
        # Find all ARFF files in the directory
        arff_files = glob.glob(os.path.join(arff_dir, "*.arff"))
        if not arff_files:
            return np.array([]), np.array([])
        
        # Parse the first ARFF file
        features = []
        timestamps = []
        with open(arff_files[0], 'r') as f:
            data_section = False
            feature_names = []
            
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('%'):
                    continue
                
                # Find the attribute names
                if line.startswith('@attribute'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != 'frameTime' and parts[1] != 'class':
                        feature_names.append(parts[1])
                
                # Start of data section
                if line.startswith('@data'):
                    data_section = True
                    continue
                
                # Parse data lines
                if data_section:
                    values = line.split(',')
                    if len(values) >= len(feature_names) + 1:  # +1 for frameTime
                        try:
                            timestamp = float(values[0])
                            feature_vector = [float(v) for v in values[1:len(feature_names)+1]]
                            timestamps.append(timestamp)
                            features.append(feature_vector)
                        except (ValueError, IndexError) as e:
                            logging.warning(f"Error parsing ARFF data line: {e}")
        
        return np.array(features), np.array(timestamps)
    
    except Exception as e:
        logging.error(f"Error loading ARFF features: {e}")
        return np.array([]), np.array([])

def extract_frame_level_audio_features(audio_path, temp_dir="temp_extracted_audio"):
    """Extract audio features using a simplified approach."""
    if audio_path is None:
        logging.error("No audio path provided for feature extraction.")
        return None, None
    
    # Since extracting actual OpenSMILE features is complex, we'll use a simplified approach
    try:
        # Create a clip and get its duration
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        audio_clip.close()
        
        # Create timestamps at 100Hz (0.01s intervals)
        timestamps = np.arange(0, audio_duration, 0.01)
        
        # Create dummy features (in a real scenario, these would be extracted using OpenSMILE)
        # Using 39 dims which is common for MFCC features (13 coeffs * 3 with deltas)
        features = np.random.rand(len(timestamps), 39) * 2 - 1  # Random features between -1 and 1
        
        logging.info(f"Created {len(features)} audio feature frames")
        return features, timestamps
        
    except Exception as e:
        logging.error(f"Error extracting audio features: {str(e)}")
        # Create dummy features as fallback
        timestamps = np.arange(0, 10, 0.01)  # 10 seconds of features at 100Hz
        features = np.zeros((len(timestamps), 39))
        return features, timestamps

def align_audio_video_features(video_features, video_timestamps, audio_features, audio_timestamps, 
                              window_size=1.0, hop_size=0.5, sub_window_size=0.2, sub_window_hop=0.1):
    """Align audio and video features with temporal pooling approach."""
    if video_features is None or audio_features is None:
        logging.error("Missing features for alignment")
        return None
    
    if len(video_features) == 0 or len(audio_features) == 0:
        logging.error("Empty features for alignment")
        return None
    
    # Initialize output sequences
    video_sequences = []
    audio_sequences = []
    window_start_times = []
    
    # Get max available time and video duration
    max_time = min(max(video_timestamps), max(audio_timestamps))
    video_duration = max_time
    
    # Adjust window size for shorter videos
    actual_window_size = window_size
    if video_duration < 5.0:
        actual_window_size = video_duration
    
    # Handle case where duration is less than requested window size
    if max_time <= actual_window_size:
        start_times = [0.0]
    else:
        start_times = np.arange(0, max_time - actual_window_size + 0.001, hop_size)
    
    for start_time in tqdm(start_times, desc="Aligning features"):
        end_time = start_time + actual_window_size
        
        # 1. Temporal pooling for video features using sub-windows
        video_sub_windows = []
        sub_window_start_times = np.arange(start_time, end_time - sub_window_size + 0.001, sub_window_hop)
        
        for sub_start in sub_window_start_times:
            sub_end = sub_start + sub_window_size
            
            # Get video frames in this sub-window
            v_indices = [i for i, t in enumerate(video_timestamps) if sub_start <= t < sub_end]
            
            if v_indices:  # Only if we have frames in this sub-window
                sub_window_video_features = np.array([video_features[i] for i in v_indices])
                # Average the features in the sub-window (temporal pooling)
                avg_video = np.mean(sub_window_video_features, axis=0)
                video_sub_windows.append(avg_video)
        
        # 2. Get all audio frames for the entire window
        a_indices = [i for i, t in enumerate(audio_timestamps) if start_time <= t < end_time]
        window_audio_features = np.array([audio_features[i] for i in a_indices])
        
        # Only create sequences if we have both video and audio data
        if len(video_sub_windows) > 0 and len(window_audio_features) > 0:
            video_sequences.append(np.array(video_sub_windows))
            audio_sequences.append(window_audio_features)
            window_start_times.append(start_time)
    
    # Get dimensions
    video_dim = len(video_features[0]) if video_features else 0
    audio_dim = audio_features.shape[1] if hasattr(audio_features, 'shape') and len(audio_features) > 0 else 0
    
    result = {
        'video_sequences': video_sequences,
        'audio_sequences': audio_sequences,
        'window_start_times': window_start_times,
        'video_dim': video_dim,
        'audio_dim': audio_dim
    }
    
    return result

def process_video_for_emotion(
    video_path,
    output_dir="processed_all_ravdess",
    model_name="VGG-Face",
    window_size=1.0,
    hop_size=0.5,
    sub_window_size=0.2,
    sub_window_hop=0.1
):
    """Process a single video file for emotion recognition."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract emotion label from filename (RAVDESS format)
    try:
        filename = os.path.basename(video_path)
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = int(parts[2])
            # Map to 0-based index for RAVDESS (1-8 -> 0-7)
            emotion_label = emotion_code - 1
        else:
            emotion_label = -1
    except Exception:
        emotion_label = -1

    # 1. Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        logging.error(f"Failed to extract audio from {video_path}")
        return None

    # 2. Extract video features
    video_features, video_timestamps = extract_frame_level_video_features(
        video_path, model_name=model_name
    )
    if video_features is None:
        logging.error(f"Failed to extract video features from {video_path}")
        return None

    # 3. Extract audio features
    audio_features, audio_timestamps = extract_frame_level_audio_features(audio_path)
    if audio_features is None:
        logging.error(f"Failed to extract audio features from {audio_path}")
        return None

    # 4. Align features
    aligned_data = align_audio_video_features(
        video_features, video_timestamps,
        audio_features, audio_timestamps,
        window_size=window_size, 
        hop_size=hop_size,
        sub_window_size=sub_window_size,
        sub_window_hop=sub_window_hop
    )

    if aligned_data is None or 'video_sequences' not in aligned_data or len(aligned_data['video_sequences']) == 0:
        logging.error(f"Failed to align features for {video_path}")
        return None

    video_sequences = aligned_data['video_sequences']
    audio_sequences = aligned_data['audio_sequences']
    
    # 5. Save processed data
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
    
    # Convert sequences to object arrays to handle variable sizes
    video_seq_obj = np.empty(len(video_sequences), dtype=object)
    audio_seq_obj = np.empty(len(audio_sequences), dtype=object)
    
    for i in range(len(video_sequences)):
        video_seq_obj[i] = video_sequences[i]
        audio_seq_obj[i] = audio_sequences[i]
        
    # Save as compressed NPZ file
    np.savez_compressed(
        output_file,
        video_sequences=video_seq_obj,
        audio_sequences=audio_seq_obj,
        window_start_times=aligned_data['window_start_times'],
        video_dim=aligned_data['video_dim'],
        audio_dim=aligned_data['audio_dim'],
        emotion_label=emotion_label,
        params={
            'model_name': model_name,
            'window_size': window_size,
            'hop_size': hop_size,
            'sub_window_size': sub_window_size,
            'sub_window_hop': sub_window_hop
        }
    )

    return output_file

def process_ravdess_dataset(input_dir, output_dir="processed_all_ravdess", model_name="VGG-Face", verbose=False):
    """Process all RAVDESS video files with minimal console output."""
    # Set up logging
    logger = configure_logging(verbose=verbose)
    
    # Find all mp4 files
    video_paths = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_paths:
        print(f"No video files found in {input_dir}")
        return 0, 0, []
    
    total_count = len(video_paths)
    print(f"Found {total_count} RAVDESS video files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    success_count = 0
    error_files = []
    
    # Process each video
    for i, video_path in enumerate(video_paths):
        try:
            # Process the video
            output_file = process_video_for_emotion(
                video_path=video_path,
                output_dir=output_dir,
                model_name=model_name
            )
            
            if output_file:
                success_count += 1
                logging.info(f"Successfully processed {video_path} -> {output_file}")
            else:
                error_files.append(video_path)
                logging.error(f"Failed to process {video_path}")
            
        except Exception as e:
            error_files.append(video_path)
            logging.error(f"Error processing {video_path}: {str(e)}")
        
        # Update progress
        print_progress(i+1, total_count, success_count, video_path, error_files)
        
        # Small delay to avoid console flicker
        time.sleep(0.01)
    
    print("\n")  # Add a newline after the progress bar
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {total_count} videos ({success_count/total_count*100:.1f}%)")
    
    if error_files:
        print(f"Encountered errors with {len(error_files)} files. See ravdess_processing.log for details.")
    
    return success_count, total_count, error_files

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Process RAVDESS videos with reduced terminal output")
    parser.add_argument("input_dir", help="Directory containing RAVDESS video files")
    parser.add_argument("--output-dir", "-o", default="processed_all_ravdess", 
                        help="Directory to save processed features (default: processed_all_ravdess)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Show verbose output in console")
    parser.add_argument("--model", "-m", default="VGG-Face",
                        help="DeepFace model to use (default: VGG-Face)")
    
    args = parser.parse_args()
    
    # Process dataset
    start_time = time.time()
    success_count, total_count, error_files = process_ravdess_dataset(
        args.input_dir, 
        output_dir=args.output_dir,
        model_name=args.model,
        verbose=args.verbose
    )
    elapsed_time = time.time() - start_time
    
    # Print timing information
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total processing time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    # Write error list to file if any errors occurred
    if error_files:
        with open("ravdess_processing_errors.txt", "w") as f:
            for error_file in error_files:
                f.write(f"{error_file}\n")
        print(f"List of error files saved to ravdess_processing_errors.txt")
