#!/usr/bin/env python3
"""
Process RAVDESS dataset files with minimal logging.
This script only logs errors and failed file processing to the log file,
significantly reducing the log file size.
Supports multiprocessing to speed up processing across multiple CPU cores.
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
from deepface.DeepFace import analyze
import time
import multiprocessing
from functools import partial

# Configure logging with minimal output
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
    file_handler = logging.FileHandler("ravdess_errors.log")
    file_handler.setLevel(logging.ERROR)  # Only log errors to file
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Only capture WARNING+ logs by default
    root_logger.handlers = []
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
    video_basename = os.path.basename(video_path)
    audio_filename = os.path.splitext(video_basename)[0] + ".wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
            video.close()
            return audio_path
        else:
            logging.error(f"No audio track found in {video_path}")
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
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    areas = []
    for obj in embedding_objs:
        x, y, w, h = obj["facial_area"].values()
        areas.append((w * h, obj["embedding"]))

    primary_embedding = max(areas, key=lambda x: x[0])[1]
    return primary_embedding

def extract_frame_level_video_features(video_path, model_name="VGG-Face", fps=None):
    """Extract features from video frames."""
    # Create a local cache for this process
    deepface_cache = {}
    
    def get_cache_key(video_path, frame_idx):
        return f"{video_path}_{model_name}_{frame_idx}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None, None, None

    # Get video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0

    # Determine sampling rate
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

    # Extract features for each selected frame
    frame_features = []
    valid_frames = []  # Track which frames were successfully processed
    for idx in tqdm(frame_indices, desc=f"Extracting video features ({model_name})"):
        cache_key = get_cache_key(video_path, idx)

        # Check cache
        if cache_key in deepface_cache:
            frame_features.append(deepface_cache[cache_key])
            valid_frames.append(True)  # Cached frames are valid
            continue

        # Not in cache, need to process
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Check if the frame is valid
            if frame is None:
                logging.warning(f"Invalid frame {idx} from {video_path}")
                frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                valid_frames.append(False)  # Mark as invalid
                continue

            # Use DeepFace.analyze to check for face detection
            try:
                # Note: actions=['detection'] might need to be updated based on the API version
                # If there's an error about invalid action, handle it accordingly
                try:
                    analyze_result = DeepFace.analyze(img_path=frame, actions=['detection'], enforce_detection=False, silent=True)
                    if len(analyze_result) == 0 or "face_1" not in analyze_result[0]:
                        logging.warning(f"No faces detected in frame {idx} of {video_path}.")
                        frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                        valid_frames.append(False)  # Mark as invalid
                        continue
                except Exception as detect_err:
                    # If 'detection' is not a valid action, just proceed with represent directly
                    pass

                retries = 0
                max_retries = 3
                while retries < max_retries:
                    try:
                        embedding_objs = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
                        
                        try:
                            embedding = select_primary_face(embedding_objs)
                            # Cache the result
                            deepface_cache[cache_key] = embedding
                            frame_features.append(embedding)
                            valid_frames.append(True)  # Mark as valid
                            break  # Exit loop if successful
                        except Exception as e:
                            logging.error(f"Error in select_primary_face at frame {idx}: {str(e)}")
                            embedding = np.zeros(get_embedding_dimension(model_name))
                            frame_features.append(embedding)
                            valid_frames.append(False)  # Mark as invalid
                            break
                    except Exception as e:
                        retries += 1
                        logging.error(f"DeepFace.represent error at frame {idx}, attempt {retries}/{max_retries}: {str(e)}")
                        time.sleep(0.5)  # Wait before retrying
                if retries == max_retries:
                    logging.error(f"Giving up on frame {idx} after {max_retries} attempts.")
                    frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                    valid_frames.append(False)  # Mark as invalid
            except Exception as e:
                logging.error(f"Error analyzing frame {idx} of {video_path}: {str(e)}")
                frame_features.append(np.zeros(get_embedding_dimension(model_name)))
                valid_frames.append(False)  # Mark as invalid
        else:
            logging.warning(f"Failed to read frame {idx} from {video_path}")
            frame_features.append(np.zeros(get_embedding_dimension(model_name)))
            valid_frames.append(False)  # Mark as invalid

    cap.release()
    return frame_features, timestamps, valid_frames

def extract_frame_level_audio_features(audio_path, temp_dir="temp_extracted_audio", opensmile_config="opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"):
    """Extract audio features using openSMILE."""
    if audio_path is None:
        logging.error("No audio path provided for feature extraction.")
        return None, None
    
    try:
        # Create a clip and get its duration
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        audio_clip.close()
        
        # Create temporary output files for openSMILE results
        audio_basename = os.path.basename(audio_path)
        filename_base = os.path.splitext(audio_basename)[0]
        temp_dir_path = os.path.join(os.getcwd(), temp_dir)  # Ensure we use an absolute path
        
        # Create separate files for different outputs
        csv_path = os.path.join(temp_dir_path, f"{filename_base}.csv")
        lld_csv_path = os.path.join(temp_dir_path, f"{filename_base}.lld.csv")
        time_path = os.path.join(temp_dir_path, f"{filename_base}.txt")
        
        # Ensure temp directory exists
        os.makedirs(temp_dir_path, exist_ok=True)
        
        # Path to openSMILE executable (must be in PATH or specify full path)
        opensmile_path = os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract")
        
        # Verify that openSMILE exists and is executable
        if not os.path.exists(opensmile_path):
            logging.error(f"openSMILE executable not found at: {opensmile_path}")
            return None, None
        
        if not os.access(opensmile_path, os.X_OK):
            logging.error(f"openSMILE executable is not executable: {opensmile_path}")
            return None, None
        
        # Build openSMILE command with proper command-line parameters
        cmd = [
            opensmile_path,
            "-C", opensmile_config,
            "-I", audio_path,           # Input file specified with -I
            "-csvoutput", csv_path,     # Summary/functional features output
            "-lldcsvoutput", lld_csv_path,  # LLD (frame-level) output
            "-noconsoleoutput", "0"     # Enable console output for debugging
        ]

        # Execute openSMILE with proper error handling
        try:
            logging.info(f"Running openSMILE command: {' '.join(cmd)}")
            output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, timeout=30)
            logging.info(f"openSMILE stdout: {output.stdout}")
            logging.info(f"openSMILE stderr: {output.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"openSMILE execution failed: {e}")
            logging.error(f"Command: {' '.join(cmd)}")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            return None, None
        except subprocess.TimeoutExpired as e:
            logging.error(f"openSMILE execution timed out after {e.timeout} seconds")
            return None, None
        except Exception as e:
            logging.error(f"Error executing openSMILE: {str(e)}")
            return None, None

        # Read the LLD CSV output (low-level descriptors = frame-level features)
        try:
            logging.info(f"Loading LLD data from: {lld_csv_path}")
            
            # Check if the file exists and has content
            if not os.path.exists(lld_csv_path):
                logging.error(f"LLD output file not found: {lld_csv_path}")
                return None, None
                
            if os.path.getsize(lld_csv_path) == 0:
                logging.error(f"LLD output file is empty: {lld_csv_path}")
                return None, None
            
            # Load data, skipping the first row (header)
            # We're directly processing the LLD (low-level descriptor) output
            data = np.genfromtxt(lld_csv_path, delimiter=';', skip_header=1)
            
            if data.size == 0 or data.ndim < 2:
                logging.error(f"openSMILE produced empty or invalid output for {audio_path}")
                return None, None
            
            # First column is frame/time index, rest are features
            timestamps = data[:, 0]  # First column is time in seconds
            features = data[:, 1:]   # Rest are the actual features
            
            # Clean up temporary files
            try:
                for file_path in [csv_path, lld_csv_path, time_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                logging.warning(f"Warning: Could not remove temporary files: {str(e)}")

            return features, timestamps
        except Exception as e:
            logging.error(f"Error reading CSV output from openSMILE: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None, None

    except Exception as e:
        logging.error(f"Error extracting audio features: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

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
    sampling_fps=15
):
    """Process a single video file for emotion recognition.
    
    Each video is treated as a single segment, keeping all frame-level features
    as a sequence for LSTM processing.
    
    Args:
        video_path: Path to the video file.
        output_dir: Directory to save processed features.
        model_name: DeepFace model to use.
        sampling_fps: Frames per second to sample from the video. Lower values reduce
                     computation but may lose temporal information. If None, uses the
                     original video FPS.
    """
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
        logging.error(f"Could not extract emotion label from filename: {video_path}")
        emotion_label = -1

    # 1. Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        logging.error(f"Failed to extract audio from {video_path}")
        return None

    # 2. Extract video features
    video_features, video_timestamps, valid_frames = extract_frame_level_video_features(
        video_path, model_name=model_name, fps=sampling_fps
    )
    if video_features is None or len(video_features) == 0:
        logging.error(f"Failed to extract video features from {video_path}")
        return None

    # 3. Extract audio features
    audio_features, audio_timestamps = extract_frame_level_audio_features(audio_path)
    if audio_features is None or len(audio_features) == 0:
        logging.error(f"Failed to extract audio features from {audio_path}")
        return None

    # Calculate dimensions
    video_dim = len(video_features[0]) if video_features else 0
    audio_dim = audio_features.shape[1] if hasattr(audio_features, 'shape') and len(audio_features) > 0 else 0
    
    # Save processed data - each video is a single segment with all its frame-level features
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
    
    # Save as compressed NPZ file
    np.savez_compressed(
        output_file,
        video_features=np.array(video_features),
        video_timestamps=np.array(video_timestamps),
        audio_features=audio_features,
        audio_timestamps=audio_timestamps,
        valid_frames=np.array(valid_frames),
        video_dim=video_dim,
        audio_dim=audio_dim,
        emotion_label=emotion_label,
        params={
            'model_name': model_name,
            'sampling_fps': sampling_fps,
            'is_single_segment': True  # Flag to indicate this is a single segment per video
        }
    )

    return output_file

def process_single_video(video_path, output_dir, model_name, sampling_fps=15, lock=None):
    """Process a single video file with proper error handling.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        lock: Optional multiprocessing lock for updating shared resources
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Check if output file already exists
        output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.npz"
        output_file_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_file_path):
            # Skip processing if file already exists
            logging.info(f"Skipping {video_path}, output file already exists")
            return (True, None)
            
        # Process the video
        output_file = process_video_for_emotion(
            video_path=video_path,
            output_dir=output_dir,
            model_name=model_name,
            sampling_fps=sampling_fps
        )
        
        if output_file:
            return (True, None)
        else:
            error_msg = f"Failed to process {video_path}"
            logging.error(error_msg)
            return (False, error_msg)
            
    except Exception as e:
        error_msg = f"Error processing {video_path}: {str(e)}"
        logging.error(error_msg)
        # Also log the stack trace for detailed debugging
        import traceback
        logging.error(traceback.format_exc())
        return (False, error_msg)

def process_ravdess_dataset(input_path, output_dir="processed_all_ravdess", model_name="VGG-Face", sampling_fps=15, verbose=False, n_workers=None):
    """Process RAVDESS video files with minimal console output and multiprocessing support.
    
    Each video is processed as a single segment, keeping the sequence of frame-level
    features intact for LSTM processing.
    
    Args:
        input_path: Path to directory containing RAVDESS video files or a single .mp4 file
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        sampling_fps: Frames per second to sample from videos (default: 15)
        verbose: If True, show INFO level logs in console
        n_workers: Number of worker processes to use (None = use all available CPUs)
        
    Returns:
        Tuple of (success_count, total_count, error_files)
    """
    # Set up logging with minimal file output
    logger = configure_logging(verbose=verbose)
    
    # Determine if input_path is a file or directory
    video_paths = []
    if os.path.isfile(input_path) and input_path.endswith(".mp4"):
        # It's a single video file
        video_paths.append(input_path)
    else:
        # It's a directory, find all mp4 files recursively
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(".mp4"):
                    video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        print(f"No video files found in {input_path}")
        return 0, 0, []
    
    initial_count = len(video_paths)
    print(f"Found {initial_count} RAVDESS video files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which files have already been processed
    existing_npz_files = set()
    for file in os.listdir(output_dir):
        if file.endswith(".npz"):
            existing_npz_files.add(file)
    
    # Filter out already processed videos
    unprocessed_videos = []
    for video_path in video_paths:
        # Construct the expected output npz filename
        npz_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.npz"
        if npz_filename not in existing_npz_files:
            unprocessed_videos.append(video_path)
    
    skipped_count = initial_count - len(unprocessed_videos)
    total_count = len(unprocessed_videos)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} videos that have already been processed")
    
    if total_count == 0:
        print("All videos have already been processed")
        return initial_count - skipped_count, initial_count, []
    
    print(f"Processing {total_count} videos")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # Create a manager for shared resources
    manager = multiprocessing.Manager()
    error_files = manager.list()
    success_count = manager.Value('i', 0)
    lock = manager.Lock()
    
    print(f"Processing with {n_workers} worker processes")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_video, output_dir=output_dir, model_name=model_name, sampling_fps=sampling_fps, lock=lock)
    
    # Create and start the pool
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Use imap_unordered to process files as they complete (more efficient)
        results_iter = pool.imap_unordered(process_func, unprocessed_videos)
        
        # Use tqdm for progress bar
        results = []
        for i, (success, error_msg) in enumerate(tqdm(results_iter, total=total_count, desc="Processing videos")):
            # Update shared counters
            if success:
                with lock:
                    success_count.value += 1
            else:
                with lock:
                    error_files.append(error_msg)
            
            # Print progress update (less frequent than before to reduce overhead)
            if i % 10 == 0 or i == total_count - 1:
                print_progress(i+1, total_count, success_count.value, None, list(error_files))
    
    print("\n")  # Add a newline after the progress bar
    
    # Convert back to normal Python types since manager objects won't be accessible after manager shutdown
    final_success_count = success_count.value
    final_error_files = list(error_files)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {final_success_count} out of {total_count} videos ({final_success_count/total_count*100:.1f}%)")
    
    if final_error_files:
        print(f"Encountered errors with {len(final_error_files)} files. See ravdess_errors.log for details.")
    
    return final_success_count, total_count, final_error_files

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Process RAVDESS videos with minimal logging")
    parser.add_argument("input_dir", help="Directory containing RAVDESS video files")
    parser.add_argument("--output-dir", "-o", default="processed_all_ravdess", 
                        help="Directory to save processed features (default: processed_all_ravdess)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Show verbose output in console")
    parser.add_argument("--model", "-m", default="VGG-Face",
                        help="DeepFace model to use (default: VGG-Face)")
    parser.add_argument("--sampling-fps", "-s", type=int, default=15,
                        help="Frames per second to sample from videos (default: 15)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of worker processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    # Process dataset
    start_time = time.time()
    success_count, total_count, error_files = process_ravdess_dataset(
        args.input_dir, 
        output_dir=args.output_dir,
        model_name=args.model,
        sampling_fps=args.sampling_fps,
        verbose=args.verbose,
        n_workers=args.workers
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
