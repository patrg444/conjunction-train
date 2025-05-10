#!/usr/bin/env python3
"""
Process CREMA-D dataset as single segments for emotion recognition.
Each video is treated as a single segment with its emotion label mapped to our 7-class schema.
Supports multiprocessing for faster processing.
"""

import os
import sys
import glob
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
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crema_d_processing.log"),
        logging.StreamHandler()
    ]
)

# CREMA-D emotion mapping to our 7-class schema
CREMAD_EMOTION_MAPPING = {
    'NEU': 0,  # Neutral → Neutral/Calm (0)
    'HAP': 1,  # Happy → Happy (1)
    'SAD': 2,  # Sad → Sad (2)
    'ANG': 3,  # Angry → Angry (3)
    'FEA': 4,  # Fear → Fearful (4)
    'DIS': 5   # Disgust → Disgust (5)
    # No mapping for Surprised (6) as CREMA-D doesn't have this category
}

# Text labels for emotions
EMOTION_LABELS = [
    'Neutral/Calm', 'Happy', 'Sad', 
    'Angry', 'Fearful', 'Disgust', 'Surprised'
]

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
    """Selects the primary face embedding based on face area."""
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
    for idx in frame_indices:
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
                # Try to detect faces
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
    return np.array(frame_features), np.array(timestamps), np.array(valid_frames)

def extract_frame_level_audio_features(audio_path, opensmile_config="opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"):
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
        temp_dir_path = os.path.join(os.getcwd(), "temp_extracted_audio")  # Ensure we use an absolute path
        
        # Create separate files for different outputs
        csv_path = os.path.join(temp_dir_path, f"{filename_base}.csv")
        lld_csv_path = os.path.join(temp_dir_path, f"{filename_base}.lld.csv")
        
        # Ensure temp directory exists
        os.makedirs(temp_dir_path, exist_ok=True)
        
        # Path to openSMILE executable
        opensmile_path = os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract")
        
        # Verify that openSMILE exists and is executable
        if not os.path.exists(opensmile_path):
            logging.error(f"openSMILE executable not found at: {opensmile_path}")
            return None, None
        
        if not os.access(opensmile_path, os.X_OK):
            logging.error(f"openSMILE executable is not executable: {opensmile_path}")
            return None, None
        
        # Build openSMILE command
        cmd = [
            opensmile_path,
            "-C", opensmile_config,
            "-I", audio_path,
            "-csvoutput", csv_path,
            "-lldcsvoutput", lld_csv_path,
            "-noconsoleoutput", "0"
        ]

        # Execute openSMILE
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                         universal_newlines=True, timeout=30)
        except Exception as e:
            logging.error(f"openSMILE execution failed: {e}")
            return None, None

        # Read the LLD CSV output (low-level descriptors = frame-level features)
        try:
            # Check if the file exists and has content
            if not os.path.exists(lld_csv_path) or os.path.getsize(lld_csv_path) == 0:
                logging.error(f"LLD output file not found or empty: {lld_csv_path}")
                return None, None
            
            # Load data, skipping the first row (header)
            data = np.genfromtxt(lld_csv_path, delimiter=';', skip_header=1)
            
            if data.size == 0 or data.ndim < 2:
                logging.error(f"openSMILE produced empty or invalid output for {audio_path}")
                return None, None
            
            # First column is frame/time index, rest are features
            timestamps = data[:, 0]  # First column is time in seconds
            features = data[:, 1:]   # Rest are the actual features
            
            # Clean up temporary files
            try:
                for file_path in [csv_path, lld_csv_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                logging.warning(f"Warning: Could not remove temporary files: {str(e)}")

            return features, timestamps
        except Exception as e:
            logging.error(f"Error reading CSV output from openSMILE: {e}")
            return None, None

    except Exception as e:
        logging.error(f"Error extracting audio features: {str(e)}")
        return None, None

def extract_emotion_from_filename(filename):
    """Extract emotion label from CREMA-D filename.
    
    CREMA-D filename format: 1076_MTI_SAD_XX.flv or 1076_MTI_SAD_XX.mp4
    
    Returns:
        Tuple of (original_emotion_code, mapped_emotion_code, emotion_text)
    """
    # Extract filename without extension
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # CREMA-D filename format: 1076_MTI_SAD_XX
    parts = basename.split('_')
    if len(parts) >= 3:
        emotion_code = parts[2]
        if emotion_code in CREMAD_EMOTION_MAPPING:
            mapped_emotion = CREMAD_EMOTION_MAPPING[emotion_code]
            return emotion_code, mapped_emotion, EMOTION_LABELS[mapped_emotion]
    
    # Default to "unknown" if parsing fails
    return None, -1, "unknown"

def process_video_for_emotion(
    video_path,
    output_dir,
    model_name="VGG-Face",
    sampling_fps=15,
    opensmile_config="opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
):
    """Process a single CREMA-D video file as a single segment.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        sampling_fps: Frames per second to sample from the video
        opensmile_config: Path to openSMILE config file
        
    Returns:
        Path to output file or None if processing failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract emotion from filename
    original_emotion, emotion_label, emotion_text = extract_emotion_from_filename(video_path)
    if emotion_label == -1:
        logging.error(f"Could not extract emotion from filename: {video_path}")
        return None

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
    audio_features, audio_timestamps = extract_frame_level_audio_features(
        audio_path, opensmile_config=opensmile_config
    )
    if audio_features is None or len(audio_features) == 0:
        logging.error(f"Failed to extract audio features from {audio_path}")
        return None

    # Calculate dimensions
    video_dim = video_features.shape[1] if video_features.shape[0] > 0 else 0
    audio_dim = audio_features.shape[1] if audio_features.shape[0] > 0 else 0
    
    # Save processed data - each video is a single segment with all its frame-level features
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
    
    # Save as compressed NPZ file
    np.savez_compressed(
        output_file,
        video_features=video_features,
        video_timestamps=video_timestamps,
        audio_features=audio_features,
        audio_timestamps=audio_timestamps,
        valid_frames=valid_frames,
        video_dim=video_dim,
        audio_dim=audio_dim,
        emotion_label=emotion_label,
        original_emotion=original_emotion,
        params={
            'model_name': model_name,
            'sampling_fps': sampling_fps,
            'is_single_segment': True,
            'dataset': 'CREMA-D'
        }
    )

    return output_file

def process_single_video(video_path, output_dir, model_name, sampling_fps=15, opensmile_config=None):
    """Process a single video file with proper error handling."""
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
            sampling_fps=sampling_fps,
            opensmile_config=opensmile_config
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
        import traceback
        logging.error(traceback.format_exc())
        return (False, error_msg)

def process_crema_d_dataset(
    input_path, 
    output_dir="processed_crema_d_single_segment", 
    model_name="VGG-Face",
    sampling_fps=15,
    n_workers=None,
    sample_count=None,
    opensmile_config=None
):
    """Process CREMA-D video files as single segments.
    
    Args:
        input_path: Path to directory containing CREMA-D video files
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        sampling_fps: Frames per second to sample from videos
        n_workers: Number of worker processes to use (None = all CPUs)
        sample_count: Number of videos to process (None = all)
        opensmile_config: Path to openSMILE config file
        
    Returns:
        Tuple of (success_count, total_count, error_files)
    """
    # Find all video files
    video_files = []
    
    # Handle both .flv (original) and .mp4 (converted) formats
    for ext in ['*.flv', '*.mp4']:
        pattern = os.path.join(input_path, ext)
        video_files.extend(glob.glob(pattern))
    
    if not video_files:
        logging.error(f"No video files found in {input_path}")
        return 0, 0, []
        
    # Sort for consistent processing order
    video_files = sorted(video_files)
    
    # Limit sample count if specified
    if sample_count is not None:
        video_files = video_files[:sample_count]
    
    total_count = len(video_files)
    logging.info(f"Found {total_count} CREMA-D video files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which files have already been processed
    existing_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.npz'))
    unprocessed_videos = []
    
    for video_path in video_files:
        basename = os.path.splitext(os.path.basename(video_path))[0]
        if basename not in existing_files:
            unprocessed_videos.append(video_path)
    
    skipped_count = total_count - len(unprocessed_videos)
    total_to_process = len(unprocessed_videos)
    
    if skipped_count > 0:
        logging.info(f"Skipping {skipped_count} videos that have already been processed")
    
    if total_to_process == 0:
        logging.info("All videos have already been processed")
        return total_count - skipped_count, total_count, []
    
    logging.info(f"Processing {total_to_process} videos")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    logging.info(f"Using {n_workers} worker processes")
    
    # Process videos
    if n_workers <= 1:
        # Process sequentially
        success_count = 0
        error_files = []
        
        for i, video_path in enumerate(tqdm(unprocessed_videos, desc="Processing videos")):
            success, error = process_single_video(video_path, output_dir, model_name, sampling_fps, opensmile_config)
            if success:
                success_count += 1
            else:
                error_files.append(error)
            
            # Print progress
            if (i+1) % 10 == 0 or i == total_to_process - 1:
                logging.info(f"Processed {i+1}/{total_to_process} videos. Success rate: {success_count/(i+1)*100:.1f}%")
    else:
        # Process in parallel
        # Create a manager for shared resources
        manager = multiprocessing.Manager()
        error_files = manager.list()
        success_count = manager.Value('i', 0)
        lock = manager.Lock()
        
        # Create a partial function with fixed arguments
        process_func = partial(process_single_video, output_dir=output_dir, model_name=model_name, 
                             sampling_fps=sampling_fps, opensmile_config=opensmile_config)
        
        # Create and start the pool
        with multiprocessing.Pool(processes=n_workers) as pool:
            # Process files
            results = []
            for i, (success, error) in enumerate(tqdm(pool.imap_unordered(process_func, unprocessed_videos), 
                                                    total=total_to_process, desc="Processing videos")):
                if success:
                    with lock:
                        success_count.value += 1
                else:
                    with lock:
                        error_files.append(error)
                
                # Print progress
                if (i+1) % 10 == 0 or i == total_to_process - 1:
                    logging.info(f"Processed {i+1}/{total_to_process} videos. Success rate: {success_count.value/(i+1)*100:.1f}%")
        
        # Convert manager objects to standard Python types
        success_count = success_count.value
        error_files = list(error_files)
    
    # Print summary
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successfully processed {success_count} out of {total_to_process} videos ({success_count/total_to_process*100:.1f}%)")
    
    if error_files:
        logging.info(f"Encountered errors with {len(error_files)} files. See log for details.")
        
        # Write error list to file
        error_log = os.path.join(output_dir, "processing_errors.txt")
        with open(error_log, "w") as f:
            for error in error_files:
                f.write(f"{error}\n")
        logging.info(f"Error list saved to {error_log}")
    
    # Return total including previously processed files
    return success_count + skipped_count, total_count, error_files

def print_emotion_distribution(output_dir):
    """Print the distribution of emotions in the processed dataset."""
    files = glob.glob(os.path.join(output_dir, "*.npz"))
    if not files:
        logging.info(f"No processed files found in {output_dir}")
        return
    
    emotion_counts = {i: 0 for i in range(7)}  # 7 classes
    
    for file in files:
        try:
            data = np.load(file)
            if 'emotion_label' in data:
                emotion = data['emotion_label'].item()
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    
    total = sum(emotion_counts.values())
    logging.info(f"Emotion distribution in {total} processed files:")
    
    for emotion, count in emotion_counts.items():
        if emotion < len(EMOTION_LABELS):
            percent = (count / total) * 100 if total > 0 else 0
            logging.info(f"  {EMOTION_LABELS[emotion]}: {count} ({percent:.1f}%)")
        else:
            logging.info(f"  Unknown ({emotion}): {count}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process CREMA-D dataset as single segments")
    parser.add_argument("--input-dir", "-i", default="data/CREMA-D",
                      help="Directory containing CREMA-D video files (default: data/CREMA-D)")
    parser.add_argument("--output-dir", "-o", default="processed_crema_d_single_segment",
                      help="Directory to save processed features (default: processed_crema_d_single_segment)")
    parser.add_argument("--model", "-m", default="VGG-Face",
                      help="DeepFace model to use (default: VGG-Face)")
    parser.add_argument("--sampling-fps", "-s", type=int, default=15,
                      help="Frames per second to sample from videos (default: 15)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                      help="Number of worker processes (default: number of CPU cores)")
    parser.add_argument("--sample-count", "-n", type=int, default=None,
                      help="Number of videos to process, for testing (default: all)")
    parser.add_argument("--opensmile-config", "-c", 
                      default="opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf",
                      help="Path to openSMILE configuration file")
    
    args = parser.parse_args()
    
    # Print configuration
    logging.info("=== CREMA-D Single Segment Processing ===")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Sampling FPS: {args.sampling_fps}")
    logging.info(f"Workers: {args.workers if args.workers else 'auto'}")
    logging.info(f"Sample count: {args.sample_count if args.sample_count else 'all'}")
    
    # Process dataset
    start_time = time.time()
    success_count, total_count, error_files = process_crema_d_dataset(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        sampling_fps=args.sampling_fps,
        n_workers=args.workers,
        sample_count=args.sample_count,
        opensmile_config=args.opensmile_config
    )
    elapsed_time = time.time() - start_time
    
    # Print timing information
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total processing time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    # Print emotion distribution
    print_emotion_distribution(args.output_dir)
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
