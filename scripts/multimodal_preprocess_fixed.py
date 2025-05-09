#!/usr/bin/env python3
"""
Modified multimodal preprocessing module that extracts audio functionals instead of LLDs.
This addresses the 26 vs 88 feature dimension issue identified in the analysis.
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
# Import directly from submodules as a workaround for potential import issues
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from deepface import DeepFace
# Import our custom FaceNet extractor
from facenet_extractor import FaceNetExtractor
import time
import pandas as pd
import tempfile

# Import our custom ARFF parser
from utils import load_arff_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("multimodal_preprocess_fixed.log"),
        logging.StreamHandler()
    ]
)

def extract_audio_from_video(video_path, output_dir="temp_extracted_audio", codec="pcm_s16le"):
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
            audio.write_audiofile(audio_path, codec=codec, verbose=False, logger=None)
            video.close()
            return audio_path
        else:
            logging.warning(f"No audio track found in {video_path}")
            video.close()
            return None
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {str(e)}")
        return None

def get_embedding_dimension(model_name="VGG-Face"):
    """Returns the embedding dimension for a given DeepFace model."""
    if model_name == "VGG-Face":
        return 4096
    elif model_name == "Facenet":
        return 128
    elif model_name == "Facenet512":
        return 512
    elif model_name == "OpenFace":
        return 128
    elif model_name == "DeepFace":  # default model
        return 4096
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def select_primary_face(embedding_objs):
    """Selects the primary (largest) face embedding from a list of embeddings."""
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    areas = [(obj["facial_area"]["w"] * obj["facial_area"]["h"], obj["embedding"]) for obj in embedding_objs]
    return max(areas, key=lambda x: x[0])[1]  # Return embedding of largest face

def resample_video(video_path, fps=15, output_dir="temp_resampled_videos"):
    """Resample video to target FPS before processing.
    
    This function creates a new video file with the target fps using FFmpeg,
    which is much more efficient than loading all frames and skipping some.
    
    If the video is already at the target fps, it returns the original video path.
    """
    # Make sure we're using absolute paths
    video_path_abs = os.path.abspath(video_path)
    
    # First check the current fps of the video
    try:
        cap = cv2.VideoCapture(video_path_abs)
        if not cap.isOpened():
            logging.error(f"Could not open video to check fps: {video_path_abs}")
        else:
            current_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # If the video is already at the target fps (or very close), use it directly
            if abs(current_fps - fps) < 0.1:  # Allow small rounding differences
                logging.info(f"Video {video_path} is already at target fps ({current_fps:.2f}), skipping resampling")
                return video_path_abs
    except Exception as e:
        logging.warning(f"Error checking video fps: {str(e)}. Will proceed with resampling.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(output_dir)
    
    # Create unique filename for the resampled video
    video_basename = os.path.basename(video_path_abs)
    resampled_basename = f"resampled_{fps}fps_{video_basename}"
    output_path = os.path.join(output_dir_abs, resampled_basename)
    
    # Skip if the resampled file already exists
    if os.path.exists(output_path):
        logging.info(f"Using existing resampled video: {output_path}")
        return output_path
    
    logging.info(f"Resampling video {video_path} to {fps} fps")
    
    try:
        # Check if ffmpeg is available
        if shutil.which("ffmpeg") is None:
            logging.warning("FFmpeg not found in PATH, falling back to OpenCV for resampling")
            return resample_video_with_opencv(video_path, fps, output_path)
        
        # Use FFmpeg for resampling (much faster and more reliable than OpenCV)
        cmd = [
            "ffmpeg", "-i", video_path_abs, 
            "-r", str(fps),          # Target frame rate
            "-c:v", "libx264",       # Video codec
            "-preset", "ultrafast",  # Fast encoding
            "-crf", "23",            # Quality balance
            "-c:a", "copy",          # Copy audio stream without re-encoding
            "-y",                    # Overwrite output file
            output_path
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error resampling video with FFmpeg: {result.stderr}")
            # Try OpenCV as fallback
            return resample_video_with_opencv(video_path, fps, output_path)
            
        return output_path
        
    except Exception as e:
        logging.error(f"Error during video resampling: {str(e)}")
        # Return original video path as fallback
        return video_path

def resample_video_with_opencv(video_path, fps=15, output_path=None):
    """Resample video using OpenCV (fallback method)."""
    # Make sure we're using absolute paths
    video_path_abs = os.path.abspath(video_path)
    
    if output_path is None:
        # Create temporary output path
        video_basename = os.path.basename(video_path_abs)
        output_dir = "temp_resampled_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_dir_abs = os.path.abspath(output_dir)
        output_path = os.path.join(output_dir_abs, f"resampled_{fps}fps_{video_basename}")
    
    logging.info(f"Resampling video with OpenCV: {video_path_abs} -> {output_path}")
    
    try:
        # Open the input video
        input_cap = cv2.VideoCapture(video_path_abs)
        if not input_cap.isOpened():
            logging.error(f"Could not open video: {video_path_abs}")
            return video_path_abs
            
        # Get original video properties
        orig_fps = input_cap.get(cv2.CAP_PROP_FPS)
        width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create the output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Calculate the frame sampling interval
        if fps < orig_fps:
            interval = orig_fps / fps
        else:
            interval = 1  # Use all frames if target fps >= original fps
            
        frame_count = 0
        next_frame_to_keep = 0
        
        while True:
            ret, frame = input_cap.read()
            if not ret:
                break
                
            # Only write frames at the target interval
            if frame_count >= next_frame_to_keep:
                out.write(frame)
                next_frame_to_keep += interval
                
            frame_count += 1
            
        # Release resources
        input_cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error in OpenCV resampling: {str(e)}")
        return video_path

def extract_frame_level_video_features(video_path, model_name="VGG-Face", fps=None, detector_backend="mtcnn", feature_extractor_type="deepface"):
    """Extract features from video frames using either DeepFace or FaceNet.
    
    Args:
        video_path: Path to the video file.
        model_name: DeepFace model to use (e.g., "VGG-Face"). Only used if feature_extractor_type="deepface".
        fps: Target frames per second. If None, uses original video fps.
        detector_backend: Face detector backend ("mtcnn", "retinaface", "opencv", etc.).
            Only used if feature_extractor_type="deepface".
        feature_extractor_type: Type of feature extractor to use ("deepface" or "facenet").
    """
    logging.info(f"Extracting video features from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None, None
    
    # Get basic video info
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If fps is specified, use it to determine frame sampling
    if fps is not None and fps < video_fps:
        # More precise calculation to ensure proper fps reduction
        sample_interval = int(round(video_fps / fps))
        # Force minimum interval of 2 for videos with fps < 2*target_fps
        if video_fps > fps and sample_interval < 2:
            sample_interval = 2
        effective_fps = video_fps / sample_interval
        logging.info(f"Resampling video from {video_fps:.2f} fps to {effective_fps:.2f} fps (target: {fps} fps, interval: every {sample_interval} frames)")
    else:
        sample_interval = 1  # Process every frame
        logging.info(f"Using original video fps: {video_fps:.2f}")
    
    # Initialize feature extractor based on the specified type
    facenet_extractor = None
    embedding_dim = None
    
    if feature_extractor_type == "facenet":
        facenet_extractor = FaceNetExtractor(
            keep_all=False,  # Only keep the largest face
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]  # MTCNN thresholds
        )
        embedding_dim = facenet_extractor.embedding_dim
        logging.info(f"Using FaceNet extractor with embedding dimension: {embedding_dim}")
    else:  # Default to DeepFace
        embedding_dim = get_embedding_dimension(model_name)
        logging.info(f"Using DeepFace extractor with model {model_name} and detector {detector_backend}")
        logging.info(f"Embedding dimension: {embedding_dim}")

    features = []
    timestamps = []
    logged_errors = set()  # Track errors to avoid duplicate logging
    
    frame_idx = 0
    valid_frame_count = 0
    zero_frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Only process frames at the specified interval
        if frame_idx % sample_interval == 0:
            # Extract features based on the specified extractor type
            if feature_extractor_type == "facenet":
                try:
                    # Use FaceNet extractor
                    embedding = facenet_extractor.extract_features(frame)
                    features.append(embedding)
                    timestamps.append(frame_idx / video_fps)
                    
                    # Track statistics
                    if np.any(embedding != 0):
                        valid_frame_count += 1
                    else:
                        zero_frame_count += 1
                        error_msg = f"No face detected in frame {frame_idx} of {video_path} by FaceNet"
                        if error_msg not in logged_errors:
                            logging.warning(error_msg)
                            logged_errors.add(error_msg)
                except Exception as e:
                    # Handle errors
                    features.append(np.zeros(embedding_dim))
                    timestamps.append(frame_idx / video_fps)
                    zero_frame_count += 1
                    
                    error_msg = f"Error processing frame {frame_idx} of {video_path} with FaceNet: {str(e)}"
                    if error_msg not in logged_errors:
                        logging.error(error_msg)
                        logged_errors.add(error_msg)
            else:
                # Use DeepFace (original implementation)
                try:
                    # Use DeepFace to represent the frame with the specified detector backend
                    embedding_objs = DeepFace.represent(
                        img_path=frame, 
                        model_name=model_name, 
                        enforce_detection=False,
                        detector_backend=detector_backend
                    )
                    
                    if embedding_objs:
                        primary_embedding = select_primary_face(embedding_objs)
                        features.append(primary_embedding)
                        timestamps.append(frame_idx / video_fps)
                        valid_frame_count += 1
                    else:
                        # No face detected, use zero vector
                        features.append(np.zeros(embedding_dim))
                        timestamps.append(frame_idx / video_fps)
                        zero_frame_count += 1
                        
                        error_msg = f"No face detected in frame {frame_idx} of {video_path} by DeepFace"
                        if error_msg not in logged_errors:
                            logging.warning(error_msg)
                            logged_errors.add(error_msg)
                            
                except Exception as e:
                    # Handle errors - use zero vector for frames where processing fails
                    features.append(np.zeros(embedding_dim))
                    timestamps.append(frame_idx / video_fps)
                    zero_frame_count += 1
                    
                    error_msg = f"Error processing frame {frame_idx} of {video_path} with DeepFace: {str(e)}"
                    if error_msg not in logged_errors:
                        logging.error(error_msg)
                        logged_errors.add(error_msg)
        
        frame_idx += 1
    
    cap.release()  # Release the video capture object
    
    if not features:
        logging.error(f"No features extracted from {video_path}")
        return None, None
    
    # Log face detection statistics
    total_processed = valid_frame_count + zero_frame_count
    if total_processed > 0:
        zero_percentage = (zero_frame_count / total_processed) * 100
        logging.info(f"Face detection statistics for {video_path}:")
        logging.info(f"  Frames with detected faces: {valid_frame_count} ({100 - zero_percentage:.1f}%)")
        logging.info(f"  Frames with no faces: {zero_frame_count} ({zero_percentage:.1f}%)")
        
    return np.array(features), np.array(timestamps)

def extract_audio_functionals(audio_path, opensmile_path=None, temp_dir="temp_extracted_audio"):
    """Extract audio functionals (88 features) using openSMILE.
    
    Instead of using LLDs (26 features) as in the original implementation,
    this function directly extracts the full functionals (88 features) that the model expects.
    """
    if audio_path is None:
        logging.error("No audio path provided for feature extraction.")
        return None, None
    
    # Create temporary output directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique output filenames
    audio_basename = os.path.basename(audio_path)
    csv_output = os.path.join(temp_dir, f"{os.path.splitext(audio_basename)[0]}_functionals.csv")
    
    # Find openSMILE executable path
    if opensmile_path is None:
        # Try to find openSMILE in the common location
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(project_root, "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
            os.path.join(project_root, "opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
            os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
            os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
            os.path.abspath("../opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
            "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract",
            "opensmile-3.0.2-macos-armv8/bin/SMILExtract",
            "SMILExtract"  # If in PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                opensmile_path = path
                logging.info(f"Found openSMILE at: {opensmile_path}")
                break
        
        if opensmile_path is None:
            logging.error("Could not find openSMILE executable. Please provide the path.")
            return None, None
    
    # Find the config file
    config_file = None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_configs = [
        # Use the original config file in the openSMILE installation where all relative paths work
        os.path.join(project_root, "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"),
        os.path.join(project_root, "opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"),
        os.path.join(project_root, "config/eGeMAPSv02.conf"),
        os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"),
        os.path.join(os.getcwd(), "config/eGeMAPSv02.conf"),
        "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf",
        "opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf",
        "config/eGeMAPSv02.conf"
    ]
    
    for path in possible_configs:
        if os.path.exists(path):
            config_file = path
            logging.info(f"Found openSMILE config at: {config_file}")
            break
    
    if config_file is None:
        logging.error("Could not find openSMILE config file.")
        return None, None
    
    try:
        # Run openSMILE with full eGeMAPS functionals extraction
        # OpenSMILE uses different argument formats in different versions
        # Use the proper parameter format with dashes
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", audio_path,   # Input file parameter with dash
            "-csvoutput", csv_output,  # CSV output parameter with dash
            "-instname", audio_basename  # Instance name parameter with dash
        ]
        
        logging.info(f"Running openSMILE functionals extraction: {' '.join(command)}")
        
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error running openSMILE: {result.stderr}")
            return None, None
        
        # Check if the output file exists
        if not os.path.exists(csv_output):
            logging.error(f"openSMILE did not create output file: {csv_output}")
            return None, None
        
        # Read the CSV file
        try:
            # First check what we got
            with open(csv_output, 'r') as f:
                header = f.readline().strip()
            
            # Count the number of features
            feature_count = len(header.split(';')) - 1  # Subtract 1 for 'name' column
            logging.info(f"Extracted {feature_count} audio features using openSMILE functionals")
            
            # Read the CSV with pandas
            df = pd.read_csv(csv_output, sep=';')
            
            # Extract features (first column is name, we drop it)
            if len(df) > 0:
                feature_names = df.columns[1:]  # Skip the 'name' column
                features = df[feature_names].values
                
                # Since functionals are one row per file, duplicate to match video length
                # Get audio duration
                audio_clip = AudioFileClip(audio_path)
                audio_duration = audio_clip.duration
                audio_clip.close()
                
                # Generate timestamps at 100Hz (standard for our processing)
                timestamps = np.arange(0, audio_duration, 0.01)
                
                # Create feature array with repeated functionals
                # This is a compromise - the traditional method uses LLDs which vary over time
                # Since functionals are fixed per file, we repeat them to create time series 
                # This approach allows the model to still process using existing logic
                repeated_features = np.tile(features, (len(timestamps), 1))
                
                logging.info(f"Created time series with {len(timestamps)} frames and {features.shape[1]} features")
                return repeated_features, timestamps
            else:
                logging.error("Empty CSV file")
                return None, None
                
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None, None
            
    except Exception as e:
        logging.error(f"Error extracting audio functionals: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def process_video_for_single_segment(video_path, output_dir="processed_features_fixed", model_name="VGG-Face", detector_backend="mtcnn", feature_extractor_type="deepface", skip_video=False, downsampled_dir=None, input_base_dir=None):
    """Process a video file to extract audio and video features and save as a single segment.
    
    This is a simpler version that extracts features from the entire video as one segment,
    instead of using windowing and alignment as in the original implementation.
    
    If input_base_dir is provided, the directory structure relative to that base will be preserved
    in the output directory.
    """
    # Calculate output path that preserves the directory structure
    if input_base_dir and os.path.exists(input_base_dir):
        # Get the relative path from the input base directory
        try:
            rel_path = os.path.relpath(os.path.dirname(video_path), input_base_dir)
            # If we're at the base directory, rel_path would be '.'
            if rel_path == '.':
                # Create the base output directory
                os.makedirs(output_dir, exist_ok=True)
                output_subdir = output_dir
            else:
                # Create subdirectory in output that matches the input structure
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
        except ValueError:
            # This can happen if the paths are on different drives on Windows
            logging.warning(f"Could not calculate relative path for {video_path}, using base output directory")
            os.makedirs(output_dir, exist_ok=True)
            output_subdir = output_dir
    else:
        # No input base directory provided, use the output directory directly
        os.makedirs(output_dir, exist_ok=True)
        output_subdir = output_dir
    
    logging.info(f"Processing video: {video_path}")
    logging.info(f"Output directory: {output_subdir}")
    
    # Extract emotion label from filename
    try:
        emotion_label = -1
        filename = os.path.basename(video_path)
        
        # RAVDESS format: 01-01-06-01-02-01-12.mp4
        # Third segment (06) is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        if '-' in filename:
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                # Map to 0-based index for RAVDESS (1-8 -> 0-7)
                emotion_label = emotion_code - 1
                logging.info(f"Extracted RAVDESS emotion label from filename: {emotion_label}")
        
        # CREMA-D format: 1001_DFA_ANG_XX.flv
        # Third segment (ANG) is emotion: NEU=neutral, HAP=happy, SAD=sad, ANG=angry, FEA=fearful, DIS=disgust
        elif '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                # Map CREMA-D emotion code to our unified schema
                crema_emotion_map = {
                    'NEU': 0,  # Neutral/Calm
                    'HAP': 1,  # Happy
                    'SAD': 2,  # Sad
                    'ANG': 3,  # Angry
                    'FEA': 4,  # Fearful
                    'DIS': 5,  # Disgust
                }
                if emotion_code in crema_emotion_map:
                    emotion_label = crema_emotion_map[emotion_code]
                    logging.info(f"Extracted CREMA-D emotion label from filename: {emotion_label}")
    except Exception as e:
        emotion_label = -1
        logging.warning(f"Error extracting emotion label: {str(e)}")
    
    # 1. Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        logging.error(f"Failed to extract audio from {video_path}")
        return None
    
    # 2. Extract video features (or create dummy features if skipping)
    if skip_video:
        logging.info(f"Skipping video feature extraction for {video_path}")
        # Create dummy video features - just for testing the pipeline
        dummy_dim = get_embedding_dimension(model_name)
        video_features = np.zeros((10, dummy_dim))  # 10 frames of zeros
        video_timestamps = np.linspace(0, 1, 10)    # 10 evenly spaced timestamps
    else:
        # Check if we should use a pre-downsampled video
        if downsampled_dir is not None:
            # Calculate relative path and look for the video in the downsampled directory
            rel_path = os.path.basename(video_path)
            downsampled_video_path = os.path.join(downsampled_dir, rel_path)
            
            if os.path.exists(downsampled_video_path):
                logging.info(f"Using pre-downsampled video: {downsampled_video_path}")
                # Use the pre-downsampled video directly (no need to specify fps)
                video_features, video_timestamps = extract_frame_level_video_features(
                    downsampled_video_path, 
                    model_name=model_name,
                    detector_backend=detector_backend,
                    feature_extractor_type=feature_extractor_type
                )
            else:
                logging.warning(f"Pre-downsampled video not found: {downsampled_video_path}, using original video instead")
                # First resample the video to 15 fps
                resampled_video = resample_video(video_path, fps=15)
                # Then process all frames from the resampled video
                video_features, video_timestamps = extract_frame_level_video_features(
                    resampled_video, 
                    model_name=model_name,
                    detector_backend=detector_backend,
                    feature_extractor_type=feature_extractor_type
                )
        else:
            # First resample the video to 15 fps
            resampled_video = resample_video(video_path, fps=15)
            # Then process all frames from the resampled video (no need to specify fps again)
            video_features, video_timestamps = extract_frame_level_video_features(
                resampled_video, 
                model_name=model_name,
                detector_backend=detector_backend,
                feature_extractor_type=feature_extractor_type
            )
        
        if video_features is None:
            logging.error(f"Failed to extract video features from {video_path}")
            return None
    
    # 3. Extract audio features (functionals version)
    audio_features, audio_timestamps = extract_audio_functionals(audio_path)
    if audio_features is None:
        logging.error(f"Failed to extract audio features from {audio_path}")
        return None
    
    # 4. Create output filename (in the appropriate subdirectory)
    output_file = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
    
    # 5. Save the features as a single segment
    np.savez_compressed(
        output_file,
        video_features=video_features,
        audio_features=audio_features,
        video_timestamps=video_timestamps,
        audio_timestamps=audio_timestamps,
        emotion_label=emotion_label,
        params={
            'model_name': model_name,
            'is_single_segment': True
        }
    )
    
    logging.info(f"Saved single-segment features to: {output_file}")
    return output_file

def process_dataset(video_dir, pattern="*.mp4", output_dir="processed_features_fixed", model_name="VGG-Face", detector_backend="mtcnn", feature_extractor_type="deepface", skip_video=False, downsampled_dir=None, preserve_structure=True):
    """Process all videos in a directory, including subdirectories.
    
    If preserve_structure is True, the directory structure relative to video_dir
    will be preserved in the output directory.
    """
    # Handle nested directory structure (especially for RAVDESS)
    if "RAVDESS" in video_dir or preserve_structure:
        # Use recursive glob for RAVDESS which has Actor subdirectories
        # or any dataset where we want to preserve directory structure
        video_paths = []
        for root, _, _ in os.walk(video_dir):
            video_paths.extend(glob.glob(os.path.join(root, pattern)))
    else:
        # For flat directories like CREMA-D
        video_paths = glob.glob(os.path.join(video_dir, pattern))
    
    if not video_paths:
        logging.error(f"No videos found matching pattern {pattern} in {video_dir}")
        return []
    
    logging.info(f"Found {len(video_paths)} videos matching pattern {pattern} in {video_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos
    output_files = []
    input_base_dir = video_dir if preserve_structure else None
    
    for video_path in tqdm(video_paths, desc=f"Processing videos ({pattern})"):
        output_file = process_video_for_single_segment(
            video_path=video_path,
            output_dir=output_dir,
            model_name=model_name,
            detector_backend=detector_backend,
            feature_extractor_type=feature_extractor_type,
            skip_video=skip_video,
            downsampled_dir=downsampled_dir,
            input_base_dir=input_base_dir
        )
        if output_file:
            output_files.append(output_file)
    
    # Filter out None results
    output_files = [f for f in output_files if f]
    logging.info(f"Successfully processed {len(output_files)} out of {len(video_paths)} videos")
    return output_files

if __name__ == "__main__":
    # Usage examples
    print("This module provides functions for multimodal feature extraction.")
    print("Example: python multimodal_preprocess_fixed.py <ravdess_dir> <output_dir> [--skip-video]")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract multimodal features from videos")
    parser.add_argument("input_dir", help="Input directory containing videos")
    parser.add_argument("output_dir", help="Output directory for processed features")
    parser.add_argument("--skip-video", action="store_true", help="Skip video feature extraction (for testing)")
    parser.add_argument("--downsampled-dir", help="Directory containing pre-downsampled videos")
    parser.add_argument("--no-preserve-structure", action="store_true", help="Do not preserve input directory structure in output")
    parser.add_argument("--detector", default="mtcnn", help="Face detector backend (mtcnn, retinaface, opencv, etc.)")
    parser.add_argument("--feature-extractor", default="deepface", choices=["deepface", "facenet"],
                      help="Feature extractor to use (deepface or facenet)")
    
    args = parser.parse_args()
    
    # Extract arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    skip_video = args.skip_video
    downsampled_dir = args.downsampled_dir
    preserve_structure = not args.no_preserve_structure
    
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Choose pattern based on directory name
    if "CREMA-D" in input_dir:
        pattern = "*.flv"
        logging.info(f"Processing CREMA-D dataset from {input_dir}")
    else:
        pattern = "*.mp4"  # RAVDESS
        logging.info(f"Processing RAVDESS dataset from {input_dir} (including subdirectories)")
    
    # Log options
    if skip_video:
        logging.info("Skipping video feature extraction (creating dummy features)")
    if downsampled_dir:
        logging.info(f"Using pre-downsampled videos from: {downsampled_dir}")
        
    # Log options
    if preserve_structure:
        logging.info(f"Preserving input directory structure in output")
    else:
        logging.info(f"Using flat output directory structure")
        
    # Process dataset
    processed_files = process_dataset(
        video_dir=input_dir, 
        pattern=pattern, 
        output_dir=output_dir, 
        model_name=args.model_name,
        detector_backend=args.detector,
        feature_extractor_type=args.feature_extractor,
        skip_video=skip_video,
        downsampled_dir=downsampled_dir,
        preserve_structure=preserve_structure
    )
    
    logging.info(f"Successfully processed {len(processed_files)} files to {output_dir}")
