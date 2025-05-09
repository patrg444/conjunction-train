#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Real-time Emotion Recognition with Webcam

This script captures video from the webcam and audio from the microphone,
extracts features using FaceNet and OpenSMILE (matching the trained model's
input requirements), and performs real-time emotion prediction.

The video is downsampled to 15fps and features are extracted with the same
pipeline used in training the model.
"""

import os
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import threading
import queue
import pyaudio
import wave
import subprocess
import argparse
from collections import deque
import logging
import pandas as pd
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("realtime_emotion_recognition.log"),
        logging.StreamHandler()
    ]
)

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our custom FaceNet extractor
try:
    from scripts.facenet_extractor import FaceNetExtractor
except ImportError:
    # Try relative import
    from facenet_extractor import FaceNetExtractor

# Set up argument parser
parser = argparse.ArgumentParser(description='Real-time Emotion Recognition with Webcam')
parser.add_argument('--model', type=str, 
                    default='models/dynamic_padding_no_leakage/best_model.h5',
                    help='Path to the trained model file')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
parser.add_argument('--display_width', type=int, default=1200,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=700,
                    help='Height of display window')
parser.add_argument('--window_size', type=int, default=45,
                    help='Size of the feature window (default: 45 frames, 3 seconds at 15fps)')
parser.add_argument('--opensmile_config', type=str, 
                    default='scripts/test_opensmile_config.conf',
                    help='OpenSMILE configuration file')
parser.add_argument('--opensmile_path', type=str, 
                    default='opensmile-3.0.2-macos-armv8/bin/SMILExtract',
                    help='Path to OpenSMILE executable')
parser.add_argument('--cam_index', type=int, default=0,
                    help='Camera index to use (default: 0)')
args = parser.parse_args()

# Global variables
running = True
frame_queue = queue.Queue(maxsize=30)  # Queue for frames
audio_queue = queue.Queue(maxsize=30)  # Queue for audio chunks
feature_lock = threading.Lock()        # Lock for feature access
video_features = None                  # Latest video features
audio_features = None                  # Latest audio features
predicted_emotion = "Starting..."      # Latest emotion prediction
prediction_confidence = 0.0            # Confidence in the prediction
raw_emotion_probs = [0.0] * 6          # Raw emotion probabilities

# Emotion labels and colors (using the same set as in the training data)
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
emotion_colors = [
    (0, 0, 255),     # Anger: Red (BGR)
    (0, 140, 255),   # Disgust: Orange
    (0, 255, 255),   # Fear: Yellow
    (0, 255, 0),     # Happy: Green
    (255, 255, 0),   # Neutral: Cyan
    (255, 0, 0)      # Sad: Blue
]

# Smoothing buffers for each emotion probability
smoothing_window_size = args.window_size
emotion_buffers = [deque(maxlen=smoothing_window_size) for _ in range(len(emotions))]

# Initialize the buffers with zeros
for buffer in emotion_buffers:
    for _ in range(smoothing_window_size):
        buffer.append(0.0)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz sampling rate
TEMP_AUDIO_DIR = "temp_extracted_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def load_model(model_path):
    """Load the trained emotion recognition model."""
    try:
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully")
        # Print model summary
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try alternative loading approach
        try:
            print("Trying alternative loading approach...")
            # The model was trained with custom objects, try to load it with a custom scope
            with tf.keras.utils.custom_object_scope({'Functional': tf.keras.models.Model}):
                model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully with alternative approach")
            model.summary()
            return model
        except Exception as e2:
            print(f"Error with alternative loading approach: {e2}")
            sys.exit(1)

def video_capture_thread():
    """Thread to capture video frames from the webcam with precise 15fps timing."""
    global running, frame_queue
    
    # Initialize camera
    try:
        cap = cv2.VideoCapture(args.cam_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera at index {args.cam_index}.")
            running = False
            return
    except Exception as e:
        print(f"Error initializing camera: {e}")
        running = False
        return
        
    # Try to set camera properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera default FPS: {original_fps}")
    
    # Try to set the camera to the target FPS (may not work on all cameras)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get the actual FPS after trying to set it
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS after setting: {actual_fps}")
    
    # Calculate frame interval for consistent timing
    target_fps = args.fps
    frame_interval = 1.0 / target_fps  # Time between frames at target FPS
    
    # Calculate sample interval (for frame skipping if needed)
    if actual_fps > target_fps * 1.1:  # If camera fps is significantly higher
        sample_interval = max(1, round(actual_fps / target_fps))
        print(f"Using frame sampling: every {sample_interval} frames (for {actual_fps / sample_interval:.1f} fps)")
    else:
        sample_interval = 1
        print(f"Processing all frames with timing control (target: {target_fps} fps)")
    
    try:
        frame_count = 0
        last_frame_time = time.time()
        last_fps_check_time = time.time()
        frames_processed = 0
        
        while running:
            # Timing for FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Maintain target FPS
            if elapsed < frame_interval:
                # Sleep to maintain target FPS (rather than busy-waiting)
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:  # Only sleep for meaningful intervals
                    time.sleep(sleep_time)
                continue
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from camera.")
                time.sleep(0.1)  # Small delay to avoid hammering if there's an issue
                continue
            
            # Only process frames at the sample interval
            frame_count += 1
            if frame_count % sample_interval == 0:
                # Put the frame in the queue if there's room
                if not frame_queue.full():
                    frame_queue.put(frame)
                else:
                    # If the queue is full, we're processing frames too slowly
                    # Discard the oldest frame to make room
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(frame)
                    except queue.Empty:
                        pass
                        
                # Update timing and statistics
                last_frame_time = current_time
                frames_processed += 1
                
                # Calculate and print actual FPS periodically
                if current_time - last_fps_check_time > 5.0:  # Every 5 seconds
                    actual_fps = frames_processed / (current_time - last_fps_check_time)
                    print(f"Actual capture rate: {actual_fps:.1f} fps")
                    frames_processed = 0
                    last_fps_check_time = current_time
    
    except Exception as e:
        print(f"Error in video capture thread: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cap.release()
        running = False
        print("Video capture thread stopped.")

def audio_capture_thread():
    """Thread to capture audio from the microphone."""
    global running, audio_queue
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open audio stream
        stream = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)
        
        print("Audio stream opened successfully")
        
        while running:
            try:
                # Read audio data
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Put audio data in the queue if there's room
                if not audio_queue.full():
                    audio_queue.put(audio_data)
                else:
                    # If the queue is full, discard the oldest data
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put(audio_data)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                print(f"Error reading audio stream: {e}")
                time.sleep(0.1)  # Sleep to prevent hammering if there's an error
                
    except Exception as e:
        print(f"Error in audio capture thread: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass
        running = False
        print("Audio capture thread stopped")

def extract_opensmile_features(audio_file):
    """Extract audio features using OpenSMILE."""
    try:
        output_file = os.path.join(TEMP_AUDIO_DIR, "features.csv")
        
        # Find the OpenSMILE configuration file
        config_file = find_opensmile_config(args.opensmile_config)
        if not config_file:
            print(f"Error: Could not find OpenSMILE config file: {args.opensmile_config}")
            return None
            
        # Find the OpenSMILE executable
        opensmile_path = find_opensmile_executable(args.opensmile_path)
        if not opensmile_path:
            print(f"Error: Could not find OpenSMILE executable: {args.opensmile_path}")
            return None
        
        # Construct the OpenSMILE command
        opensmile_cmd = [
            opensmile_path,
            "-C", config_file,
            "-I", audio_file,
            "-csvoutput", output_file
        ]
        
        # Run OpenSMILE
        result = subprocess.run(opensmile_cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        
        if result.returncode != 0:
            print(f"OpenSMILE error: {result.stderr}")
            return None
            
        # Read the features from the CSV file
        try:
            # First check what we got
            with open(output_file, 'r') as f:
                header = f.readline().strip()
            
            # Count the number of features
            feature_count = len(header.split(';')) - 1  # Subtract 1 for 'name' column
            
            # Read the CSV with pandas
            df = pd.read_csv(output_file, sep=';')
            
            # Extract features (first column is name, we drop it)
            if len(df) > 0:
                feature_names = df.columns[1:]  # Skip the 'name' column
                features = df[feature_names].values
                
                # Return features reshaped to the format needed by the model
                return features
            else:
                print("Empty CSV file")
                return None
                
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error extracting audio functionals: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def find_opensmile_config(config_path):
    """Find the OpenSMILE configuration file."""
    # Check if the provided path exists directly
    if os.path.exists(config_path):
        return config_path
        
    # Try several possible locations
    possible_paths = [
        config_path,
        os.path.join('scripts', config_path),
        os.path.join('opensmile-3.0.2-macos-armv8', 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf'),
        os.path.join('opensmile-3.0.2-macos-armv8', 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf'),
        os.path.join('..', 'opensmile-3.0.2-macos-armv8', 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf'),
        os.path.join(parent_dir, 'opensmile-3.0.2-macos-armv8', 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf'),
        os.path.join(parent_dir, 'scripts', 'test_opensmile_config.conf'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found OpenSMILE config at: {path}")
            return path
            
    return None

def find_opensmile_executable(opensmile_path):
    """Find the OpenSMILE executable."""
    # Check if the provided path exists directly
    if os.path.exists(opensmile_path):
        return opensmile_path
        
    # Try several possible locations
    possible_paths = [
        opensmile_path,
        os.path.join('opensmile-3.0.2-macos-armv8', 'bin', 'SMILExtract'),
        os.path.join('..', 'opensmile-3.0.2-macos-armv8', 'bin', 'SMILExtract'),
        os.path.join(parent_dir, 'opensmile-3.0.2-macos-armv8', 'bin', 'SMILExtract'),
        'SMILExtract',  # If in PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found OpenSMILE executable at: {path}")
            return path
            
    return None

def video_feature_extraction_thread():
    """Thread to extract features from video frames using FaceNet."""
    global running, frame_queue, video_features, feature_lock
    
    try:
        # Initialize FaceNet extractor
        facenet_extractor = FaceNetExtractor(
            keep_all=False,  # Only keep the largest face
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]  # MTCNN thresholds
        )
        print(f"Initialized FaceNet extractor with embedding dimension: {facenet_extractor.embedding_dim}")
        
        # Buffer to collect frames for batch processing
        frame_buffer = []
        embedding_dim = facenet_extractor.embedding_dim
        
        # Initialize video feature buffer (for model input)
        all_video_features = np.zeros((args.window_size, embedding_dim))
        feature_index = 0
        
        while running:
            # Get a frame if available
            try:
                if not frame_queue.empty():
                    frame = frame_queue.get_nowait()
                    
                    # Extract features from the frame
                    try:
                        # Convert to RGB for face detection
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Extract face embedding
                        embedding = facenet_extractor.extract_features(rgb_frame)
                        
                        # Update the feature buffer
                        all_video_features[feature_index % args.window_size] = embedding
                        feature_index += 1
                        
                        # Update global video features
                        with feature_lock:
                            video_features = all_video_features.copy()
                            
                    except Exception as e:
                        print(f"Error extracting features from frame: {e}")
                        # On error, use zeros as fallback
                        all_video_features[feature_index % args.window_size] = np.zeros(embedding_dim)
                        feature_index += 1
                        with feature_lock:
                            video_features = all_video_features.copy()
            
            except queue.Empty:
                time.sleep(0.01)  # Sleep a bit if no frames
                continue
            
            # Avoid using too much CPU
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Error in video feature extraction thread: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        running = False
        print("Video feature extraction thread stopped")

def audio_feature_extraction_thread():
    """Thread to extract features from audio using OpenSMILE."""
    global running, audio_queue, audio_features, feature_lock
    
    try:
        # Buffer to store audio chunks
        audio_buffer = []
        
        # Audio file for OpenSMILE processing
        temp_audio_file = os.path.join(TEMP_AUDIO_DIR, "temp_audio.wav")
        
        # Initialize audio feature buffer
        # Using a placeholder size of 88 features (standard eGeMAPS size)
        all_audio_features = np.zeros((1, 88))
        
        # Loop waiting for audio data
        while running:
            # Get audio data if available
            try:
                if not audio_queue.empty():
                    audio_data = audio_queue.get_nowait()
                    audio_buffer.append(audio_data)
                    
                    # Process audio after collecting enough data (around 3 seconds)
                    # 16000 Hz * 3 seconds / 1024 samples per chunk ≈ 47 chunks
                    if len(audio_buffer) >= 47:
                        # Save audio to a WAV file
                        try:
                            with wave.open(temp_audio_file, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(2)  # 16-bit audio
                                wf.setframerate(RATE)
                                wf.writeframes(b''.join(audio_buffer))
                            
                            # Extract OpenSMILE features
                            features = extract_opensmile_features(temp_audio_file)
                            
                            if features is not None and features.size > 0:
                                # Update audio features
                                with feature_lock:
                                    audio_features = features
                                    all_audio_features = features
                            
                            # Clear part of the buffer (sliding window approach)
                            # Remove oldest 15 chunks (about 1 second) for overlap
                            audio_buffer = audio_buffer[15:]
                            
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            audio_buffer = audio_buffer[15:]  # Still clear buffer on error
                            
            except queue.Empty:
                time.sleep(0.01)  # Sleep a bit if no audio
                continue
            
            # Avoid using too much CPU
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Error in audio feature extraction thread: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        running = False
        print("Audio feature extraction thread stopped")

def prediction_thread(model):
    """Thread to run the emotion recognition model on extracted features."""
    global running, video_features, audio_features, feature_lock, predicted_emotion
    global prediction_confidence, emotion_buffers, raw_emotion_probs
    
    try:
        # Wait for initial features to be available
        print("Waiting for initial features to be available...")
        features_available = False
        while running and not features_available:
            with feature_lock:
                features_available = (video_features is not None and audio_features is not None)
            if not features_available:
                time.sleep(0.1)
                
        print("Features available, starting prediction...")
        
        while running:
            # Check if we have both video and audio features
            with feature_lock:
                v_features = video_features
                a_features = audio_features
            
            if v_features is not None and a_features is not None:
                try:
                    # Make prediction using the model
                    video_input = np.expand_dims(v_features, axis=0)  # Add batch dimension
                    audio_input = np.expand_dims(a_features, axis=0)  # Add batch dimension
                    
                    # Make prediction
                    prediction = model.predict([video_input, audio_input], verbose=0)
                    
                    # Get predicted emotion class and confidence
                    if isinstance(prediction, list):
                        pred_probs = prediction[0]  # First element may be the classification output
                    else:
                        pred_probs = prediction
                    
                    # Extract probabilities from the first sample in the batch
                    if len(pred_probs.shape) > 1:
                        pred_probs = pred_probs[0]
                    
                    # Store the raw probabilities
                    raw_emotion_probs = pred_probs
                    
                    # Update smoothing buffers
                    for i, prob in enumerate(pred_probs):
                        if i < len(emotions):
                            emotion_buffers[i].append(float(prob))
                    
                    # Calculate smoothed probabilities
                    smoothed_probs = [sum(buffer) / len(buffer) for buffer in emotion_buffers]
                    
                    # Get predicted emotion
                    pred_class = np.argmax(smoothed_probs)
                    pred_confidence = smoothed_probs[pred_class]
                    
                    # Update global variables
                    predicted_emotion = emotions[pred_class]
                    prediction_confidence = pred_confidence
                    
                except Exception as e:
                    print(f"Error making prediction: {e}")
            
            # Don't run predictions too frequently
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in prediction thread: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        running = False
        print("Prediction thread stopped")

def display_thread():
    """Thread to display the video with emotion prediction overlay."""
    global running, frame_queue, predicted_emotion, prediction_confidence, emotion_buffers, raw_emotion_probs
    
    try:
        # Create window
        cv2.namedWindow('Real-time Emotion Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Emotion Recognition', args.display_width, args.display_height)
        
        # Frame rate control
        target_fps = args.fps
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        
        # Detection statistics
        face_detected_count = 0
        total_frames = 0
        
        while running:
            # Timing for FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Sleep to maintain target FPS
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            
            # Update frame timing
            last_frame_time = current_time
            
            # Get a frame if available
            try:
                if not frame_queue.empty():
                    # Get the newest frame (peek without removing)
                    frame = frame_queue.queue[-1].copy()
                    
                    # Update statistics
                    total_frames += 1
                    
                    # Create display frame
                    display_height = args.display_height
                    display_width = args.display_width
                    display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                    
                    # Resize the frame to fit the left side of the display
                    frame_height = display_height
                    frame_width = int(frame.shape[1] * (frame_height / frame.shape[0]))
                    frame_resized = cv2.resize(frame, (frame_width, frame_height))
                    
                    # Place the frame on the left side of the display
                    if frame_width <= display_width // 2:
                        display[0:frame_height, 0:frame_width] = frame_resized
                    else:
                        # If the frame is too wide, resize it further
                        frame_width = display_width // 2
                        frame_resized = cv2.resize(frame, (frame_width, frame_height))
                        display[0:frame_height, 0:frame_width] = frame_resized
                    
                    # Calculate smoothed emotion probabilities
                    smoothed_probs = [sum(buffer) / len(buffer) for buffer in emotion_buffers]
                    
                    # Find dominant emotion
                    dominant_emotion_idx = np.argmax(smoothed_probs)
                    dominant_color = emotion_colors[dominant_emotion_idx]
                    
                    # Add emotion prediction text
                    cv2.putText(display, f"Emotion: {predicted_emotion}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, dominant_color, 2)
                    
                    # Add confidence display
                    cv2.putText(display, f"Confidence: {prediction_confidence:.2f}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add face detection indicator (based on non-zero video features)
                    with feature_lock:
                        if video_features is not None and np.any(video_features != 0):
                            cv2.putText(display, "Face Detected", (20, 120), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            face_detected_count += 1
                        else:
                            cv2.putText(display, "No Face Detected", (20, 120), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Add detection rate
                    detection_rate = (face_detected_count / max(1, total_frames)) * 100
                    cv2.putText(display, f"Detection Rate: {detection_rate:.1f}%", (20, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    
                    # Draw emotions visualization on the right side
                    # Add title for emotions
                    cv2.putText(display, "Emotion Probabilities", (display_width//2 + 50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add subtitles for raw and smoothed
                    cv2.putText(display, "Raw", (display_width//2 + 100, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.putText(display, "Smoothed", (display_width//2 + 250, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    
                    # Draw graph for each emotion
                    bar_height = 30
                    raw_bar_width = 100
                    smoothed_bar_width = 100
                    x_start = display_width//2 + 50
                    y_start = 100
                    
                    for i, emotion in enumerate(emotions):
                        # Draw emotion label
                        cv2.putText(display, emotion, (x_start, y_start + i*60 + bar_height//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw raw value bar background
                        cv2.rectangle(display, 
                                      (x_start + 100, y_start + i*60),
                                      (x_start + 100 + raw_bar_width, y_start + i*60 + bar_height),
                                      (50, 50, 50), -1)
                        
                        # Draw raw value bar fill
                        raw_val = 0.0
                        if i < len(raw_emotion_probs):
                            raw_val = raw_emotion_probs[i]
                        raw_width = int(raw_bar_width * raw_val)
                        cv2.rectangle(display, 
                                      (x_start + 100, y_start + i*60),
                                      (x_start + 100 + raw_width, y_start + i*60 + bar_height),
                                      emotion_colors[i], -1)
                        
                        # Draw raw value text
                        cv2.putText(display, f"{raw_val:.2f}", 
                                   (x_start + 100 + raw_bar_width + 10, y_start + i*60 + bar_height - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # Draw smoothed value bar background
                        cv2.rectangle(display, 
                                      (x_start + 250, y_start + i*60),
                                      (x_start + 250 + smoothed_bar_width, y_start + i*60 + bar_height),
                                      (50, 50, 50), -1)
                        
                        # Draw smoothed value bar fill
                        smoothed_val = smoothed_probs[i]
                        smoothed_width = int(smoothed_bar_width * smoothed_val)
                        cv2.rectangle(display, 
                                      (x_start + 250, y_start + i*60),
                                      (x_start + 250 + smoothed_width, y_start + i*60 + bar_height),
                                      emotion_colors[i], -1)
                        
                        # Draw smoothed value text
                        cv2.putText(display, f"{smoothed_val:.2f}", 
                                   (x_start + 250 + smoothed_bar_width + 10, y_start + i*60 + bar_height - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Add performance info
                    processing_fps = frame_count / (time.time() - start_time) if time.time() > start_time else 0
                    cv2.putText(display, f"Processing: {processing_fps:.1f} FPS", 
                               (20, display_height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Add instructions
                    cv2.putText(display, "Press 'q' or ESC to quit", 
                               (display_width - 250, display_height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show display
                    cv2.imshow('Real-time Emotion Recognition', display)
                    
                    # Check for keypress
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        running = False
                        break
            
            except Exception as e:
                print(f"Error in display loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
    except Exception as e:
        print(f"Error in display thread: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        cv2.destroyAllWindows()
        running = False
        print("Display thread stopped")

def main():
    """Main function to run the real-time emotion recognition system."""
    global running

    # Create output directory for temporary files
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    try:
        # Load the model
        model = load_model(args.model)
        
        # Start threads
        threads = []
        
        # Start video capture thread
        video_thread = threading.Thread(target=video_capture_thread)
        video_thread.daemon = True
        video_thread.start()
        threads.append(video_thread)
        
        # Start audio capture thread
        audio_thread = threading.Thread(target=audio_capture_thread)
        audio_thread.daemon = True
        audio_thread.start()
        threads.append(audio_thread)
        
        # Start feature extraction threads
        video_feature_thread = threading.Thread(target=video_feature_extraction_thread)
        video_feature_thread.daemon = True
        video_feature_thread.start()
        threads.append(video_feature_thread)
        
        audio_feature_thread = threading.Thread(target=audio_feature_extraction_thread)
        audio_feature_thread.daemon = True
        audio_feature_thread.start()
        threads.append(audio_feature_thread)
        
        # Start prediction thread
        pred_thread = threading.Thread(target=prediction_thread, args=(model,))
        pred_thread.daemon = True
        pred_thread.start()
        threads.append(pred_thread)
        
        # Start display thread (in main thread)
        display_thread()
        
        # Join threads
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        running = False
        print("Shutting down...")
        # Remove temp directory
        try:
            for file in os.listdir(TEMP_AUDIO_DIR):
                file_path = os.path.join(TEMP_AUDIO_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
