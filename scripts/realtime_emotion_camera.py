#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Real-time Emotion Recognition from Camera Feed

This script provides a simplified implementation focusing on camera capture
and displaying emotion recognition results using our fixed model loading approach.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import threading
import time
import argparse
import h5py
import json
from tensorflow.keras.models import model_from_json
from facenet_extractor import FaceNetExtractor
import opensmile
from collections import deque

# Set up argument parser
parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
parser.add_argument('--model', type=str, default='models/branched_no_leakage_84_1/best_model.h5',
                    help='Path to the trained model')
parser.add_argument('--window_size', type=int, default=45,
                    help='Size of sliding window for features in frames (3 seconds at 15fps)')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
parser.add_argument('--display_width', type=int, default=800,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=600,
                    help='Height of display window')
args = parser.parse_args()

# Global variables
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
emotion_colors = [
    (0, 0, 255),     # Anger: Red (BGR)
    (0, 140, 255),   # Disgust: Orange
    (0, 255, 255),   # Fear: Yellow
    (0, 255, 0),     # Happy: Green
    (255, 255, 0),   # Neutral: Cyan
    (255, 0, 0)      # Sad: Blue
]
running = True
current_emotion = None
emotion_probabilities = None
audio_features_buffer = deque(maxlen=args.window_size)
video_features_buffer = deque(maxlen=args.window_size)

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 0.5  # Process in half-second chunks

# Frame rate control
target_fps = args.fps
frame_interval = 1.0 / target_fps
last_frame_time = 0

def direct_model_load(model_path):
    """
    Load the model using a direct file-based approach
    that avoids custom object scopes and compatibility issues
    """
    try:
        print(f"Loading model from {model_path}...")
        
        # Open the H5 file and extract the model architecture
        with h5py.File(model_path, 'r') as h5file:
            # Get model config
            if 'model_config' in h5file.attrs:
                model_config = h5file.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                config_dict = json.loads(model_config)
                
                # Clean up the model config (remove time_major parameter from LSTM)
                def clean_config(conf):
                    if isinstance(conf, dict):
                        # Check for LSTM layer config
                        if conf.get('class_name') == 'LSTM' and 'config' in conf:
                            if 'time_major' in conf['config']:
                                del conf['config']['time_major']
                                print("Removed 'time_major' parameter from LSTM config")
                        
                        # Process all items in dict recursively
                        for k, v in conf.items():
                            if isinstance(v, dict):
                                clean_config(v)
                            elif isinstance(v, list):
                                for i in v:
                                    if isinstance(i, dict):
                                        clean_config(i)
                    return conf
                
                # Clean the configuration to remove incompatible parameters
                fixed_config = clean_config(config_dict)
                
                # Create the model from the cleaned config
                model = model_from_json(json.dumps(fixed_config))
                
                # Load the weights directly from the h5 file
                model.load_weights(model_path)
                
                # Compile the model (only needed for inference)
                model.compile(loss='categorical_crossentropy', optimizer='adam')
                
                print("Model loaded successfully with custom loading approach!")
                print(f"Input shapes: {[input.shape for input in model.inputs]}")
                print(f"Output shape: {model.output.shape}")
                
                return model
            else:
                raise ValueError("No model_config found in the H5 file")
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def setup_opensmile():
    """Set up OpenSMILE for audio feature extraction."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_workers=2
    )

def setup_facenet():
    """Set up FaceNet for video feature extraction."""
    return FaceNetExtractor()

def process_audio(audio_data, smile):
    """Process audio data to extract features."""
    try:
        # Convert byte array to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Extract features using OpenSMILE
        features = smile.process_signal(audio_np, RATE)

        # Average the features across the time dimension to get one feature vector
        avg_features = features.mean(axis=0).values

        return avg_features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def process_video_frame(frame, face_extractor):
    """Process video frame to extract facial features."""
    try:
        # Extract face embeddings using FaceNet
        embedding = face_extractor.extract_features_from_frame(frame)

        if embedding is not None:
            return embedding
        else:
            return np.zeros(512)  # Return zeros if no face detected
    except Exception as e:
        print(f"Error processing video frame: {e}")
        return np.zeros(512)

def predict_emotion(model, audio_features, video_features, window_size):
    """Predict emotion from features."""
    if len(audio_features) < 5 or len(video_features) < 5:
        return None, None

    # Prepare inputs for model
    audio_input = np.array(list(audio_features))
    video_input = np.array(list(video_features))

    # Pad if necessary to match model input shape
    if len(audio_input) < window_size:
        padding = np.zeros((window_size - len(audio_input), audio_input.shape[1]))
        audio_input = np.vstack((padding, audio_input))

    if len(video_input) < window_size:
        padding = np.zeros((window_size - len(video_input), video_input.shape[1]))
        video_input = np.vstack((padding, video_input))

    # Reshape to match model input shape
    audio_input = audio_input.reshape(1, window_size, -1)
    video_input = video_input.reshape(1, window_size, -1)

    # Make prediction
    try:
        prediction = model.predict([audio_input, video_input], verbose=0)
        emotion_idx = np.argmax(prediction[0])
        return emotion_labels[emotion_idx], prediction[0]
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, None

def audio_capture_thread(smile):
    """Thread function to capture audio and extract features."""
    global running, audio_features_buffer

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    try:
        # Open stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Audio capture started")

        while running:
            # Read audio chunk
            try:
                audio_data = []
                for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
                    if not running:
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data.append(data)

                if audio_data and running:
                    # Process audio and add features to buffer
                    features = process_audio(b''.join(audio_data), smile)
                    if features is not None:
                        audio_features_buffer.append(features)
            except Exception as e:
                print(f"Error capturing audio: {e}")
                time.sleep(0.1)  # Short sleep to prevent CPU overuse on error

    finally:
        # Clean up
        if 'stream' in locals() and stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Audio capture stopped")

def main():
    """Main function to run the real-time emotion recognition system."""
    global running, last_frame_time, current_emotion, emotion_probabilities
    global audio_features_buffer, video_features_buffer

    # Initialize feature extractors
    smile = setup_opensmile()
    face_extractor = setup_facenet()

    # Load model
    model = direct_model_load(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Try several webcam indices
    cap = None
    for cam_index in [0, 1, 2]:
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully opened camera at index {cam_index}")
                break
            else:
                print(f"Camera at index {cam_index} opened but failed to read frame")
                cap.release()
                cap = None
        else:
            print(f"Could not open camera at index {cam_index}")
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam.")
        return

    # Set camera properties - attempt to match our target FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Start audio capture thread
    audio_thread = threading.Thread(target=audio_capture_thread, args=(smile,))
    audio_thread.daemon = True
    audio_thread.start()

    print(f"Starting real-time emotion recognition with target {target_fps} FPS")
    print("Press 'q' or ESC to quit")

    try:
        while running:
            # Timing for FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # Sleep to maintain target FPS
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            # Record the time we're processing this frame
            last_frame_time = current_time

            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process video frame and add features to buffer
            video_features = process_video_frame(frame, face_extractor)
            if video_features is not None:
                video_features_buffer.append(video_features)

            # Create display frame
            display = frame.copy()

            # Predict emotion
            emotion, probs = predict_emotion(model, audio_features_buffer, video_features_buffer, args.window_size)
            
            if emotion is not None and probs is not None:
                current_emotion = emotion
                emotion_probabilities = probs

            # Add FPS counter
            fps_text = f"Processing: {1.0 / elapsed:.1f} FPS (Target: {args.fps})"
            cv2.putText(display, fps_text, (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display emotion text and confidence
            if current_emotion is not None and emotion_probabilities is not None:
                # Text settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2

                # Display current emotion
                emotion_text = f"Emotion: {current_emotion}"
                confidence = emotion_probabilities[emotion_labels.index(current_emotion)]
                confidence_text = f"Confidence: {confidence:.2f}"

                # Draw emotion label
                cv2.putText(display, emotion_text, (20, 40), font, font_scale,
                            emotion_colors[emotion_labels.index(current_emotion)], thickness)

                # Draw confidence
                cv2.putText(display, confidence_text, (20, 80), font, font_scale, (255, 255, 255), thickness)

                # Draw all emotion probabilities as bars
                bar_height = 20
                bar_width = 200
                x_start = 20
                y_start = 120

                for i, emotion in enumerate(emotion_labels):
                    prob = emotion_probabilities[i]
                    filled_width = int(bar_width * prob)

                    # Draw label
                    cv2.putText(display, emotion, (x_start, y_start + i*40), font, 0.5, (255, 255, 255), 1)

                    # Draw empty bar
                    cv2.rectangle(display, (x_start, y_start + i*40 + 5),
                                 (x_start + bar_width, y_start + i*40 + 5 + bar_height),
                                 (100, 100, 100), -1)

                    # Draw filled portion
                    cv2.rectangle(display, (x_start, y_start + i*40 + 5),
                                 (x_start + filled_width, y_start + i*40 + 5 + bar_height),
                                 emotion_colors[i], -1)

            # Resize display for better viewing
            display = cv2.resize(display, (args.display_width, args.display_height))

            # Show frame
            cv2.imshow('Real-time Emotion Recognition', display)

            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                running = False
                break

    except Exception as e:
        print(f"Error in main loop: {e}")

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()

        # Wait for threads to finish
        if audio_thread.is_alive():
            audio_thread.join(timeout=1.0)

        print("Application stopped.")

if __name__ == "__main__":
    main()
