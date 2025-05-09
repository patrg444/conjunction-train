#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Real-time Emotion Recognition from Webcam

This script captures video from the webcam extracts facial features using FaceNet
extracts audio features using OpenSMILE and feeds them into a trained emotion
recognition model to classify emotions in real-time.

The model used is the branched_no_leakage model with dynamic padding which
operates on features extracted from 15fps downsampled video and regular audio.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import threading
import time
import queue
import matplotlib.pyplot as plt
from collections import deque
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM  # Import LSTM explicitly
import opensmile
from facenet_extractor import FaceNetExtractor
import json
import h5py

# Register LSTM layer to fix deserialization issue
tf.keras.utils.get_custom_objects()['LSTM'] = LSTM

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
audio_buffer = deque(maxlen=args.window_size)
video_buffer = deque(maxlen=args.window_size)
emotion_history = deque(maxlen=100)  # For plotting emotion over time
processing_queue = queue.Queue(maxsize=10)

# Set up OpenSMILE for audio feature extraction
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    num_workers=2
)

# Initialize FaceNet extractor
face_extractor = FaceNetExtractor()

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

def patch_lstm_config(config):
    """Remove 'time_major' from LSTM config if present."""
    if isinstance(config, dict):
        # Remove time_major from LSTM config
        if config.get('class_name') == 'LSTM' and 'time_major' in config.get('config', {}):
            del config['config']['time_major']
        
        # Recursively process nested dictionaries
        for key, value in config.items():
            if isinstance(value, dict):
                patch_lstm_config(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        patch_lstm_config(item)
    
    return config

def load_emotion_model(model_path):
    """Load the trained emotion recognition model."""
    try:
        print(f"Loading model from {model_path}...")
        
        # First try to load the model using standard approach
        try:
            model = load_model(model_path, compile=False)
            print("Model loaded successfully using standard method!")
            return model
        except Exception as e:
            print(f"Standard loading failed: {e}, trying alternative approach...")
        
        # If that fails, try patching the model config
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError("No model_config found in the H5 file")
            
            config_dict = json.loads(model_config.decode('utf-8'))
            patched_config = patch_lstm_config(config_dict)
            
            # Save the modified config to a temp file
            temp_file = f"{model_path}_temp.json"
            with open(temp_file, 'w') as config_file:
                json.dump(patched_config, config_file)
            
            # Now create the model from the patched config and load weights
            from tensorflow.keras.models import model_from_json
            model = model_from_json(json.dumps(patched_config))
            model.load_weights(model_path)
            
            # Clean up
            os.remove(temp_file)
        
        # Compile with dummy loss - we only use the model for inference
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        print(f"Model loaded successfully")
        print(f"Input shapes: {[input.shape for input in model.inputs]}")
        print(f"Output shape: {model.output.shape}")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def audio_capture_thread():
    """Thread function to capture audio from microphone."""
    global running, audio_buffer

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
                    # Add to processing queue
                    processing_queue.put(('audio', b''.join(audio_data)))
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

def process_audio(audio_data):
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

def process_video_frame(frame):
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

def processing_thread():
    """Thread function to process audio and video data."""
    global running, audio_buffer, video_buffer, current_emotion, emotion_probabilities, emotion_history

    model = load_emotion_model(args.model)
    if model is None:
        print("Failed to load model. Exiting processing thread.")
        running = False
        return

    print("Processing thread started")

    while running:
        try:
            # Get item from queue with timeout
            try:
                data_type, data = processing_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Process based on data type
            if data_type == 'audio':
                audio_features = process_audio(data)
                if audio_features is not None:
                    audio_buffer.append(audio_features)

            elif data_type == 'video':
                video_features = process_video_frame(data)
                if video_features is not None:
                    video_buffer.append(video_features)

            # If we have enough data predict emotion
            if len(audio_buffer) >= 5 and len(video_buffer) >= 5:
                # Prepare inputs for model
                audio_input = np.array(list(audio_buffer))
                video_input = np.array(list(video_buffer))

                # Pad if necessary to match model input shape
                if len(audio_input) < args.window_size:
                    padding = np.zeros((args.window_size - len(audio_input), audio_input.shape[1]))
                    audio_input = np.vstack((padding, audio_input))

                if len(video_input) < args.window_size:
                    padding = np.zeros((args.window_size - len(video_input), video_input.shape[1]))
                    video_input = np.vstack((padding, video_input))

                # Reshape to match model input shape
                audio_input = audio_input.reshape(1, args.window_size, -1)
                video_input = video_input.reshape(1, args.window_size, -1)

                # Make prediction
                prediction = model.predict([audio_input, video_input], verbose=0)

                # Update global variables with result
                emotion_probabilities = prediction[0]
                current_emotion = emotion_labels[np.argmax(emotion_probabilities)]

                # Add to history for plotting
                emotion_history.append(emotion_probabilities)

            # Mark task as done
            processing_queue.task_done()

        except Exception as e:
            print(f"Error in processing thread: {e}")

    print("Processing thread stopped")

def update_plot(frame, emotion_history, ax):
    """Update plot with emotion probability history."""
    if not emotion_history:
        return

    try:
        # Clear previous plot
        ax.clear()

        # Convert deque to numpy array
        data = np.array(list(emotion_history))

        # Plot each emotion probability over time
        for i, emotion in enumerate(emotion_labels):
            ax.plot(data[:, i], label=emotion, color=tuple(c/255 for c in emotion_colors[i][::-1]))

        # Set plot properties
        ax.set_ylim(0, 1)
        ax.set_title('Emotion Probability Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper right')

        # Remove x tick labels to save space
        ax.set_xticklabels([])

    except Exception as e:
        print(f"Error updating plot: {e}")

def main():
    """Main function to run the real-time emotion recognition system."""
    global running, last_frame_time

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

    # Initialize plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.tight_layout()

    # Start audio capture thread
    audio_thread = threading.Thread(target=audio_capture_thread)
    audio_thread.daemon = True
    audio_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(target=processing_thread)
    proc_thread.daemon = True
    proc_thread.start()

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

            # Add to processing queue
            processing_queue.put(('video', frame.copy()))

            # Create display frame
            display = frame.copy()

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

            # Update plot if we have new data
            if len(emotion_history) > 0:
                update_plot(0, emotion_history, ax)
                plt.pause(0.01)

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
        plt.close()

        # Wait for threads to finish
        if audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        if proc_thread.is_alive():
            proc_thread.join(timeout=1.0)

        print("Application stopped.")

if __name__ == "__main__":
    main()
