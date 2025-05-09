#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Real-time Emotion Recognition Demo

This application demonstrates the emotion recognition model in real-time using:
- Webcam for video input (downsampled to 15fps)
- Microphone for audio input
- Pre-trained branched model with dynamic padding and no leakage
- Real-time visualization of results

Requirements:
- OpenCV
- PyAudio
- TensorFlow
- OpenSMILE (for audio feature extraction)
- FaceNet-PyTorch (for facial feature extraction)
- Matplotlib
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import pyaudio
import threading
import time
import matplotlib.pyplot as plt
from collections import deque
import argparse
from tensorflow.keras.models import load_model
import opensmile
import torch
from facenet_extractor import FaceNetExtractor

# Set up argument parser
parser = argparse.ArgumentParser(description='Enhanced Real-time Emotion Recognition Demo')
parser.add_argument('--model', type=str, default='models/branched_no_leakage_84_1/best_model.h5',
                    help='Path to the trained model')
parser.add_argument('--window_size', type=int, default=30,
                    help='Size of sliding window for features (in frames)')
parser.add_argument('--display_width', type=int, default=800,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=600,
                    help='Height of display window')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
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

# FPS control variables
target_fps = args.fps
frame_interval = 1.0 / target_fps
last_frame_time = 0

def load_emotion_model(model_path):
    """Load the trained emotion recognition model."""
    try:
        model = load_model(model_path, compile=False)
        
        # Compile with dummy loss - we only use the model for inference
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        print(f"Model loaded from {model_path}")
        print(f"Input shapes: {[input.shape for input in model.inputs]}")
        print(f"Output shape: {model.output.shape}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_video_frame(frame):
    """Extract FaceNet features from video frame."""
    try:
        # Use our FaceNetExtractor class for consistent feature extraction
        features = face_extractor.extract_features(frame)
        return features
    except Exception as e:
        print(f"Error processing video frame: {e}")
        return None

def process_audio_chunk(audio_data):
    """Extract audio features from audio chunk using OpenSMILE."""
    try:
        # Convert audio data to the format expected by OpenSMILE
        signal = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Extract features
        features = smile.process_signal(signal, RATE)
        
        return features.values
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_emotion(model, video_features, audio_features):
    """Predict emotion using the trained model."""
    if len(video_features) < 5 or len(audio_features) < 5:
        # Not enough data to make a prediction
        return None, None
    
    try:
        # Convert lists to numpy arrays and ensure proper shape for the model
        video_sequence = np.array([video_features])
        audio_sequence = np.array([audio_features])
        
        # Make prediction
        prediction = model.predict([video_sequence, audio_sequence], verbose=0)
        
        # Get predicted class and probabilities
        predicted_class = np.argmax(prediction[0])
        probabilities = prediction[0]
        
        return predicted_class, probabilities
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, None

def audio_capture_thread(audio_buffer):
    """Thread for capturing audio from microphone."""
    global running
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("Audio capture started...")
    
    try:
        while running:
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not running:
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            # Process audio data
            audio_data = b''.join(frames)
            features = process_audio_chunk(audio_data)
            
            if features is not None:
                audio_buffer.append(features)
    except Exception as e:
        print(f"Error in audio capture: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio capture stopped.")

def update_plot(frame_num, emotion_history, ax):
    """Update function for the animation of emotion probabilities."""
    if not emotion_history or emotion_probabilities is None:
        return
    
    ax.clear()
    x = range(len(emotion_history))
    
    for i, emotion in enumerate(emotion_labels):
        values = [probs[i] for probs in emotion_history]
        ax.plot(x, values, label=emotion)
    
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Probabilities Over Time')
    ax.legend(loc='upper right')
    
    plt.tight_layout()

def main():
    global running, current_emotion, emotion_probabilities, emotion_history, last_frame_time
    
    print(f"Starting real-time emotion recognition at {args.fps}fps...")
    print(f"Using model: {args.model}")
    
    # Load pre-trained model
    model = load_emotion_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Start audio capture thread
    audio_thread = threading.Thread(target=audio_capture_thread, args=(audio_buffer,))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Set up real-time plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.tight_layout()
    
    try:
        while running:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Enforce frame rate (15fps) for video processing
            if elapsed < frame_interval:
                # Skip this iteration if we're processing too fast
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                continue
                
            # Record the time we're processing this frame
            last_frame_time = current_time
            
            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Process video frame
            video_features = preprocess_video_frame(frame)
            if video_features is not None:
                video_buffer.append(video_features)
            
            # If we have enough data in both buffers, make a prediction
            if len(video_buffer) > 5 and len(audio_buffer) > 5:
                pred_class, probs = predict_emotion(model, list(video_buffer), list(audio_buffer))
                
                if pred_class is not None:
                    current_emotion = emotion_labels[pred_class]
                    emotion_probabilities = probs
                    emotion_history.append(probs)
            
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
            cv2.imshow('Enhanced Real-time Emotion Recognition', display)
            
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
        
        # Wait for audio thread to finish
        if audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        
        print("Demo application stopped.")

if __name__ == "__main__":
    main()
