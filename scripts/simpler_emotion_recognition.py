#!/usr/bin/env python3
"""
Simpler Real-time Emotion Recognition using a direct model loading approach
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
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import LSTM
import opensmile
from facenet_extractor import FaceNetExtractor
import h5py
import json
import sys

print("Using TensorFlow version:", tf.__version__)
print("Using Python version:", sys.version)

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
        return None

def test_camera():
    """Test if camera is accessible and working."""
    for cam_index in [0, 1, 2]:
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera at index {cam_index}")
                cap.release()
                return True, cam_index
            else:
                print(f"Camera at index {cam_index} opened but couldn't capture frame")
                cap.release()
        else:
            print(f"Failed to open camera at index {cam_index}")
    
    print("No working camera found")
    return False, -1

def test_model():
    """Test if the model can be loaded."""
    model = direct_model_load(args.model)
    if model is not None:
        print("Model loaded successfully!")
        return True
    else:
        print("Failed to load model")
        return False

def main():
    """Main entry point for testing."""
    camera_ok, cam_index = test_camera()
    if not camera_ok:
        print("Camera test failed - please check permissions and connections")
        return

    model_ok = test_model()
    if not model_ok:
        print("Model test failed - please check the model file")
        return
    
    print("\nAll tests passed! The system is ready to run.")
    print("Camera and model are working correctly.")
    print("You can run the full application with:")
    print("./run_realtime_emotion_recognition.sh")

if __name__ == "__main__":
    main()
