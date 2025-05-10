#!/usr/bin/env python3
"""
Run Real-time Emotion Recognition with proper layer registration
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pyaudio
import argparse
import time
import json
import h5py

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
        
        print("Model loaded successfully with patched config!")
        print(f"Input shapes: {[input.shape for input in model.inputs if hasattr(input, 'shape')]}")
        print(f"Output shape: {model.output.shape}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model', type=str, default='models/branched_no_leakage_84_1/best_model.h5',
                       help='Path to the trained model')
    args = parser.parse_args()
    
    # Test camera access first to make sure permissions are granted
    print("Testing camera access...")
    
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
    
    # Get a test frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        cap.release()
        return
    
    # If we made it here, camera works, release it for now
    cap.release()
    print("Camera access successful!")
    
    # Now try to load the model
    model = load_emotion_model(args.model)
    if model is None:
        return
    
    print("Model and camera check completed successfully!")
    print("Now you can run the full application with:")
    print("./run_realtime_emotion_recognition.sh")

if __name__ == "__main__":
    main()
