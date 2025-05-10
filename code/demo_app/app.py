#!/usr/bin/env python3
"""
Flask web application server for the emotion recognition demo with WebGazer integration.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import base64
import json
import time
from flask import Flask, render_template, request, jsonify, Response
import io
import soundfile as sf
import threading
from collections import deque
import logging

# Add parent directory to path so we can import facenet_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.facenet_extractor import FaceNetExtractor

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'models/dynamic_padding_no_leakage/model_best.h5'
TARGET_FPS = 15  # Target 15fps for video processing
WINDOW_SIZE = 30  # Sliding window for features
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and buffers
model = None
face_extractor = None
video_buffer = deque(maxlen=WINDOW_SIZE)  # Buffer for video frames
audio_buffer = deque(maxlen=WINDOW_SIZE)  # Buffer for audio frames
current_predictions = {label: 0.0 for label in EMOTION_LABELS}
lock = threading.Lock()  # For thread safety when accessing buffers

def load_model(model_path):
    """Load the trained emotion recognition model."""
    try:
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        logger.info(f"Model loaded successfully with input shapes: {[input.shape for input in model.inputs]}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def process_image(image_data):
    """
    Process an image from base64 encoding and extract face features.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        Face embedding features
    """
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Extract face features using FaceNet
        if face_extractor:
            features = face_extractor.extract_features(frame)
            return features, frame
        else:
            logger.error("Face extractor not initialized")
            return np.zeros(512), frame
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return np.zeros(512), None

def process_audio(audio_data):
    """
    Process audio data and extract audio features.
    
    Args:
        audio_data: Base64 encoded audio data
        
    Returns:
        Audio features (using a placeholder implementation for now)
    """
    try:
        # In a real implementation, we would use OpenSMILE here
        # For the demo, we'll use some basic features
        audio_bytes = base64.b64decode(audio_data)
        
        # Create a mock audio feature vector (88-dim like openSMILE GeMAPSv01b)
        # This is just a placeholder - in real use, we'd extract proper features
        features = np.zeros(88)
        
        return features
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return np.zeros(88)

def predict_emotion():
    """
    Make an emotion prediction based on current audio and video buffers.
    
    Returns:
        dict: Predicted emotion probabilities
    """
    global current_predictions
    
    try:
        with lock:
            if len(video_buffer) < 5 or len(audio_buffer) < 5:
                logger.info("Not enough data in buffers for prediction")
                return current_predictions
            
            # Prepare inputs for model
            video_features = np.array([list(video_buffer)])
            audio_features = np.array([list(audio_buffer)])
            
            # Make prediction
            prediction = model.predict([video_features, audio_features], verbose=0)
            
            # Update current predictions
            pred_dict = {EMOTION_LABELS[i]: float(prediction[0][i]) 
                        for i in range(len(EMOTION_LABELS))}
            current_predictions = pred_dict
            
            logger.debug(f"Prediction: {pred_dict}")
            return pred_dict
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        return current_predictions

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process a video frame and return emotion predictions.
    
    Expects JSON with:
    - image: Base64 encoded image
    - audio: Base64 encoded audio (optional)
    
    Returns:
    - JSON with emotion probabilities
    """
    data = request.json
    
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Process video frame
    video_features, frame = process_image(data['image'])
    
    # Process audio if provided
    audio_features = np.zeros(88)  # Default empty features
    if 'audio' in data and data['audio']:
        audio_features = process_audio(data['audio'])
    
    # Add to buffers
    with lock:
        video_buffer.append(video_features)
        audio_buffer.append(audio_features)
    
    # Make prediction
    predictions = predict_emotion()
    
    # Get most confident emotion
    top_emotion = max(predictions.items(), key=lambda x: x[1])
    confidence = top_emotion[1]
    label = top_emotion[0]
    
    # Calculate an engagement score based on emotion and confidence
    # (this is a simplified example for demo purposes)
    engagement = min(100, int((confidence * 100) + 
                             (25 if label in ['Happy', 'Fear'] else 0) + 
                             (np.random.normal(0, 5))))
    
    # Build response
    response = {
        'emotions': predictions,
        'top_emotion': label,
        'confidence': confidence,
        'engagement_metrics': {
            'score': engagement,
            'attention_duration': f"{min(10, len(video_buffer) * 0.5):.1f}s",
            'emotional_impact': min(100, int(confidence * 100 + np.random.normal(0, 10)))
        }
    }
    
    return jsonify(response)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Handle calibration data from WebGazer.
    
    This is a placeholder endpoint that would normally store calibration data.
    """
    return jsonify({'status': 'success'})

def initialize():
    """Initialize the model and face extractor."""
    global model, face_extractor
    
    # Load emotion recognition model
    model = load_model(MODEL_PATH)
    if model is None:
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Initialize face extractor
    face_extractor = FaceNetExtractor()
    logger.info("Face extractor initialized")

if __name__ == '__main__':
    initialize()
    app.run(debug=True)
