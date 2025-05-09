#!/usr/bin/env python
"""
Example script showing how to load and use the TensorFlow 2.x compatible model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow_compatible_model import EmotionRecognitionModel

print(f"TensorFlow version: {tf.__version__}")

def main():
    print("\n======= Loading TensorFlow 2.x Compatible Model =======")
    
    # Load the model using our compatible implementation
    model_path = os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5')
    model = EmotionRecognitionModel(model_path)
    
    print("\nModel successfully loaded!")
    
    # Create dummy inputs for testing
    batch_size = 1
    seq_len = 10
    video_features = np.random.random((batch_size, seq_len, 512))  # FaceNet features
    audio_features = np.random.random((batch_size, seq_len, 88))   # OpenSMILE features
    
    # Make a prediction
    print("\nMaking prediction with dummy data...")
    prediction = model.predict(video_features, audio_features)
    
    # Display emotion probabilities
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
    print("\nEmotion probabilities:")
    for i, emotion in enumerate(emotions):
        print(f"{emotion}: {prediction[0][i]:.4f}")
    
    # Get the top emotion
    top_emotion = model.get_emotion_label(prediction[0])
    print(f"\nTop predicted emotion: {top_emotion}")
    
    return 0

if __name__ == "__main__":
    main()
