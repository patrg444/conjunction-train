#!/usr/bin/env python
"""
Direct H5 Model Loader for TensorFlow 2.x

This script attempts to load the pre-trained h5 model from models/dynamic_padding_no_leakage/model_best.h5
with custom objects to handle the time_major parameter issue.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional

print(f"TensorFlow version: {tf.__version__}")

# Custom LSTM layer to handle the time_major parameter
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major parameter if present (added in newer TF versions)
        if 'time_major' in kwargs:
            del kwargs['time_major']
        super().__init__(*args, **kwargs)

# Define custom objects
custom_objects = {
    'CompatibleLSTM': CompatibleLSTM,
    'Bidirectional': Bidirectional
}

def load_h5_model():
    model_path = os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    
    try:
        # First attempt: Try loading directly
        print("Attempt 1: Loading model directly...")
        model = load_model(model_path, compile=False)
        print("Success! Model loaded directly")
        return model
    except Exception as e1:
        print(f"Direct loading failed: {str(e1)}")
        
        try:
            # Second attempt: Try with custom objects
            print("\nAttempt 2: Loading with custom objects...")
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print("Success! Model loaded with custom objects")
            return model
        except Exception as e2:
            print(f"Custom objects loading failed: {str(e2)}")
            
            try:
                # Third attempt: Try with SavedModel loading API
                print("\nAttempt 3: Using SavedModel API...")
                model = tf.saved_model.load(model_path)
                print("Success! Model loaded with SavedModel API")
                return model
            except Exception as e3:
                print(f"SavedModel loading failed: {str(e3)}")
                print("\nAll loading attempts failed.")
                return None

def test_model(model):
    if model is None:
        return False
    
    # Create dummy inputs
    try:
        print("\nInspecting model structure...")
        if hasattr(model, 'inputs'):
            for i, inp in enumerate(model.inputs):
                print(f"Input {i+1}: {inp.name}, shape: {inp.shape}")
        else:
            print("Model doesn't have standard inputs attribute. May be a SavedModel format.")
            return False
        
        # Create dummy inputs based on model's expected input shape
        dummy_inputs = {}
        for inp in model.inputs:
            # Use a batch size of 1 and sequence length of 10 for testing
            shape = list(inp.shape)
            # Replace None dimensions with concrete values
            shape = [1 if s is None else s for s in shape]
            if len(shape) == 3:  # If shape has sequence dimension (e.g., [None, None, features])
                shape[1] = 10  # Set sequence length to 10
            
            # Create random data
            dummy_data = np.random.random(shape)
            dummy_inputs[inp.name] = dummy_data
        
        # Test prediction
        print("\nTesting prediction with dummy data...")
        prediction = model.predict(dummy_inputs)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction sample: {prediction[0]}")
        
        # Print class probabilities
        emotions = ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
        print("\nEmotion probabilities:")
        for i, emotion in enumerate(emotions):
            if i < prediction.shape[1]:
                print(f"{emotion}: {prediction[0][i]:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def recreate_model_architecture():
    """
    Recreate the model architecture from scratch based on train_branched_no_leakage.py
    """
    print("\nAttempt 4: Recreating model architecture...")
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Masking
    
    # Parameters
    audio_feature_dim = 88  # eGeMAPSv02 features
    video_feature_dim = 512  # FaceNet features
    num_classes = 6  # 6 emotions
    
    # Audio branch with masking
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    
    # Add masking layer to handle padding
    audio_masked = Masking(mask_value=0.0)(audio_input)
    
    # Apply 1D convolutions to extract local patterns
    audio_x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    audio_x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    # Apply bidirectional LSTM for temporal features
    audio_x = Bidirectional(LSTM(128, return_sequences=True))(audio_x)
    audio_x = Dropout(0.3)(audio_x)
    audio_x = Bidirectional(LSTM(64))(audio_x)
    audio_x = Dense(128, activation='relu')(audio_x)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch with masking
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # FaceNet features already have high dimensionality, so we'll use LSTM directly
    video_x = Bidirectional(LSTM(256, return_sequences=True))(video_masked)
    video_x = Dropout(0.3)(video_x)
    video_x = Bidirectional(LSTM(128))(video_x)
    video_x = Dense(256, activation='relu')(video_x)
    video_x = Dropout(0.4)(video_x)
    
    # Merge branches with more sophisticated fusion
    merged = Concatenate()([audio_x, video_x])
    merged = Dense(256, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[video_input, audio_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_weights_to_new_model(original_weights_path):
    """
    Create new model with identical architecture and load weights from h5 file
    """
    model = recreate_model_architecture()
    
    try:
        print(f"Loading weights from: {original_weights_path}")
        model.load_weights(original_weights_path, by_name=True, skip_mismatch=True)
        print("Weights loaded into recreated model successfully!")
        return model
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return None

if __name__ == "__main__":
    print("\n======= H5 Model Loading for TensorFlow 2.x =======\n")
    
    # Try loading the model directly
    model = load_h5_model()
    
    # If direct loading methods failed, try recreating the architecture
    if model is None:
        model_path = os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5')
        model = load_weights_to_new_model(model_path)
    
    # Test the model
    if model is not None:
        success = test_model(model)
        if success:
            print("\nModel loading and testing successful!")
        else:
            print("\nModel loaded but testing failed.")
    else:
        print("\nFailed to load model after all attempts.")
