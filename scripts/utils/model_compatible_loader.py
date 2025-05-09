#!/usr/bin/env python
"""
Compatible Model Loader

This script creates a compatible model architecture matching train_branched_no_leakage.py
and loads the weights directly from the h5 file using current TensorFlow versions.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking

# Display TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

def create_model_architecture(audio_feature_dim=88, video_feature_dim=512, num_classes=6):
    """
    Recreate the original model architecture from train_branched_no_leakage.py
    but with compatibility for newer TensorFlow versions
    
    Args:
        audio_feature_dim: Dimensionality of audio features (default: 88 for eGeMAPSv02)
        video_feature_dim: Dimensionality of video features (default: 512 for FaceNet)
        num_classes: Number of emotion classes (default: 6)
        
    Returns:
        Compiled Keras model
    """
    print(f"Creating model architecture matching train_branched_no_leakage.py")
    print(f"- Audio feature dimension: {audio_feature_dim}")
    print(f"- Video feature dimension: {video_feature_dim}")
    
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
    # Note: omitting time_major parameter for compatibility
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
    # Note: omitting time_major parameter for compatibility
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

def load_weights_from_h5(model, h5_path):
    """
    Load weights from h5 file into the compatible model
    
    Args:
        model: The compatible model
        h5_path: Path to the h5 file containing weights
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Loading weights directly from {h5_path}")
        model.load_weights(h5_path, by_name=True)
        print("Weights loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading weights directly: {str(e)}")
        try:
            # Try alternative loading method
            print("Trying alternative loading method...")
            
            # This will load the entire model and then copy weights
            loaded_model = tf.keras.models.load_model(h5_path, compile=False)
            
            # Print layers to help with debugging
            print("\nOriginal model layers:")
            for i, layer in enumerate(loaded_model.layers):
                print(f"Layer {i}: {layer.name}, type: {type(layer).__name__}, shape: {layer.output_shape}")
            
            print("\nCompatible model layers:")
            for i, layer in enumerate(model.layers):
                print(f"Layer {i}: {layer.name}, type: {type(layer).__name__}, shape: {layer.output_shape}")
            
            # Try to copy weights layer by layer where names match
            copy_count = 0
            for layer in model.layers:
                if not layer.weights:
                    continue  # Skip layers without weights
                    
                # Try to find matching layer in loaded model
                for orig_layer in loaded_model.layers:
                    if orig_layer.name == layer.name and orig_layer.weights:
                        # Copy weights if shapes match
                        for i, w in enumerate(layer.weights):
                            if i < len(orig_layer.weights) and w.shape == orig_layer.weights[i].shape:
                                layer.set_weights(orig_layer.get_weights())
                                copy_count += 1
                                print(f"Copied weights for layer {layer.name}")
                                break
            
            print(f"Copied weights for {copy_count} layers using alternative method")
            return copy_count > 0
            
        except Exception as e2:
            print(f"Error with alternative loading method: {str(e2)}")
            return False

def test_model(model):
    """
    Test the loaded model with dummy inputs to verify it works
    
    Args:
        model: The model to test
    """
    if model is None:
        return
    
    # Get input shapes
    input_shapes = [(input.name, input.shape) for input in model.inputs]
    print("\nModel input shapes:")
    for name, shape in input_shapes:
        print(f"- {name}: {shape}")
    
    # Create dummy inputs
    dummy_inputs = {}
    for name, shape in input_shapes:
        # Create a batch of 1 with dynamic sequence length of 10
        # Replace None with a concrete value (10) for testing
        concrete_shape = [1] + [10 if dim is None else dim for dim in shape[1:]]
        dummy_input = np.random.random(concrete_shape)
        dummy_inputs[name] = dummy_input
    
    # Run prediction
    try:
        print("\nTesting model with dummy inputs...")
        predictions = model.predict(dummy_inputs)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction: {predictions}")
        print("Model test successful!")
        
        # Show emotion class probabilities
        emotions = ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
        print("\nEmotion probabilities:")
        for i, emotion in enumerate(emotions):
            print(f"{emotion}: {predictions[0][i]:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def main():
    """
    Main function to load and test the model
    """
    print("Starting compatible model loader...")
    
    # Model paths to try
    model_paths = [
        os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5'),
        os.path.join('models', 'dynamic_padding_no_leakage', 'final_model.h5'),
        os.path.join('models', 'branched_no_leakage', 'model_best.h5')
    ]
    
    # Find available model
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Error: No model file found in the expected locations.")
        return False
    
    print(f"Using model file: {model_path}")
    
    # Create compatible model architecture
    model = create_model_architecture()
    
    # Print model summary
    model.summary()
    
    # Load weights
    success = load_weights_from_h5(model, model_path)
    
    if success:
        # Test model
        test_success = test_model(model)
        if test_success:
            print("\nModel loaded and tested successfully!")
            return True
    
    print("\nFailed to load or test model properly.")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nCompatible model is ready to use!")
    else:
        print("\nEnsure you have the model files in the expected locations.")
