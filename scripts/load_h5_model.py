#!/usr/bin/env python
"""
Load H5 Model Directly

This script loads the pre-trained h5 model from models/dynamic_padding_no_leakage/model_best.h5
using a compatible TensorFlow version that supports the time_major parameter.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Display TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

def load_model():
    """
    Load the pre-trained model directly from h5 file
    """
    model_path = os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model directly
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model summary
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def test_model(model):
    """
    Test the loaded model with dummy inputs
    """
    if model is None:
        return
    
    # Get input shapes from model
    input_shapes = {input.name: input.shape for input in model.inputs}
    print(f"Model input shapes: {input_shapes}")
    
    # Create some dummy input data
    dummy_inputs = {}
    for input_name, shape in input_shapes.items():
        # Create a batch of 1 with dynamic sequence length of 10
        # Replace None with a concrete value (10) for testing
        concrete_shape = [1] + [10 if dim is None else dim for dim in shape[1:]]
        dummy_inputs[input_name] = np.random.random(concrete_shape)
    
    # Run prediction
    try:
        print("Testing model with dummy inputs...")
        predictions = model.predict(dummy_inputs)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction: {predictions}")
        print("Model test successful!")
    except Exception as e:
        print(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    print("Starting model loading script...")
    model = load_model()
    if model is not None:
        test_model(model)
    else:
        print("Model loading failed, skipping test.")
