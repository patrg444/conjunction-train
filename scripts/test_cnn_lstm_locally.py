#!/usr/bin/env python3
"""
Testing script for final_cnn_lstm_model.py

This script imports the key components from the final CNN-LSTM model and tests:
1. Creating a synthetic dummy batch with the correct fixed shape
2. Building and compiling the model
3. Verifying a single forward and backward pass works

If this script runs without errors, it confirms the model architecture 
is sound and ready for deployment to EC2.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Import key components from the final model
from final_cnn_lstm_model import (
    MAX_SEQ_LENGTH,
    NUM_CLASSES,
    BATCH_SIZE,
    create_lstm_attention_model
)

print("Testing CNN-LSTM model with fixed sequence length")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version}")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def test_model():
    """Run a basic test of the model architecture."""
    # Define a sample feature dimension (matches CNN features dimension)
    feature_dim = 2048  # This value should match your actual CNN features
    
    print(f"\nCreating test model with parameters:")
    print(f"- Feature dimension: {feature_dim}")
    print(f"- Fixed sequence length: {MAX_SEQ_LENGTH}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Number of classes: {NUM_CLASSES}")
    
    # Create the model
    model = create_lstm_attention_model(feature_dim, MAX_SEQ_LENGTH)
    
    # Print model summary
    model.summary()
    
    # Create a synthetic batch of data with the correct shape
    print("\nGenerating synthetic test batch...")
    dummy_x = np.random.randn(BATCH_SIZE, MAX_SEQ_LENGTH, feature_dim)
    dummy_y = tf.keras.utils.to_categorical(
        np.random.randint(NUM_CLASSES, size=BATCH_SIZE), 
        NUM_CLASSES
    )
    
    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {dummy_y.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    predictions = model.predict(dummy_x, verbose=1)
    print(f"Predictions shape: {predictions.shape}")
    
    # Test backward pass (single training step)
    print("\nTesting backward pass (single train step)...")
    history = model.fit(
        dummy_x, 
        dummy_y,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=1
    )
    
    loss = history.history['loss'][0]
    print(f"Training loss: {loss:.4f}")
    
    print("\n✅ Test completed successfully! The model architecture is valid.")
    print("The model can now be safely deployed to EC2.")

if __name__ == "__main__":
    test_model()
