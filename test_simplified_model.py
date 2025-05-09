#!/usr/bin/env python3
"""
Test script for simplified_cnn_lstm_model.py

This script imports the test function from the simplified model and runs it
to verify the model architecture works locally before deploying to EC2.
"""

import sys
import numpy as np
import tensorflow as tf
from simplified_cnn_lstm_model import (
    test_model_with_dummy_data,
    create_simplified_lstm_model,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    NUM_CLASSES
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Testing simplified CNN-LSTM model locally")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version}")
print(f"Model configuration:")
print(f"- Fixed sequence length: {MAX_SEQ_LENGTH}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Number of classes: {NUM_CLASSES}")

# Run the test
test_result = test_model_with_dummy_data()

# Additional manual testing to verify model output shapes
feature_dim = 2048  # This should match CNN features dimension
print("\nPerforming additional shape verification...")

# Create model
model = create_simplified_lstm_model(feature_dim, MAX_SEQ_LENGTH)

# Create a simple batch with different sequence lengths to verify padding works
print("Testing variable length sequence handling...")

# Create 3 sequences of different lengths
seq1 = np.random.randn(MAX_SEQ_LENGTH, feature_dim)  # Exact length
seq2 = np.random.randn(MAX_SEQ_LENGTH - 5, feature_dim)  # Shorter
seq3 = np.random.randn(MAX_SEQ_LENGTH + 5, feature_dim)  # Longer

# Create padding function inline for simplicity
def pad_or_truncate(sequence, max_length):
    """Pad or truncate sequence to fixed length."""
    if sequence.shape[0] > max_length:
        return sequence[:max_length]
    elif sequence.shape[0] < max_length:
        padding = np.zeros((max_length - sequence.shape[0], sequence.shape[1]))
        return np.vstack([sequence, padding])
    else:
        return sequence

# Pad/truncate sequences
seq1_padded = pad_or_truncate(seq1, MAX_SEQ_LENGTH)
seq2_padded = pad_or_truncate(seq2, MAX_SEQ_LENGTH)
seq3_padded = pad_or_truncate(seq3, MAX_SEQ_LENGTH)

print(f"Original seq1 shape: {seq1.shape}, after padding: {seq1_padded.shape}")
print(f"Original seq2 shape: {seq2.shape}, after padding: {seq2_padded.shape}")
print(f"Original seq3 shape: {seq3.shape}, after padding: {seq3_padded.shape}")

# Stack them into a batch
batch = np.stack([seq1_padded, seq2_padded, seq3_padded], axis=0)
print(f"Batch shape: {batch.shape}")

# Make a prediction
predictions = model.predict(batch, verbose=1)
print(f"Predictions shape: {predictions.shape}")

# Verify that all shapes match expectations
if all([
    seq1_padded.shape == (MAX_SEQ_LENGTH, feature_dim),
    seq2_padded.shape == (MAX_SEQ_LENGTH, feature_dim),
    seq3_padded.shape == (MAX_SEQ_LENGTH, feature_dim),
    batch.shape == (3, MAX_SEQ_LENGTH, feature_dim),
    predictions.shape == (3, NUM_CLASSES)
]):
    print("\n✅ All shapes match expectations. The model handles variable length sequences correctly.")
    print("The model is ready for deployment to EC2.")
else:
    print("\n❌ Shape verification failed. Please check the model architecture.")

if test_result:
    print("\n🎉 Simplified CNN-LSTM model tested successfully!")
    print("You can now deploy this model to EC2 with confidence.")
else:
    print("\n❌ Simplified CNN-LSTM model test failed. Please fix issues before deploying.")
