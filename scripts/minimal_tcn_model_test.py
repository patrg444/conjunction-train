#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for TCN model compatibility
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Concatenate
from tensorflow.keras.layers import Conv1D, BatchNormalization, Masking
from tensorflow.keras.layers import Activation, Add, LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# Simplified TCN block without as many parameters
def simple_tcn_block(x, filters, kernel_size=3, dilation_rate=1):
    """
    Creates a simplified TCN block
    """
    # Save input for skip connection
    input_tensor = x
    
    # First dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu'
    )(x)
    conv1 = BatchNormalization()(conv1)
    
    # Residual connection
    if input_tensor.shape[-1] != filters:
        # If dimensions don't match, use 1x1 conv to adapt dimensions
        input_tensor = Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same'
        )(input_tensor)
    
    # Add residual connection
    result = Add()([conv1, input_tensor])
    return Activation('relu')(result)

def create_simple_test_model(audio_dim=40, video_dim=512):
    """
    Create a simplified model for testing
    """
    print("Creating simplified test model...")
    
    # Audio branch
    audio_input = Input(shape=(None, audio_dim), name='audio_input')
    audio_x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(audio_input)
    audio_x = GlobalAveragePooling1D()(audio_x)
    audio_x = Dense(64, activation='relu')(audio_x)
    
    # Video branch with simplified TCN
    video_input = Input(shape=(None, video_dim), name='video_input')
    video_x = Conv1D(32, kernel_size=1, padding='same')(video_input)
    
    # Apply simplified TCN blocks with moderate dilation rates
    video_x = simple_tcn_block(video_x, filters=32, dilation_rate=1)
    video_x = simple_tcn_block(video_x, filters=32, dilation_rate=2)
    
    video_x = GlobalAveragePooling1D()(video_x)
    video_x = Dense(64, activation='relu')(video_x)
    
    # Merge branches
    merged = Concatenate()([audio_x, video_x])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    
    # Output layer
    output = Dense(6, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)
    
    # Use standard Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_batch():
    """
    Create a small dummy batch for testing
    """
    # Create dummy data
    batch_size = 2
    seq_len = 10
    audio_dim = 40
    video_dim = 512
    
    # Random data
    audio_data = np.random.random((batch_size, seq_len, audio_dim))
    video_data = np.random.random((batch_size, seq_len, video_dim))
    labels = np.random.random((batch_size, 6))
    
    # Normalize labels to create a valid probability distribution
    labels = labels / np.sum(labels, axis=1, keepdims=True)
    
    return {'audio_input': audio_data, 'video_input': video_data}, labels

def test_inference():
    """
    Test model creation and a single inference step
    """
    print("Creating model...")
    model = create_simple_test_model()
    
    print("Model created successfully!")
    model.summary()
    
    print("\nTesting inference with dummy batch...")
    inputs, labels = create_dummy_batch()
    
    # Single prediction step
    predictions = model.predict(inputs)
    print("Predictions shape:", predictions.shape)
    
    print("\nTesting a single training step...")
    # Single training step
    history = model.fit(inputs, labels, epochs=1, verbose=1)
    
    print("\nAll tests completed successfully!")
    return model

if __name__ == '__main__':
    try:
        print("Starting minimal TCN model compatibility test...")
        test_inference()
        print("Test completed successfully!")
    except Exception as e:
        import traceback
        print('ERROR:', str(e))
        print(traceback.format_exc())
