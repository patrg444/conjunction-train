#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training script to verify the environment is working.
"""

print("Training environment test - UTF-8 characters working: ©®µ§")
print("TRAINING STARTED SUCCESSFULLY WITH PROPER ENCODING")
print("This is a minimal test script to confirm the environment works.")

# Import the main libraries to check they're available
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version}")

# Create a very simple model just to test
inputs = Input(shape=(10,))
x = Dense(5, activation='relu')(inputs)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compilation successful!")
print("Environment test successful. You can now upload and run your full training script.")
print("The issue was with the encoding declaration in the Python file.")
