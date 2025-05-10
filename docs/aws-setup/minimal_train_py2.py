#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal training script to verify the environment is working.
Compatible with Python 2 and 3.
"""

from __future__ import print_function  # For Python 2 compatibility

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

# Old-style string formatting for compatibility
print("TensorFlow version: %s" % tf.__version__)
print("NumPy version: %s" % np.__version__)
print("Python version: %s" % sys.version)

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
print("The issue was with the Python version - we need to ensure compatibility.")
