#!/usr/bin/env python3
"""
Debug script to understand the TensorFlow learning rate issue.
"""

import tensorflow as tf

class SimpleCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_begin(self, epoch, logs=None):
        new_lr = 0.001
        print(f"Learning rate type: {type(self.model.optimizer.learning_rate)}")
        # Method 1: Using set_value
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        print(f"After set_value: {self.model.optimizer.learning_rate}")
        
        # Method 2: Direct assignment (for testing)
        self.model.optimizer.learning_rate = new_lr
        print(f"After direct assignment: {self.model.optimizer.learning_rate}")

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Generate dummy data
import numpy as np
X = np.random.random((100, 5))
y = np.random.randint(0, 2, (100,))

# Train model with our debug callback
model.fit(
    X, y,
    epochs=2,
    batch_size=32,
    callbacks=[SimpleCallback()]
)
