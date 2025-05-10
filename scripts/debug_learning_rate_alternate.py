#!/usr/bin/env python3
"""
Debug script with alternative methods to update learning rate.
"""

import tensorflow as tf
import numpy as np

class AlternativeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_begin(self, epoch, logs=None):
        # Print detailed information about the learning rate variable
        print(f"LR type: {type(self.model.optimizer.learning_rate)}")
        print(f"LR value: {self.model.optimizer.learning_rate}")
        print(f"LR dir: {dir(self.model.optimizer.learning_rate)}")
        
        new_lr = 0.001
        
        # Attempt 1: Direct assignment with explicit conversion to Variable
        try:
            print("Attempt 1: Direct assignment with Variable")
            self.model.optimizer.learning_rate = tf.Variable(new_lr)
            print(f"After attempt 1: {self.model.optimizer.learning_rate}")
        except Exception as e:
            print(f"Attempt 1 failed: {e}")
        
        # Attempt 2: Using assign method if available
        try:
            print("Attempt 2: Using assign method")
            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                self.model.optimizer.learning_rate.assign(new_lr)
                print(f"After attempt 2: {self.model.optimizer.learning_rate}")
            else:
                print("No assign method found")
        except Exception as e:
            print(f"Attempt 2 failed: {e}")
        
        # Attempt 3: Using K.set_value with explicit comma
        try:
            print("Attempt 3: Using K.set_value with explicit comma")
            print(f"Parameters: {self.model.optimizer.learning_rate}, {new_lr}")
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"After attempt 3: {self.model.optimizer.learning_rate}")
        except Exception as e:
            print(f"Attempt 3 failed: {e}")
            print(f"Exception type: {type(e)}")

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Use learning rate as a string to replicate the issue
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Generate dummy data
X = np.random.random((100, 5))
y = np.random.randint(0, 2, (100,))

# Train model with our debug callback
try:
    model.fit(
        X, y,
        epochs=1,
        batch_size=32,
        callbacks=[AlternativeCallback()],
        verbose=0
    )
except Exception as e:
    print(f"Main exception: {e}")
    import traceback
    traceback.print_exc()
