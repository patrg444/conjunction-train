#!/usr/bin/env python3
"""
Simple test script to verify the correct syntax for set_value in TensorFlow.
"""
import tensorflow as tf
import numpy as np

class TestModel:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

def test_set_value():
    """Test that the set_value syntax is correct."""
    model = TestModel()
    
    # Test both scenarios that were fixed
    
    # 1. Warmup case
    warmup_lr = 0.001
    try:
        tf.keras.backend.set_value(model.optimizer.learning_rate, warmup_lr)
        print("✓ First test passed: set_value with comma works correctly")
    except Exception as e:
        print(f"✗ First test failed: {e}")
    
    # 2. New LR case
    old_lr = float(model.optimizer.learning_rate)
    new_lr = old_lr * 0.5
    try:
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        print("✓ Second test passed: set_value with comma works correctly")
    except Exception as e:
        print(f"✗ Second test failed: {e}")
    
    # 3. Test the original buggy syntax to confirm it fails
    try:
        # This should fail - intentionally missing comma
        # tf.keras.backend.set_value(model.optimizer.learning_rate warmup_lr)
        print("The buggy syntax would be tested here, but it's commented out to avoid syntax errors")
    except Exception as e:
        print(f"Expected error from buggy syntax would be: {e}")
        
    print("Syntax tests completed")

if __name__ == "__main__":
    test_set_value()
