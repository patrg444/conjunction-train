#!/usr/bin/env python3
"""
Script to directly fix Python environment and install required packages
"""
import sys
import os
import subprocess
import time

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0

def check_import(module):
    """Check if a module can be imported"""
    try:
        exec(f"import {module}")
        print(f"✓ Successfully imported {module}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module}: {e}")
        return False

# Print Python info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Ensure pip is up to date
run_command(f"{sys.executable} -m pip install --upgrade pip")

# List of required packages
required_packages = [
    "numpy",
    "tensorflow",
    "pandas", 
    "scikit-learn",
    "matplotlib",
    "h5py"
]

# Install required packages using the current Python
for package in required_packages:
    print(f"\nInstalling {package}...")
    run_command(f"{sys.executable} -m pip install {package}")
    
    # Verify installation
    check_import(package)

# Create a simple test script to verify TensorFlow
test_script = """
import numpy as np
import tensorflow as tf

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Create a small test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
print("Test model compiled successfully")

# Create some dummy data
x = np.random.random((10, 5))
y = np.random.randint(0, 2, (10, 1))

# Train for just one step
model.fit(x, y, epochs=1)
print("Test complete!")
"""

# Write test script
with open("test_tensorflow.py", "w") as f:
    f.write(test_script)

# Run the test script
print("\nRunning TensorFlow test...")
success = run_command(f"{sys.executable} test_tensorflow.py")

if success:
    print("\n✓ TensorFlow environment is correctly set up!")
else:
    print("\n✗ TensorFlow environment setup failed")

# Create a wrapper script for running the LSTM attention model
wrapper_script = """#!/bin/bash
cd ~/emotion_training
export PYTHONPATH=$PYTHONPATH:~/emotion_training
echo "Starting LSTM attention model training at $(date)"
python3 scripts/train_branched_attention.py 2>&1 | tee training_lstm_attention_model.log
echo "Training completed with exit code $? at $(date)"
"""

# Write wrapper script
with open("run_lstm_attention.sh", "w") as f:
    f.write(wrapper_script)

# Make it executable
run_command("chmod +x run_lstm_attention.sh")

print("\nEnvironment fix completed. Ready to restart LSTM attention model training.")
