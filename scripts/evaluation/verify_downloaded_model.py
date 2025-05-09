#!/usr/bin/env python
# Script to verify the downloaded model for integrity and structure

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def check_file_exists(filepath):
    """Check if the file exists and return its size in MB."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False, 0
    
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"✅ File found: {filepath}")
    print(f"   Size: {size_mb:.2f} MB")
    return True, size_mb

def check_model_structure(model_path):
    """Load the model and print its structure."""
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully")
        
        # Print model summary to string
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = '\n'.join(model_summary)
        print("\nModel Summary:")
        print("-" * 80)
        print(model_summary)
        print("-" * 80)
        
        # Check output layer for number of classes
        output_layer = model.layers[-1]
        num_classes = output_layer.output_shape[-1]
        print(f"\nNumber of output classes: {num_classes}")
        
        return True, model_summary
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False, None

def main():
    # Define paths
    model_dir = "models/dynamic_padding_no_leakage"
    model_paths = [
        os.path.join(model_dir, "model_best.h5"),
        os.path.join(model_dir, "final_model.h5")
    ]
    info_path = os.path.join(model_dir, "model_info.json")
    
    print("\n" + "=" * 80)
    print("VERIFYING DOWNLOADED MODEL FROM EC2 INSTANCE")
    print("=" * 80)
    
    # Check model files
    for model_path in model_paths:
        exists, size = check_file_exists(model_path)
        if not exists:
            print(f"❌ Model file {model_path} is missing!")
            continue
    
    # Check info file
    info_exists, _ = check_file_exists(info_path)
    if info_exists:
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"✅ Model info loaded successfully")
            print(f"\nModel Information:")
            print(f"   Name: {info.get('model_name')}")
            print(f"   Validation Accuracy: {info.get('validation_accuracy')*100:.2f}%")
            print(f"   Training Epoch: {info.get('training_epoch')}")
            print(f"   Classes: {', '.join(info.get('classes', []))}")
        except Exception as e:
            print(f"❌ Error reading model info: {str(e)}")
    
    # Check model structure
    print("\nChecking model structure...")
    success, _ = check_model_structure(model_paths[0])  # Check the first model file
    
    print("\n" + "=" * 80)
    if success:
        print("✅ MODEL VERIFICATION COMPLETE - READY FOR USE")
    else:
        print("❌ MODEL VERIFICATION FAILED - PLEASE CHECK ERRORS")
    print("=" * 80)

if __name__ == "__main__":
    main()
