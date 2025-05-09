#!/usr/bin/env python3
"""
Simple test script to verify the shape handling fix for CNN feature extraction
"""

import numpy as np
import os
import sys
import tensorflow as tf

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Try to import from the scripts directory
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    # Import needed components
    from scripts.spectrogram_cnn_pooling_generator import build_cnn_feature_extractor
    from scripts.preprocess_spectrograms import N_MELS
    print(f"Successfully imported dependencies. N_MELS = {N_MELS}")
except ImportError as e:
    print(f"Error importing required components: {e}")
    sys.exit(1)

def test_single_file(file_path):
    """Test CNN extraction on a single spectrogram file with detailed logging"""
    
    print(f"\n=== Testing file: {file_path} ===")
    
    # Build CNN model
    print("Building CNN model...")
    cnn_input_shape = (None, N_MELS, 1)  # (time, mels, channels)
    cnn_model = build_cnn_feature_extractor(input_shape=cnn_input_shape)
    print(f"CNN model built. Input shape: {cnn_input_shape}, Output dimension: {cnn_model.output_shape[-1]}")
    
    # Load the spectrogram
    print("Loading spectrogram...")
    try:
        spectrogram = np.load(file_path)
        print(f"Loaded spectrogram with shape: {spectrogram.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Apply our shape handling logic
    print("\nApplying shape handling...")
    print(f"Before: spectrogram.shape = {spectrogram.shape}")
    
    # Handle shape transformations
    if len(spectrogram.shape) == 2:
        # Check if spectrogram is in (n_mels, time) format and transpose to (time, n_mels)
        if spectrogram.shape[0] == N_MELS:
            print(f"Detected (n_mels, time) format with n_mels={spectrogram.shape[0]}, transposing...")
            spectrogram = spectrogram.T
            print(f"After transpose: {spectrogram.shape}")
        else:
            print(f"Shape appears to already be in (time, n_mels) format")
        
        # Add channel dimension
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        print(f"After adding channel dimension: {spectrogram.shape}")
    
    # Add batch dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)
    print(f"After adding batch dimension: {spectrogram.shape}")
    
    # Run prediction
    print("\nRunning prediction...")
    try:
        features = cnn_model.predict(spectrogram, verbose=1)
        print(f"Prediction succeeded! Output shape: {features.shape}")
        print("First few values:", features[0, :5])
        
        # Save the result to verify
        output_path = os.path.join(os.path.dirname(file_path), "test_cnn_output.npy")
        np.save(output_path, features[0])
        print(f"Saved test output to: {output_path}")
        
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Find a sample file to test
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find a spectrogram file
    search_dirs = [
        os.path.join(project_root, "data", "ravdess_features_spectrogram"),
        os.path.join(project_root, "data", "crema_d_features_spectrogram")
    ]
    
    test_file = None
    for directory in search_dirs:
        if os.path.exists(directory):
            files = os.listdir(directory)
            for file in files:
                if file.endswith(".npy"):
                    test_file = os.path.join(directory, file)
                    break
            if test_file:
                break
    
    if not test_file:
        print("Could not find any spectrogram files to test.")
        return
    
    print(f"Testing with file: {test_file}")
    success = test_single_file(test_file)
    
    if success:
        print("\n✅ Test PASSED! The shape handling fix is working correctly.")
    else:
        print("\n❌ Test FAILED! There may still be issues with the shape handling.")

if __name__ == "__main__":
    main()
