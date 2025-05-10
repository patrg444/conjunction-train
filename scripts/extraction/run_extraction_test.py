#!/usr/bin/env python3
"""
Test script to extract features for just one sample file and verify the process works.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
import glob

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Add the scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import needed components
try:
    from scripts.spectrogram_cnn_pooling_generator import build_cnn_feature_extractor
    from scripts.preprocess_spectrograms import N_MELS
    print(f"Successfully imported dependencies. N_MELS = {N_MELS}")
except ImportError as e:
    print(f"Error importing required components: {e}")
    sys.exit(1)

def process_sample_files(input_dirs, output_dirs, num_samples=3):
    """Process a few sample files from each dataset as a test"""
    
    # Build the CNN model once
    cnn_input_shape = (None, N_MELS, 1)  # (time, mels, channels)
    print("Building CNN model...")
    cnn_model = build_cnn_feature_extractor(input_shape=cnn_input_shape)
    print(f"CNN model built. Input shape: {cnn_input_shape}, Output dimension: {cnn_model.output_shape[-1]}")
    
    total_processed = 0
    
    # Process samples from each dataset
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        print(f"\n=== Processing samples from: {input_dir} ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find spectrogram files
        spec_files = glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)
        if not spec_files:
            print(f"No spectrogram files found in {input_dir}")
            continue
            
        # Take a few sample files
        sample_files = spec_files[:num_samples]
        print(f"Selected {len(sample_files)} sample files for testing")
        
        for file_path in sample_files:
            file_name = os.path.basename(file_path)
            print(f"\nProcessing: {file_name}")
            
            # Load the spectrogram
            try:
                start_time = time.time()
                spectrogram = np.load(file_path)
                print(f"Loaded spectrogram with shape: {spectrogram.shape}")
                
                # Apply shape handling
                if len(spectrogram.shape) == 2:
                    if spectrogram.shape[0] == N_MELS:
                        print(f"Transposing from (n_mels={N_MELS}, time) to (time, n_mels)")
                        spectrogram = spectrogram.T
                        print(f"After transpose: {spectrogram.shape}")
                    spectrogram = np.expand_dims(spectrogram, axis=-1)
                    print(f"After adding channel dimension: {spectrogram.shape}")
                
                # Add batch dimension
                spectrogram = np.expand_dims(spectrogram, axis=0)
                print(f"After adding batch dimension: {spectrogram.shape}")
                
                # Extract CNN features
                print("Extracting CNN features...")
                features = cnn_model.predict(spectrogram, verbose=1)
                print(f"Extracted features with shape: {features.shape}")
                
                # Save the result
                output_path = os.path.join(output_dir, file_name)
                np.save(output_path, features[0])
                print(f"Saved to: {output_path}")
                
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nCompleted test extraction. Successfully processed {total_processed} files.")
    return total_processed

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output directories
    ravdess_spec_in = os.path.join(project_root, "data", "ravdess_features_spectrogram")
    cremad_spec_in = os.path.join(project_root, "data", "crema_d_features_spectrogram")
    ravdess_cnn_out = os.path.join(project_root, "data", "ravdess_features_cnn_audio")
    cremad_cnn_out = os.path.join(project_root, "data", "crema_d_features_cnn_audio")
    
    # Process a few sample files from each dataset
    process_sample_files(
        [ravdess_spec_in, cremad_spec_in],
        [ravdess_cnn_out, cremad_cnn_out],
        num_samples=2  # Process 2 samples from each dataset
    )
