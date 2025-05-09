#!/usr/bin/env python3
"""
Prepare wav2vec embedding data for training by:
1. Creating the expected directory structure
2. Converting .npz files from models/wav2vec/ to .npy files in data/ravdess_features_wav2vec2/ and data/crema_d_features_wav2vec2/
"""

import os
import glob
import numpy as np
from tqdm import tqdm

def ensure_dir(directory):
    """Make sure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main():
    # Define paths
    source_dir = "/home/ubuntu/audio_emotion/models/wav2vec"
    ravdess_target_dir = "/home/ubuntu/audio_emotion/data/ravdess_features_wav2vec2"
    cremad_target_dir = "/home/ubuntu/audio_emotion/data/crema_d_features_wav2vec2"
    
    # Create necessary directories
    ensure_dir("/home/ubuntu/audio_emotion/data")
    ensure_dir(ravdess_target_dir)
    ensure_dir(cremad_target_dir)
    
    # Process all .npz files in source directory
    npz_files = glob.glob(os.path.join(source_dir, "*.npz"))
    print(f"Found {len(npz_files)} .npz files in {source_dir}")
    
    ravdess_count = 0
    cremad_count = 0
    error_count = 0
    
    for npz_file in tqdm(npz_files, desc="Converting files"):
        filename = os.path.basename(npz_file)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # Load wav2vec features from .npz file
            with np.load(npz_file) as data:
                if 'wav2vec_features' in data:
                    features = data['wav2vec_features']
                else:
                    print(f"Warning: 'wav2vec_features' key not found in {npz_file}")
                    error_count += 1
                    continue
                
            # Determine target directory based on filename prefix
            if filename.startswith("ravdess_"):
                # For RAVDESS files, we need to create actor subdirectories
                actor_id = None
                
                # Extract actor ID from filename (e.g., ravdess_03-01-06-01-02-01-12.npz → Actor_12)
                parts = base_name.replace("ravdess_", "").split('-')
                if len(parts) >= 7:
                    actor_id = parts[-1]  # Last part is actor ID
                
                if actor_id:
                    actor_dir = os.path.join(ravdess_target_dir, f"Actor_{actor_id}")
                    ensure_dir(actor_dir)
                    
                    # Save as .npy file in the actor directory with original filename minus "ravdess_" prefix
                    output_file = os.path.join(actor_dir, base_name.replace("ravdess_", "") + ".npy")
                    np.save(output_file, features)
                    ravdess_count += 1
                else:
                    print(f"Warning: Could not determine actor ID for {filename}")
                    error_count += 1
                    
            elif filename.startswith("cremad_"):
                # For CREMA-D, save directly in the cremad target directory
                output_file = os.path.join(cremad_target_dir, base_name.replace("cremad_", "") + ".npy")
                np.save(output_file, features)
                cremad_count += 1
                
            else:
                print(f"Warning: Unknown file format: {filename}")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
    
    print("\nData preparation complete:")
    print(f"- Processed {len(npz_files)} files")
    print(f"- Created {ravdess_count} RAVDESS .npy files")
    print(f"- Created {cremad_count} CREMA-D .npy files")
    print(f"- Encountered {error_count} errors")
    
    # Verification step - count files in created directories
    ravdess_files = sum([len(files) for r, d, files in os.walk(ravdess_target_dir)])
    cremad_files = len(glob.glob(os.path.join(cremad_target_dir, "*.npy")))
    
    print("\nVerification:")
    print(f"- {ravdess_files} files in RAVDESS directory")
    print(f"- {cremad_files} files in CREMA-D directory")
    print(f"- Total: {ravdess_files + cremad_files} .npy files created")

if __name__ == "__main__":
    main()
