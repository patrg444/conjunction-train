#!/usr/bin/env python
"""
Script to analyze the contents of NPZ feature files to check if there's a difference
in audio feature dimensionality
"""

import os
import sys
import numpy as np

def analyze_npz_file(filepath):
    """Analyze a single NPZ file and print its contents"""
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"\nAnalyzing NPZ file: {filepath}")
        print("=" * 60)
        
        # List all arrays in the file
        print("Arrays in NPZ file:")
        for array_name in data.files:
            array = data[array_name]
            print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
            
            # For audio features, print some additional analysis
            if array_name == 'audio_features':
                feature_dim = array.shape[-1]  # Last dimension is feature dimension
                print(f"\nAudio feature details:")
                print(f"  Feature dimension: {feature_dim}")
                
                # Look for statistics about the features
                nonzero_features = np.count_nonzero(array, axis=0)
                print(f"  Non-zero values per feature (first 5):")
                for i in range(min(5, feature_dim)):
                    print(f"    Feature {i}: {nonzero_features[i]}")
                    
                # Show some sample frames
                print("\nSample audio feature frames (first 3 frames, first 10 features):")
                num_frames = min(3, array.shape[0])
                num_feats = min(10, feature_dim)
                for i in range(num_frames):
                    print(f"  Frame {i}:", end=" ")
                    for j in range(num_feats):
                        print(f"{array[i, j]:.4f}", end=" ")
                    print("...")
                    
                # Check if the first feature is always zero (might be a placeholder)
                if feature_dim > 0 and np.all(array[:, 0] == 0):
                    print("\nNote: First feature (index 0) is consistently zero - might be a placeholder or padding")
                
        # Check for any separate labels
        if 'labels' in data.files:
            print("\nLabels found:")
            print(f"  Shape: {data['labels'].shape}")
            print(f"  Unique values: {np.unique(data['labels'])}")
        
        return data
        
    except Exception as e:
        print(f"Error analyzing file {filepath}: {str(e)}")
        return None

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        analyze_npz_file(filepath)
    else:
        # Use a default example file
        filepath = "ravdess_features_facenet/Actor_01/01-01-01-01-01-01-01.npz"
        analyze_npz_file(filepath)
        
        # Also look at a file from ravdess_features (non-facenet) if exists
        audio_filepath = "ravdess_features/Actor_01/01-01-01-01-01-01-01.npz"
        if os.path.exists(audio_filepath):
            analyze_npz_file(audio_filepath)

if __name__ == "__main__":
    main()
