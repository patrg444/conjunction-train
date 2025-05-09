import numpy as np
import sys
import os

def verify_shapes(file_paths):
    """Loads .npy files and prints their shapes."""
    print(f"Checking shapes for {len(file_paths)} files...")
    all_correct = True
    expected_dim = 2048  # Expected feature dimension from the CNN

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            all_correct = False
            continue
        try:
            features = np.load(file_path)
            shape = features.shape
            print(f"File: {file_path}, Shape: {shape}")
            if len(shape) != 2 or shape[1] != expected_dim:
                print(f"  WARNING: Unexpected shape! Expected (time_steps, {expected_dim})")
                all_correct = False
            elif shape[0] == 0:
                 print(f"  WARNING: Feature file has 0 time steps!")
                 all_correct = False
        except Exception as e:
            print(f"ERROR loading or checking file {file_path}: {e}")
            all_correct = False

    if all_correct:
        print("\nAll checked files have the expected shape format (time_steps, 2048).")
    else:
        print("\nWARNING: Some files had unexpected shapes or errors.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_feature_shapes.py <file1.npy> [file2.npy] ...")
        sys.exit(1)
    
    files_to_check = sys.argv[1:]
    verify_shapes(files_to_check)
