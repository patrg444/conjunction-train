import numpy as np
import os
import glob

# Directory containing the fixed CNN features
feature_dir = "/home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed"
num_files_to_check = 5

print(f"Checking shapes of first {num_files_to_check} .npy files in {feature_dir}")

npy_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))

if not npy_files:
    print("Error: No .npy files found in the directory.")
    exit(1)

checked_count = 0
for file_path in npy_files:
    if checked_count >= num_files_to_check:
        break
    try:
        features = np.load(file_path)
        print(f"  - {os.path.basename(file_path)}: Shape = {features.shape}")
        # Basic check for expected dimensions (T, 2048)
        expected_feature_dim = 2048 # Based on Reshape((-1, 16 * 128))
        if len(features.shape) != 2 or features.shape[1] != expected_feature_dim:
             print(f"    WARNING: Unexpected shape! Expected (T, {expected_feature_dim}), got {features.shape}")
        checked_count += 1
    except Exception as e:
        print(f"  - Error loading {os.path.basename(file_path)}: {e}")

print(f"\nChecked {checked_count} files.")
if checked_count < num_files_to_check:
     print(f"Warning: Found fewer than {num_files_to_check} files to check.")

print("Shape verification complete.")
