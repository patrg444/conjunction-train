#!/usr/bin/env python3
"""
Debug script that directly executes the training function from train_video_only_facenet_lstm_key_fixed.py
"""
import sys
import os

# Insert the directory containing the script at the beginning of sys.path
# This ensures that imports from the script work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the specific function we want to debug from the training script
from scripts.train_video_only_facenet_lstm_key_fixed import load_data_paths_and_labels_video_only

def main():
    """Debug main function to trace why the training script can't find the facenet files"""
    print("=== FACENET LOADING DEBUG SCRIPT ===")
    
    # Get the exact same paths as in the training script
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    
    print(f"\nRAVDESS directory exists: {os.path.exists(RAVDESS_FACENET_DIR)}")
    print(f"CREMA-D directory exists: {os.path.exists(CREMA_D_FACENET_DIR)}")
    
    # Test direct glob pattern
    import glob
    ravdess_pattern = os.path.join(RAVDESS_FACENET_DIR, "Actor_*", "*.npz") 
    cremad_pattern = os.path.join(CREMA_D_FACENET_DIR, "*.npz")
    
    ravdess_files = glob.glob(ravdess_pattern)
    cremad_files = glob.glob(cremad_pattern)
    
    print(f"\nDirect glob found:")
    print(f"- RAVDESS files: {len(ravdess_files)}")
    print(f"- CREMA-D files: {len(cremad_files)}")
    
    # Now call the function from the training script
    print("\nCalling load_data_paths_and_labels_video_only function from training script...")
    try:
        facenet_files, all_labels = load_data_paths_and_labels_video_only(
            RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
        )
        print(f"Successfully loaded {len(facenet_files)} Facenet files with labels")
        print(f"Labels shape: {all_labels.shape}")
        
        # Count by dataset
        ravdess_count = sum(1 for f in facenet_files if "Actor_" in f)
        cremad_count = len(facenet_files) - ravdess_count
        print(f"RAVDESS files: {ravdess_count}")
        print(f"CREMA-D files: {cremad_count}")
        
    except Exception as e:
        print(f"Error when calling function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
