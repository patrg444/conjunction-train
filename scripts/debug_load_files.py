#!/usr/bin/env python3
import os
import glob
import sys
import numpy as np

def debug_load_paths(ravdess_dir, cremad_dir):
    """Debug function to mimic exactly what the training script does"""
    print(f"RAVDESS dir: {ravdess_dir}")
    print(f"CREMA-D dir: {cremad_dir}")
    
    # Check if directories exist
    print(f"RAVDESS directory exists: {os.path.exists(ravdess_dir)}")
    print(f"CREMA-D directory exists: {os.path.exists(cremad_dir)}")
    
    # Try exact glob patterns from training script
    ravdess_pattern = os.path.join(ravdess_dir, "Actor_*", "*.npz")
    cremad_pattern = os.path.join(cremad_dir, "*.npz")
    
    print(f"RAVDESS glob pattern: {ravdess_pattern}")
    print(f"CREMA-D glob pattern: {cremad_pattern}")
    
    ravdess_files = glob.glob(ravdess_pattern)
    cremad_files = glob.glob(cremad_pattern)
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(cremad_files)} CREMA-D files")
    
    # Sample first few files to verify
    if ravdess_files:
        print("\nSample RAVDESS files:")
        for f in ravdess_files[:3]:
            print(f"  {f}")
    
    if cremad_files:
        print("\nSample CREMA-D files:")
        for f in cremad_files[:3]:
            print(f"  {f}")
    
    # Try to trace the problem by listing contents of the directories
    print("\nDirectory structure verification:")
    
    # Check RAVDESS directory contents
    if os.path.exists(ravdess_dir):
        ravdess_contents = os.listdir(ravdess_dir)
        print(f"RAVDESS directory has {len(ravdess_contents)} entries")
        print("First 5 entries:")
        for entry in ravdess_contents[:5]:
            full_path = os.path.join(ravdess_dir, entry)
            if os.path.isdir(full_path):
                subdir_contents = os.listdir(full_path)
                print(f"  {entry}/ (directory with {len(subdir_contents)} files)")
                if subdir_contents:
                    print(f"    First file: {subdir_contents[0]}")
            else:
                print(f"  {entry} (file)")
    
    # Check CREMA-D directory contents
    if os.path.exists(cremad_dir):
        cremad_contents = os.listdir(cremad_dir)
        print(f"CREMA-D directory has {len(cremad_contents)} entries")
        print("First 5 entries:")
        for entry in cremad_contents[:5]:
            print(f"  {entry}")
    
    # Check for permissions issues
    print("\nPermissions check:")
    if os.path.exists(ravdess_dir):
        stat_info = os.stat(ravdess_dir)
        print(f"RAVDESS directory mode: {stat_info.st_mode}")
        print(f"RAVDESS directory owner: {stat_info.st_uid}")
    
    if os.path.exists(cremad_dir):
        stat_info = os.stat(cremad_dir)
        print(f"CREMA-D directory mode: {stat_info.st_mode}")
        print(f"CREMA-D directory owner: {stat_info.st_uid}")
    
    # Return total file count
    return len(ravdess_files) + len(cremad_files)

# Main execution
if __name__ == "__main__":
    # Using the exact same paths as in the training script
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    
    total = debug_load_paths(RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR)
    print(f"\nTotal files found: {total}")

    # Try alternative approach - using os.walk
    print("\nAlternative approach using os.walk:")
    ravdess_files_walk = []
    cremad_files_walk = []
    
    if os.path.exists(RAVDESS_FACENET_DIR):
        for root, dirs, files in os.walk(RAVDESS_FACENET_DIR):
            for file in files:
                if file.endswith(".npz"):
                    ravdess_files_walk.append(os.path.join(root, file))
    
    if os.path.exists(CREMA_D_FACENET_DIR):
        for root, dirs, files in os.walk(CREMA_D_FACENET_DIR):
            for file in files:
                if file.endswith(".npz"):
                    cremad_files_walk.append(os.path.join(root, file))
    
    print(f"Found {len(ravdess_files_walk)} RAVDESS files using os.walk")
    print(f"Found {len(cremad_files_walk)} CREMA-D files using os.walk")
    print(f"Total files found using os.walk: {len(ravdess_files_walk) + len(cremad_files_walk)}")
