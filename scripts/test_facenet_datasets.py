#!/usr/bin/env python3
# Test script to verify handling of both CREMA-D and RAVDESS datasets
import os
import sys
import glob
import numpy as np
import tensorflow as tf

# Import our generator
try:
    # Try relative import
    from scripts.video_only_facenet_generator import VideoOnlyFacenetGenerator
except ImportError:
    # Fall back to absolute import
    sys.path.append('scripts')
    from video_only_facenet_generator import VideoOnlyFacenetGenerator

# Define paths for EC2 - these are the same as in the training script
RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet" 
CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"

# For local testing, we'll use relative paths
LOCAL_RAVDESS_FACENET_DIR = "./ravdess_features_facenet"
LOCAL_CREMA_D_FACENET_DIR = "./crema_d_features_facenet"

# Auto-detect if we're on EC2 or local
def get_dataset_paths():
    """Determine which paths to use based on environment"""
    if os.path.exists(RAVDESS_FACENET_DIR):
        print("Using EC2 paths")
        return RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
    else:
        print("Using local paths")
        return LOCAL_RAVDESS_FACENET_DIR, LOCAL_CREMA_D_FACENET_DIR

# Label extraction logic from training script
def extract_labels(file_paths):
    """Extract emotion labels from filenames"""
    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}
    
    valid_files = []
    skipped = 0
    
    for file_path in file_paths:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.endswith('_facenet_features'):
            base_name = base_name[:-len('_facenet_features')]
            
        label = None
        
        try:
            if "Actor_" in file_path: # RAVDESS
                parts = base_name.split('-')
                if len(parts) >= 3:
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                    else:
                        print(f"Warning: RAVDESS emotion code '{parts[2]}' not in mapping for {file_path}")
                else:
                    print(f"Warning: RAVDESS filename doesn't have enough parts: {base_name}")
            else: # CREMA-D
                parts = base_name.split('_')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                    else:
                        print(f"Warning: CREMA-D emotion code '{emotion_code}' not in mapping for {file_path}")
                else:
                    print(f"Warning: CREMA-D filename doesn't have enough parts: {base_name}")
        except Exception as e:
            print(f"Label parsing error for {file_path}: {e}")
            label = None
            
        # Check if file exists and has video_features
        if label is not None:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if 'video_features' in data and data['video_features'].shape[0] > 0:
                        valid_files.append(file_path)
                        labels.append(label)
                    else:
                        print(f"Warning: 'video_features' key missing or empty in {file_path}")
                        skipped += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                skipped += 1
        else:
            skipped += 1
            
    print(f"Valid files: {len(valid_files)}, Skipped: {skipped}")
    return valid_files, np.array(labels)

def main():
    """Main test function"""
    print("=== Testing Facenet Dataset Handling ===")
    
    # Get dataset paths
    ravdess_dir, crema_d_dir = get_dataset_paths()
    
    # Find all .npz files
    print("\nSearching for dataset files...")
    ravdess_files = glob.glob(os.path.join(ravdess_dir, "Actor_*", "*.npz"))
    crema_d_files = glob.glob(os.path.join(crema_d_dir, "*.npz"))
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(crema_d_files)} CREMA-D files")
    
    # Test loading a few files from each dataset
    print("\nTesting file loading with allow_pickle=True:")
    
    if ravdess_files:
        print("\nRAVDESS sample file:")
        sample_file = ravdess_files[0]
        print(f"  File: {sample_file}")
        try:
            with np.load(sample_file, allow_pickle=True) as data:
                print(f"  Keys: {data.files}")
                if 'video_features' in data:
                    print(f"  video_features shape: {data['video_features'].shape}")
                else:
                    print("  'video_features' key not found!")
        except Exception as e:
            print(f"  Error loading RAVDESS file: {e}")
    else:
        print("No RAVDESS files found to test")
        
    if crema_d_files:
        print("\nCREMA-D sample file:")
        sample_file = crema_d_files[0]
        print(f"  File: {sample_file}")
        try:
            with np.load(sample_file, allow_pickle=True) as data:
                print(f"  Keys: {data.files}")
                if 'video_features' in data:
                    print(f"  video_features shape: {data['video_features'].shape}")
                else:
                    print("  'video_features' key not found!")
        except Exception as e:
            print(f"  Error loading CREMA-D file: {e}")
    else:
        print("No CREMA-D files found to test")
    
    # Test label extraction
    print("\nTesting label extraction:")
    all_files = ravdess_files + crema_d_files
    
    if all_files:
        # Take a subset for quicker testing
        test_files = all_files[:min(100, len(all_files))]
        valid_files, labels = extract_labels(test_files)
        
        print(f"Files with valid labels/features: {len(valid_files)}")
        
        if len(valid_files) > 0:
            # Count emotions in test set
            emotion_counts = np.sum(labels, axis=0)
            emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
            print("\nEmotion distribution:")
            for i, emotion in enumerate(emotions):
                print(f"  {emotion}: {int(emotion_counts[i])}")
            
            # Test generator
            print("\nTesting generator with a small batch:")
            try:
                batch_size = min(4, len(valid_files))
                generator = VideoOnlyFacenetGenerator(
                    video_feature_files=valid_files[:batch_size*2],
                    labels=labels[:batch_size*2],
                    batch_size=batch_size,
                    shuffle=True
                )
                
                print(f"Generator created, {len(generator)} batches")
                print(f"Video feature dimension: {generator.video_feature_dim}")
                
                # Try getting a batch
                x_batch, y_batch = generator[0]
                print(f"Got batch: X shape {x_batch.shape}, Y shape {y_batch.shape}")
                print("Generator test successful")
            except Exception as e:
                print(f"Error testing generator: {e}")
        else:
            print("No valid files found, cannot test generator")
    else:
        print("No files found, cannot test label extraction or generator")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
