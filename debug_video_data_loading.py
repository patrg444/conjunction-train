#!/usr/bin/env python3
import os
import glob
import numpy as np
from tqdm import tqdm

def test_load_function():
    """Directly test the load_data_paths_and_labels_video_only function from the training script"""
    # Define paths (same as in the training script)
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    
    # Print directory info
    print(f"RAVDESS directory exists: {os.path.exists(RAVDESS_FACENET_DIR)}")
    print(f"CREMA-D directory exists: {os.path.exists(CREMA_D_FACENET_DIR)}")
    
    # These are the exact globs from the training script
    facenet_files = glob.glob(os.path.join(RAVDESS_FACENET_DIR, "Actor_*", "*.npz")) + \
                    glob.glob(os.path.join(CREMA_D_FACENET_DIR, "*.npz"))
    
    print(f"Found {len(facenet_files)} precomputed Facenet files")
    
    # Define the exact same emotion maps as in the training script
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}  # Including '08' mapping
    
    # Track statistics
    labels = []
    valid_facenet_files = []
    skipped_label = 0
    skipped_reasons = {}
    
    # Process exactly as in the training script
    print("\nProcessing files to extract labels...")
    try:
        for facenet_file in tqdm(facenet_files, desc="Extracting labels"):
            base_name = os.path.splitext(os.path.basename(facenet_file))[0]
            if base_name.endswith('_facenet_features'):
                base_name = base_name[:-len('_facenet_features')]
            
            label = None
            reason = None
            
            # Extract label (exact same logic)
            try:
                if "Actor_" in facenet_file:  # RAVDESS
                    parts = base_name.split('-')
                    if len(parts) >= 3:
                        emotion_code = ravdess_emotion_map.get(parts[2], None)
                        if emotion_code in emotion_map:
                            label = np.zeros(len(emotion_map))
                            label[emotion_map[emotion_code]] = 1
                        else:
                            reason = f"RAVDESS emotion code '{parts[2]}' not mapped correctly"
                    else:
                        reason = f"RAVDESS filename doesn't have enough parts: {base_name}"
                else:  # CREMA-D
                    parts = base_name.split('_')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in emotion_map:
                            label = np.zeros(len(emotion_map))
                            label[emotion_map[emotion_code]] = 1
                        else:
                            reason = f"CREMA-D emotion code '{emotion_code}' not in mapping"
                    else:
                        reason = f"CREMA-D filename doesn't have enough parts: {base_name}"
            except Exception as e:
                reason = f"Label parsing error: {str(e)}"
                label = None
            
            # Check if file valid (exact same logic)
            if label is not None:
                if os.path.exists(facenet_file) and os.path.getsize(facenet_file) > 0:
                    try:
                        with np.load(facenet_file) as data:
                            if 'video_features' in data and data['video_features'].shape[0] > 0:
                                valid_facenet_files.append(facenet_file)
                                labels.append(label)
                            else:
                                reason = f"Missing 'video_features' key or empty shape"
                                skipped_label += 1
                    except Exception as load_e:
                        reason = f"Error loading npz: {str(load_e)}"
                        skipped_label += 1
                else:
                    reason = "File doesn't exist or is empty"
                    skipped_label += 1
            else:
                # Label is None, reason should be set
                skipped_label += 1
            
            # Track reason for skipping if applicable
            if reason:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
    except Exception as e:
        print(f"Unexpected error during processing: {str(e)}")
    
    # Print results
    print(f"\nFound {len(valid_facenet_files)} Facenet files with valid labels and 'video_features'")
    print(f"Skipped {skipped_label} files")
    
    # Print specific skipping reasons
    print("\nSkip reasons:")
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"- {reason}: {count} files")
    
    # Print a few examples of valid files
    if valid_facenet_files:
        print("\nSample valid files:")
        for file in valid_facenet_files[:5]:
            print(f"- {file}")
    
    # Print stats by dataset
    ravdess_count = sum(1 for f in valid_facenet_files if "Actor_" in f)
    cremad_count = len(valid_facenet_files) - ravdess_count
    print(f"\nValid RAVDESS files: {ravdess_count}")
    print(f"Valid CREMA-D files: {cremad_count}")
    
    return valid_facenet_files, np.array(labels) if labels else None

if __name__ == "__main__":
    files, labels = test_load_function()
    if labels is not None:
        print(f"\nTotal valid files: {len(files)}")
        print(f"Labels shape: {labels.shape}")
    else:
        print("\nNo valid files found with labels.")
