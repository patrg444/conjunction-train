#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts frame-level facial emotion recognition (FER) features from videos
using a pre-trained model (e.g., ResNet50-FER+) and saves them as .npz files.

Assumes videos are located in standard RAVDESS/CREMA-D directory structures.
Requires MTCNN for face detection and a pre-trained Keras FER model.
"""

import os
import sys
import cv2
import numpy as np
import glob
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input # We might need this depending on model loading
# Placeholder for actual FER model loading - replace with correct import/loading
# from tensorflow.keras.applications import ResNet50 # Example
# from some_fer_library import load_fer_model # Example
from tqdm import tqdm
import time
import argparse

# --- Configuration ---
# Input video directories (Absolute paths on EC2)
# RAVDESS videos seem to be directly under Actor_* directories
RAVDESS_VIDEO_BASE_DIR = "/home/ubuntu/emotion-recognition"
# CREMA_D video path is currently unknown, remove for now
# CREMA_D_VIDEO_DIR = "/home/ubuntu/emotion-recognition/datasets/crema-d/videos"
# Output feature directories (Absolute paths on EC2)
RAVDESS_FER_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_fer"
CREMA_D_FER_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_fer" # Keep output dir defined

# Model Input Size (adjust based on the chosen FER model)
TARGET_SIZE = (224, 224) # Example for ResNet50

# Face Detection Confidence Threshold
MIN_CONFIDENCE = 0.95

# Batch size for model prediction (process multiple faces at once)
PREDICTION_BATCH_SIZE = 16

# --- Helper Functions ---

def load_fer_model_placeholder(target_size=(224, 224)):
    """
    Placeholder function to load the pre-trained FER model.
    Replace this with the actual model loading code.
    Should return a Keras model instance without the final classification layer.
    """
    print("--- PLACEHOLDER: Loading FER Model ---")
    # Example using ResNet50 - replace with actual FER model loading
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*target_size, 3), pooling='avg')
    # return base_model
    # --- Replace above with actual model loading ---

    # Create a dummy model for testing structure if no real model is available yet
    print("Warning: Using a dummy FER model. Replace load_fer_model_placeholder.")
    dummy_input = tf.keras.Input(shape=(*target_size, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(dummy_input) # Mimic pooling
    x = tf.keras.layers.Dense(512, activation='relu')(x) # Dummy embedding layer
    dummy_model = Model(inputs=dummy_input, outputs=x)
    print("--- Dummy FER Model Loaded ---")
    return dummy_model

def preprocess_face_placeholder(face_img):
    """
    Placeholder for preprocessing steps required by the specific FER model.
    (e.g., normalization, channel ordering). Replace with actual preprocessing.
    Input: face_img (numpy array, BGR format from OpenCV)
    Output: preprocessed_face (numpy array, ready for model input)
    """
    # Example preprocessing for ResNet50 (adjust for your FER model)
    # face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # preprocessed = tf.keras.applications.resnet50.preprocess_input(face_rgb)
    # return preprocessed

    # Dummy preprocessing: just normalize to [0, 1] and ensure float32
    face_normalized = face_img.astype(np.float32) / 255.0
    return face_normalized


# --- Main Extraction Logic ---

def extract_features_for_video(video_path, detector, fer_model, output_dir):
    """Extracts FER features for a single video file."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}_fer_features.npz")

    # Skip if features already exist
    if os.path.exists(output_filename):
        # print(f"Skipping {base_name}, features already exist.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Processing {base_name} ({frame_count} frames)...")

    all_frame_features = []
    frames_to_process = []
    frame_indices = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Store frame and index for batch processing
        frames_to_process.append(frame)
        frame_indices.append(frame_num)
        frame_num += 1

    cap.release()

    if not frames_to_process:
        print(f"Warning: No frames read from {video_path}")
        return

    detected_faces_batches = [] # List of batches, each batch is a list of face crops
    batch_frame_indices = [] # List of frame indices corresponding to faces in batches

    # Detect faces frame by frame (MTCNN is often better one by one)
    faces_this_video = []
    indices_this_video = []
    for idx, frame in enumerate(frames_to_process):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb)

        if results:
            # Find the face with the highest confidence or largest bounding box
            best_result = max(results, key=lambda r: r['confidence'])
            if best_result['confidence'] >= MIN_CONFIDENCE:
                x1, y1, width, height = best_result['box']
                x2, y2 = x1 + width, y1 + height
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1: # Check for valid crop dimensions
                    face = frame[y1:y2, x1:x2]
                    face_resized = cv2.resize(face, TARGET_SIZE)
                    faces_this_video.append(face_resized)
                    indices_this_video.append(frame_indices[idx]) # Store original frame index

    if not faces_this_video:
        print(f"Warning: No confident faces detected in {base_name}")
        # Save empty features if no faces detected
        np.savez_compressed(output_filename, fer_features=np.array([]))
        return

    # Batch faces for prediction
    num_faces = len(faces_this_video)
    face_embeddings = [None] * num_faces # Pre-allocate list

    for i in range(0, num_faces, PREDICTION_BATCH_SIZE):
        batch_faces_orig = faces_this_video[i : i + PREDICTION_BATCH_SIZE]
        batch_indices = indices_this_video[i : i + PREDICTION_BATCH_SIZE]

        # Preprocess the batch
        batch_faces_processed = np.array([preprocess_face_placeholder(face) for face in batch_faces_orig])

        # Get embeddings from FER model
        try:
            batch_embeddings = fer_model.predict(batch_faces_processed, verbose=0)
            # Store embeddings back into the pre-allocated list
            for j, emb in enumerate(batch_embeddings):
                 list_index = i + j
                 face_embeddings[list_index] = emb

        except Exception as e:
            print(f"Error predicting batch for {base_name} (indices {batch_indices}): {e}")
            # Fill with None or zeros for failed predictions in this batch
            for j in range(len(batch_faces_orig)):
                 list_index = i + j
                 # face_embeddings[list_index] = np.zeros(EMBEDDING_DIM) # Or handle differently

    # Filter out any None entries if errors occurred
    final_embeddings = [emb for emb in face_embeddings if emb is not None]

    if final_embeddings:
        all_frame_features = np.array(final_embeddings, dtype=np.float32)
        # print(f"  Extracted {all_frame_features.shape[0]} features with dim {all_frame_features.shape[1]}")
        np.savez_compressed(output_filename, fer_features=all_frame_features)
    else:
        print(f"Warning: No features extracted successfully for {base_name}")
        np.savez_compressed(output_filename, fer_features=np.array([]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FER features from videos.")
    parser.add_argument('--dataset', type=str, default='all', choices=['ravdess', 'crema_d', 'all'],
                        help="Which dataset to process ('ravdess', 'crema_d', or 'all').")
    args = parser.parse_args()

    print("Starting FER Feature Extraction...")
    start_total_time = time.time()

    # Create output directories if they don't exist
    os.makedirs(RAVDESS_FER_DIR, exist_ok=True)
    os.makedirs(CREMA_D_FER_DIR, exist_ok=True)

    # Load models
    print("Loading MTCNN face detector...")
    try:
        # Set GPU memory growth - important for running TF models alongside MTCNN
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                print(f"Memory growth must be set before GPUs have been initialized: {e}")
        else:
            print("No GPUs detected by TensorFlow.")

        detector = MTCNN()
        print("MTCNN loaded.")
    except Exception as e:
        print(f"Error loading MTCNN: {e}")
        sys.exit(1)

    print("Loading FER model...")
    try:
        fer_model = load_fer_model_placeholder(target_size=TARGET_SIZE)
        # Determine embedding dimension dynamically if possible
        # EMBEDDING_DIM = fer_model.output_shape[-1]
        # print(f"FER model loaded. Embedding dimension: {EMBEDDING_DIM}")
        print(f"FER model loaded (Placeholder). Output shape: {fer_model.output_shape}")
    except Exception as e:
        print(f"Error loading FER model: {e}")
        sys.exit(1)


    # Get list of video files
    video_files = []
    output_dirs = {}

    if args.dataset in ['ravdess', 'all']:
        # Use the base directory and glob pattern for Actor_* subdirs
        ravdess_files = glob.glob(os.path.join(RAVDESS_VIDEO_BASE_DIR, "Actor_*", "*.mp4"))
        video_files.extend(ravdess_files)
        for f in ravdess_files: output_dirs[f] = RAVDESS_FER_DIR
        print(f"Found {len(ravdess_files)} RAVDESS videos.")

    # Temporarily disable CREMA-D processing until path is known
    # if args.dataset in ['crema_d', 'all']:
    #     cremad_files = glob.glob(os.path.join(CREMA_D_VIDEO_DIR, "*.mp4"))
    #     video_files.extend(cremad_files)
    #     for f in cremad_files: output_dirs[f] = CREMA_D_FER_DIR
    #     print(f"Found {len(cremad_files)} CREMA-D videos.")

    print(f"Total videos to process: {len(video_files)}")

    # Process videos
    for video_path in tqdm(video_files, desc="Extracting FER features"):
        output_dir = output_dirs[video_path]
        extract_features_for_video(video_path, detector, fer_model, output_dir)

    end_total_time = time.time()
    print(f"\nFER feature extraction complete.")
    print(f"Total time: {end_total_time - start_total_time:.2f} seconds")
