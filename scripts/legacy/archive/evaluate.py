#!/usr/bin/env python3
"""
Evaluation script for the multimodal emotion recognition model.
"""

import os
import sys
import glob
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from .synchronize_test import parse_ravdess_filename
from .train import load_data, create_multimodal_datasets  # Import functions from train.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluate.log"),
        logging.StreamHandler()
    ]
)

def evaluate_multimodal_model(model, X_test_video, X_test_audio, y_test):
    """Evaluates the dual-stream model on the test set.

    Args:
        model: Trained Keras model.
        X_test_video, X_test_audio: Test features for video and audio.
        y_test: Test labels.
    """
    loss, accuracy, precision, recall = model.evaluate(
        [X_test_video, X_test_audio], y_test, verbose=1
    )
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score:.4f}")

    # Calculate and print confusion matrix
    y_pred = model.predict([X_test_video, X_test_audio])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

def main():
    # --- Configuration ---
    MODEL_PATH = 'models/'  # Path to saved models
    DATA_DIR = 'processed_features_test'  # Directory with processed data
    DATASET_NAME = 'RAVDESS'
    NUM_CLASSES = 8

    # --- Load the latest trained model ---
    try:
        # Look for multi-modal models first
        list_of_files = glob.glob(os.path.join(MODEL_PATH, 'multimodal_model_*.h5'))
        if not list_of_files:
            # If no multimodal models found, look for any .h5 files
            list_of_files = glob.glob(os.path.join(MODEL_PATH, '*.h5'))
            
        if not list_of_files:
            logging.error(f"No trained models found in {MODEL_PATH}")
            sys.exit(1)
            
        latest_model = max(list_of_files, key=os.path.getctime)
        logging.info(f"Loading model: {latest_model}")
        model = load_model(latest_model)
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        sys.exit(1)

    # --- Load Data ---
    video_sequences, audio_sequences, labels, filenames = load_data(DATA_DIR, DATASET_NAME)
    if video_sequences is None or audio_sequences is None:
        logging.error("Failed to load data.")
        sys.exit(1)

    # Convert labels to categorical
    labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)

    # --- Create Datasets ---
    dataset = create_multimodal_datasets(video_sequences, audio_sequences, labels_categorical)
    if dataset is None:
        logging.error("Failed to create datasets.")
        sys.exit(1)
    
    # Extract test data
    _, _, _, _, _, _, X_test_video, X_test_audio, y_test = dataset

    # --- Padding Video Sequences ---
    video_max_len = max(seq.shape[0] for seq in X_test_video)
    video_dim = X_test_video[0].shape[1]
    
    X_test_video_padded = np.zeros((len(X_test_video), video_max_len, video_dim))
    for i, seq in enumerate(X_test_video):
        X_test_video_padded[i, :seq.shape[0]] = seq
    
    # --- Padding Audio Sequences ---
    audio_max_len = max(seq.shape[0] for seq in X_test_audio)
    audio_dim = X_test_audio[0].shape[1]
    
    X_test_audio_padded = np.zeros((len(X_test_audio), audio_max_len, audio_dim))
    for i, seq in enumerate(X_test_audio):
        X_test_audio_padded[i, :seq.shape[0]] = seq

    # --- Evaluate Model ---
    evaluate_multimodal_model(model, X_test_video_padded, X_test_audio_padded, y_test)

if __name__ == "__main__":
    main()
