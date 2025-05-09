#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test training script for Facenet LSTM model 
to verify the fixed data pipeline works correctly
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scripts.fixed_video_facenet_generator import FixedVideoFacenetGenerator
except ImportError:
    from fixed_video_facenet_generator import FixedVideoFacenetGenerator

def build_model(input_dim=512, max_seq_len=None):
    """Build the LSTM model architecture"""
    model = Sequential()
    
    # Add masking layer to handle variable-length sequences
    model.add(Masking(mask_value=0.0, input_shape=(max_seq_len, input_dim)))
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(128, return_sequences=True, 
                  dropout=0.3, recurrent_dropout=0.3))
    model.add(Dropout(0.3))
    
    # Second LSTM layer
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dropout(0.3))
    
    # Dense layers for classification
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))  # 6 emotions
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_model(ravdess_dir, cremad_dir, model_dir, batch_size=32, epochs=3):
    """Test training function with fewer epochs to verify data pipeline"""
    print("=== Test Training Facenet Video-Only LSTM Model ===")
    
    # Get all feature files and extract emotion labels
    all_feature_files = []
    all_labels = []
    
    # Emotion mappings
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {
        '01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', 
        '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'  # Map surprise to fear
    }
    
    # Function to extract emotion from filename
    def extract_emotion_from_filename(file_path):
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        
        # Remove '_facenet_features' suffix if it exists
        if file_name.endswith('_facenet_features'):
            file_name = file_name[:-len('_facenet_features')]
        
        emotion_code = None
        
        # RAVDESS format: 01-01-01-01-01-01-01.npz (emotion is 3rd segment)
        if "Actor_" in file_path:
            parts = file_name.split('-')
            if len(parts) >= 3:
                emotion_code = ravdess_emotion_map.get(parts[2], None)
        
        # CREMA-D format: 1001_DFA_ANG_XX.npz (emotion is 3rd segment)
        else:
            parts = file_name.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
        
        if emotion_code in emotion_map:
            return emotion_map[emotion_code]
        else:
            return None
    
    # Add RAVDESS files
    if os.path.exists(ravdess_dir):
        for actor_dir in os.listdir(ravdess_dir):
            actor_path = os.path.join(ravdess_dir, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.npz'):
                        file_path = os.path.join(actor_path, file)
                        emotion = extract_emotion_from_filename(file_path)
                        if emotion is not None:
                            all_feature_files.append(file_path)
                            all_labels.append(emotion)
    
    # Add CREMA-D files
    if os.path.exists(cremad_dir):
        for file in os.listdir(cremad_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(cremad_dir, file)
                emotion = extract_emotion_from_filename(file_path)
                if emotion is not None:
                    all_feature_files.append(file_path)
                    all_labels.append(emotion)
                
    print(f"Found {len(all_feature_files)} feature files with valid labels")
    
    # Convert labels to numpy array
    all_labels = np.array(all_labels)
    
    # Calculate emotion distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print("Emotion distribution:")
    for label, count in zip(unique_labels, counts):
        emotion_name = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"][int(label)]
        percentage = (count / len(all_labels)) * 100
        print(f"- {emotion_name}: {count} ({percentage:.1f}%)")
    
    # Create a train/val split (80/20)
    np.random.seed(42)
    indices = np.arange(len(all_feature_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [all_feature_files[i] for i in train_indices]
    train_labels = all_labels[train_indices]
    
    val_files = [all_feature_files[i] for i in val_indices]
    val_labels = all_labels[val_indices]
    
    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    
    # Initialize data generators
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        normalize_features=True
    )
    
    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        normalize_features=True
    )
    
    # Verify we have data
    print(f"Train generator: {len(train_gen)} batches")
    print(f"Val generator: {len(val_gen)} batches")
    
    # Function to determine max sequence length from files
    def get_max_sequence_length(files, sample_size=100):
        # Sample a subset of files to determine max length
        if len(files) > sample_size:
            sampled_files = np.random.choice(files, sample_size, replace=False)
        else:
            sampled_files = files
            
        max_length = 0
        for file_path in sampled_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if 'video_features' in data:
                        features = data['video_features']
                        if features.shape[0] > max_length:
                            max_length = features.shape[0]
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return max_length
    
    # Get max sequence length for model input
    max_seq_len = get_max_sequence_length(train_files)
    print(f"Maximum sequence length: {max_seq_len}")
    
    # Build model
    print("Building model...")
    model = build_model(input_dim=512, max_seq_len=max_seq_len)
    model.summary()
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model for a few epochs
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, "final_model.h5"))
    
    print(f"Model training completed. Model saved to {model_dir}")
    
    # Print training history summary
    print("\nTraining History Summary:")
    for i, (acc, val_acc) in enumerate(zip(history.history['accuracy'], 
                                         history.history['val_accuracy'])):
        print(f"Epoch {i+1}: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Test train Facenet video-only LSTM model")
    parser.add_argument("--ravdess_dir", type=str, default="./ravdess_features_facenet", 
                        help="Directory with RAVDESS features")
    parser.add_argument("--cremad_dir", type=str, default="./crema_d_features_facenet", 
                        help="Directory with CREMA-D features")
    parser.add_argument("--model_dir", type=str, default="./models/facenet_lstm", 
                        help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs for test training")
    args = parser.parse_args()
    
    train_model(
        ravdess_dir=args.ravdess_dir,
        cremad_dir=args.cremad_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
