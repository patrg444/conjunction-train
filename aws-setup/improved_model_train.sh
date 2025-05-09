#!/bin/bash
# Script to train an improved model with better accuracy

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Creating an improved training script with higher accuracy target..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training

# Kill any previous training processes
pkill -f "python"

# Create an enhanced training script that achieves better accuracy
cat > scripts/train_improved.py << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script for emotion recognition with improved architecture
and advanced techniques to achieve higher accuracy.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import glob
import random

# Global variables
BATCH_SIZE = 24
EPOCHS = 50  # More epochs for better convergence
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 10  # Increased patience for better convergence
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility
SEQUENCE_LENGTH = 75  # Longer sequence for better temporal features

print("IMPROVED TRAINING SCRIPT STARTING")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

def create_enhanced_model(audio_input_shape, video_input_shape):
    """Create an enhanced branched model with more sophisticated architecture."""
    print("Creating enhanced branched model with:")
    print("- Audio input shape:", audio_input_shape)
    print("- Video input shape:", video_input_shape)
    
    # Audio branch - improved with convolutional layers
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    
    # Apply 1D convolutions to extract local patterns
    audio_x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_input)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    audio_x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    # Apply bidirectional LSTM for temporal features
    audio_x = Bidirectional(LSTM(128, return_sequences=True))(audio_x)
    audio_x = Dropout(0.3)(audio_x)
    audio_x = Bidirectional(LSTM(64))(audio_x)
    audio_x = Dense(128, activation='relu')(audio_x)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch - improved with more capacity
    video_input = Input(shape=video_input_shape, name='video_input')
    
    # FaceNet features already have high dimensionality, so we'll use LSTM directly
    video_x = Bidirectional(LSTM(256, return_sequences=True))(video_input)
    video_x = Dropout(0.3)(video_x)
    video_x = Bidirectional(LSTM(128))(video_x)
    video_x = Dense(256, activation='relu')(video_x)
    video_x = Dropout(0.4)(video_x)
    
    # Merge branches with more sophisticated fusion
    merged = Concatenate()([audio_x, video_x])
    merged = Dense(256, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[audio_input, video_input], outputs=output)
    
    # Compile model with reduced learning rate for better convergence
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_data(path_pattern, max_files=None):
    """Load and process NPZ files containing features."""
    files = glob.glob(path_pattern)
    if max_files:
        files = files[:max_files]
    
    if not files:
        print("Error: No files found matching pattern: %s" % path_pattern)
        return None, None, None
    
    print("Processing %d files from %s" % (len(files), path_pattern))
    
    # Lists to hold our data
    audio_data = []
    video_data = []
    labels = []
    
    # Maps emotion strings to integers
    emotion_map = {
        'ANG': 0,  # Anger
        'DIS': 1,  # Disgust
        'FEA': 2,  # Fear
        'HAP': 3,  # Happy
        'NEU': 4,  # Neutral
        'SAD': 5   # Sad
    }
    
    # RAVDESS emotion mapping (from file naming convention)
    ravdess_emotion_map = {
        '01': 'NEU',  # neutral
        '02': 'ANG',  # calm (not used) 
        '03': 'HAP',  # happy
        '04': 'SAD',  # sad
        '05': 'ANG',  # angry
        '06': 'FEA',  # fearful
        '07': 'DIS',  # disgust
        '08': 'NEU'   # surprised (map to neutral)
    }
    
    skipped = 0
    for file_path in files:
        try:
            # Extract emotion from filename
            filename = os.path.basename(file_path)
            
            # Check if this is a RAVDESS file (contains hyphens in filename)
            if '-' in filename:
                # RAVDESS format: 01-01-03-01-01-01-12.npz
                # where the 3rd segment (03) is the emotion code
                parts = filename.split('-')
                if len(parts) >= 3:
                    ravdess_code = parts[2]
                    emotion_code = ravdess_emotion_map.get(ravdess_code, None)
                else:
                    emotion_code = None
            else:
                # CREMA-D format: 1001_DFA_ANG_XX.npz
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None
            
            if emotion_code not in emotion_map:
                # Skip files with emotions not in our map
                skipped += 1
                continue
            
            # Load the npz file
            data = np.load(file_path)
            
            # Check that both features exist
            if 'audio_features' not in data or 'video_features' not in data:
                skipped += 1
                continue
            
            # Get features
            audio_features = data['audio_features']
            video_features = data['video_features']
            
            # Skip sequences that are too short
            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue
            
            # Append to lists
            audio_data.append(audio_features)
            video_data.append(video_features)
            
            # Create one-hot encoded label
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)
            
        except Exception as e:
            print("Error processing file %s: %s" % (file_path, str(e)))
            skipped += 1
    
    print("Processed %d files, skipped %d files" % (len(audio_data), skipped))
    
    return audio_data, video_data, np.array(labels)

def manual_train_test_split(data_length, train_ratio=0.8, random_seed=42):
    """Manual implementation of train/test split using NumPy."""
    np.random.seed(random_seed)
    indices = np.arange(data_length)
    np.random.shuffle(indices)
    
    split_idx = int(data_length * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    return train_indices, val_indices

def pad_sequences(sequences, max_length=None):
    """Pad sequences to the same length with more sophisticated handling."""
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    padded_sequences = []
    for seq in sequences:
        # If sequence is too short, pad it
        if len(seq) < max_length:
            # Get the shape of the sequence items
            item_shape = seq[0].shape
            
            # Create padding - mirror the last few frames instead of zeros
            # for better feature continuity
            if len(seq) > 5:  # If we have enough frames to mirror
                # Take the last few frames and reverse them
                mirror_frames = seq[-5:][::-1]
                # Calculate how many repeats we need
                repeat_count = int(np.ceil((max_length - len(seq)) / len(mirror_frames)))
                # Create the padding array by repeating the mirror frames
                padding = np.tile(mirror_frames, (repeat_count, 1))
                # Take only what we need
                padding = padding[:(max_length - len(seq))]
            else:
                # Fall back to zero padding if not enough frames
                padding = np.zeros((max_length - len(seq),) + item_shape)
            
            # Concatenate original sequence with padding
            padded_seq = np.vstack([seq, padding])
        # If sequence is too long, use a more intelligent subsampling
        elif len(seq) > max_length:
            # Take frames evenly distributed across the sequence
            indices = np.round(np.linspace(0, len(seq) - 1, max_length)).astype(int)
            padded_seq = seq[indices]
        # If sequence is exactly the right length, use as is
        else:
            padded_seq = seq
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

def augment_data(audio_data, video_data, labels):
    """Apply data augmentation to increase dataset diversity."""
    augmented_audio = []
    augmented_video = []
    augmented_labels = []
    
    # For each sample in the dataset
    for i in range(len(audio_data)):
        # Add the original data
        augmented_audio.append(audio_data[i])
        augmented_video.append(video_data[i])
        augmented_labels.append(labels[i])
        
        # Only augment non-neutral emotions to preserve class balance
        if np.argmax(labels[i]) != 4:  # 4 is neutral
            # Apply augmentation 1: Add slight noise to audio features
            noise_level = 0.02  # 2% noise
            noise = np.random.normal(0, noise_level, audio_data[i].shape)
            noisy_audio = audio_data[i] + noise
            
            augmented_audio.append(noisy_audio)
            augmented_video.append(video_data[i])  # Keep video unchanged
            augmented_labels.append(labels[i])
            
            # Apply augmentation 2: Time stretching simulation - take fewer frames
            if len(audio_data[i]) > 10 and len(video_data[i]) > 10:
                stretch_factor = 0.9  # 90% of frames
                num_frames_audio = int(len(audio_data[i]) * stretch_factor)
                num_frames_video = int(len(video_data[i]) * stretch_factor)
                
                # Ensure we have enough frames
                if num_frames_audio >= 5 and num_frames_video >= 5:
                    # Evenly sample frames
                    audio_indices = np.round(np.linspace(0, len(audio_data[i]) - 1, num_frames_audio)).astype(int)
                    video_indices = np.round(np.linspace(0, len(video_data[i]) - 1, num_frames_video)).astype(int)
                    
                    stretched_audio = audio_data[i][audio_indices]
                    stretched_video = video_data[i][video_indices]
                    
                    augmented_audio.append(stretched_audio)
                    augmented_video.append(stretched_video)
                    augmented_labels.append(labels[i])
    
    print(f"Data augmentation: original size={len(audio_data)}, augmented size={len(augmented_audio)}")
    
    return augmented_audio, augmented_video, np.array(augmented_labels)

def normalize_features(features):
    """Normalize features for better training stability."""
    # Calculate mean and std from the data
    mean = np.mean(features, axis=(0, 1), keepdims=True)
    std = np.std(features, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    # Normalize
    normalized = (features - mean) / std
    
    return normalized

def train_model():
    """Main function to train the model with improved methodology."""
    print("Starting enhanced model training with BOTH datasets...")
    
    # Process RAVDESS data - note the nested pattern to find files in actor folders
    ravdess_pattern = "ravdess_features_facenet/*/*.npz"
    ravdess_audio, ravdess_video, ravdess_labels = process_data(ravdess_pattern)
    
    # Process CREMA-D data
    cremad_pattern = "crema_d_features_facenet/*.npz"
    cremad_audio, cremad_video, cremad_labels = process_data(cremad_pattern)
    
    # Check if either dataset loaded successfully
    if ravdess_audio is None and cremad_audio is None:
        print("Error: Failed to load any data")
        return
    
    # Combine available datasets
    all_audio = []
    all_video = []
    all_labels = None  # Initialize as None
    
    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        all_labels = ravdess_labels  # Just assign directly the first time
        print(f"Added RAVDESS: {len(ravdess_audio)} samples")
    
    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels is None:
            all_labels = cremad_labels
        else:
            all_labels = np.vstack([all_labels, cremad_labels])
        print(f"Added CREMA-D: {len(cremad_audio)} samples")
    
    print(f"Combined: {len(all_audio)} total samples")
    
    # Apply data augmentation for more diversity
    print("Applying data augmentation...")
    all_audio, all_video, all_labels = augment_data(all_audio, all_video, all_labels)
    
    print("Enhanced dataset size:")
    print("- Number of samples:", len(all_audio))
    print("- Label distribution:")
    for i in range(NUM_CLASSES):
        count = np.sum(all_labels[:, i])
        print("  Class %d: %d samples (%.1f%%)" % (i, count, count/len(all_labels)*100))
    
    # Determine sequence length
    audio_max_len = max(len(seq) for seq in all_audio)
    video_max_len = max(len(seq) for seq in all_video)
    
    print("Maximum sequence lengths:")
    print("- Audio:", audio_max_len)
    print("- Video:", video_max_len)
    
    # Use increased sequence length
    common_max_len = min(SEQUENCE_LENGTH, max(audio_max_len, video_max_len))
    print("Using common max length:", common_max_len)
    
    # Pad sequences with improved padding
    audio_padded = pad_sequences(all_audio, common_max_len)
    video_padded = pad_sequences(all_video, common_max_len)
    
    # Normalize features for better training
    print("Normalizing features...")
    audio_padded = normalize_features(audio_padded)
    # Note: FaceNet features are already normalized, so we skip that for video
    
    print("Padded data shapes:")
    print("- Audio:", audio_padded.shape)
    print("- Video:", video_padded.shape)
    print("- Labels:", all_labels.shape)
    
    # Split into train/val sets (80/20 split) with stratification
    # First, we'll create indices for each class
    class_indices = [np.where(all_labels[:, i] == 1)[0] for i in range(NUM_CLASSES)]
    
    train_idx = []
    val_idx = []
    
    # For each class, take TRAIN_RATIO for training and the rest for validation
    np.random.seed(RANDOM_SEED)
    for indices in class_indices:
        np.random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        train_idx.extend(indices[:split_idx])
        val_idx.extend(indices[split_idx:])
    
    # Shuffle the indices
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    train_audio = audio_padded[train_idx]
    train_video = video_padded[train_idx]
    train_labels = all_labels[train_idx]
    
    val_audio = audio_padded[val_idx]
    val_video = video_padded[val_idx]
    val_labels = all_labels[val_idx]
    
    print("Train/Val split with stratification:")
    print("- Train samples:", len(train_audio))
    print("- Validation samples:", len(val_audio))
    
    # Get the feature dimensionality from the padded data
    audio_input_shape = (common_max_len, train_audio.shape[2])
    video_input_shape = (common_max_len, train_video.shape[2])
    
    # Create and compile the enhanced model
    model = create_enhanced_model(audio_input_shape, video_input_shape)
    model.summary()
    
    # Create output directories if they don't exist
    model_dir = "models/improved"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define callbacks with more sophisticated setup
    checkpoint_path = os.path.join(model_dir, "model_best.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            save_best_only=True,
            save_weights_only=False,
            mode='max',  # We want to maximize accuracy
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model with class weights to handle imbalance
    print("Starting training with improved model...")
    start_time = time.time()
    
    # Calculate class weights to handle imbalance
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)
    
    print("Using class weights to handle imbalance:", class_weights)
    
    history = model.fit(
        [train_audio, train_video],
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([val_audio, val_video], val_labels),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model.h5")
    model.save(final_model_path)
    print("Final model saved to:", final_model_path)
    
    # Calculate training time
    train_time = time.time() - start_time
    print("Training completed in %.2f seconds (%.2f minutes)" % (train_time, train_time/60))
    
    # Print training history summary
    print("Training history summary:")
    print("- Final training accuracy:", history.history['accuracy'][-1])
    print("- Final validation accuracy:", history.history['val_accuracy'][-1])
    print("- Best validation accuracy:", max(history.history['val_accuracy']))
    print("- Best validation loss:", min(history.history['val_loss']))
    
    return model, history

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        print(traceback.format_exc())
PYEOF

# Make the script executable
chmod +x scripts/train_improved.py

# Clear the current training log
> training.log

# Run the improved training script using the proper conda environment
echo "Starting enhanced training for higher accuracy..." > training.log
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310
echo "Conda environment activated, running script..." >> training.log
cd ~/emotion_training
nohup python scripts/train_improved.py >> training.log 2>&1 &
echo "Training restarted with enhanced model for higher accuracy. Check training.log for progress."
ps aux | grep python >> training.log
EOF

echo "Enhanced training script applied."
echo "This script includes several improvements for higher accuracy:"
echo "  1. Enhanced model architecture with convolutions and more capacity"
echo "  2. Data augmentation to increase training variety"
echo "  3. Stratified sampling to handle class imbalance"
echo "  4. Improved feature normalization"
echo "  5. Class weighting to focus on underrepresented emotions"
echo "  6. Longer sequences (75 frames vs. 50) for better temporal context"
echo "  7. Adaptive learning rate schedule"
echo ""
echo "Target accuracy: >70% validation accuracy"
echo ""
echo "Wait a minute and then check training progress with:"
echo "cd aws-setup && ./enhanced_monitor.sh"
