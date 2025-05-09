#!/bin/bash
# Script to create and run a training script that properly uses both datasets

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Creating a script to train using both CREMA-D and RAVDESS datasets..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training

# Kill any previous training processes
pkill -f "python"

# Create a training script that uses both datasets
cat > scripts/train_combined.py << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the branched model using both CREMA-D and RAVDESS datasets.
Fixed to correctly handle RAVDESS actor folders structure.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import glob
import random

# Global variables
BATCH_SIZE = 32
EPOCHS = 20  # Reduced epochs for faster training
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 5   # Reduced patience
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility

print("COMBINED DATASETS TRAINING SCRIPT STARTING")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

def create_model(audio_input_shape, video_input_shape):
    """Create a branched model that processes audio and video features separately before merging."""
    print("Creating branched model with:")
    print("- Audio input shape:", audio_input_shape)
    print("- Video input shape:", video_input_shape)
    
    # Audio branch
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    audio_x = Bidirectional(LSTM(64, return_sequences=True))(audio_input)  # Reduced size
    audio_x = Bidirectional(LSTM(32))(audio_x)  # Reduced size
    audio_x = Dense(32, activation='relu')(audio_x)  # Reduced size
    audio_x = Dropout(0.5)(audio_x)
    
    # Video branch
    video_input = Input(shape=video_input_shape, name='video_input')
    video_x = Bidirectional(LSTM(64, return_sequences=True))(video_input)  # Reduced size
    video_x = Bidirectional(LSTM(32))(video_x)  # Reduced size
    video_x = Dense(32, activation='relu')(video_x)  # Reduced size
    video_x = Dropout(0.5)(video_x)
    
    # Merge branches
    merged = Concatenate()([audio_x, video_x])
    merged = Dense(64, activation='relu')(merged)  # Reduced size
    merged = Dropout(0.5)(merged)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[audio_input, video_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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
    """Pad sequences to the same length."""
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    padded_sequences = []
    for seq in sequences:
        # If sequence is too short, pad it
        if len(seq) < max_length:
            # Get the shape of the sequence items (e.g., (88,) for audio features)
            item_shape = seq[0].shape
            # Create padding
            padding = np.zeros((max_length - len(seq),) + item_shape)
            # Concatenate original sequence with padding
            padded_seq = np.vstack([seq, padding])
        # If sequence is too long, truncate it
        elif len(seq) > max_length:
            padded_seq = seq[:max_length]
        # If sequence is exactly the right length, use as is
        else:
            padded_seq = seq
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

def train_model():
    """Main function to train the model."""
    print("Starting branched model training with BOTH datasets...")
    
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
    all_labels = []
    
    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        if all_labels == []:
            all_labels = ravdess_labels
        else:
            all_labels = np.vstack([all_labels, ravdess_labels])
    
    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels == []:
            all_labels = cremad_labels
        else:
            all_labels = np.vstack([all_labels, cremad_labels])
    
    print("Dataset size:")
    print("- Number of samples:", len(all_audio))
    print("- Label distribution:")
    for i in range(NUM_CLASSES):
        count = np.sum(all_labels[:, i])
        print("  Class %d: %d samples (%.1f%%)" % (i, count, count/len(all_labels)*100))
    
    # Determine maximum sequence lengths
    audio_max_len = max(len(seq) for seq in all_audio)
    video_max_len = max(len(seq) for seq in all_video)
    
    print("Maximum sequence lengths:")
    print("- Audio:", audio_max_len)
    print("- Video:", video_max_len)
    
    # Standardize sequence lengths - use smaller length for quicker training
    common_max_len = min(50, max(audio_max_len, video_max_len))
    print("Using common max length:", common_max_len)
    
    # Pad sequences
    audio_padded = pad_sequences(all_audio, common_max_len)
    video_padded = pad_sequences(all_video, common_max_len)
    
    print("Padded data shapes:")
    print("- Audio:", audio_padded.shape)
    print("- Video:", video_padded.shape)
    print("- Labels:", all_labels.shape)
    
    # Split into train/val sets (80/20 split) using our manual function
    train_idx, val_idx = manual_train_test_split(
        len(all_labels), train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )
    
    train_audio = audio_padded[train_idx]
    train_video = video_padded[train_idx]
    train_labels = all_labels[train_idx]
    
    val_audio = audio_padded[val_idx]
    val_video = video_padded[val_idx]
    val_labels = all_labels[val_idx]
    
    print("Train/Val split:")
    print("- Train samples:", len(train_audio))
    print("- Validation samples:", len(val_audio))
    
    # Get the feature dimensionality from the padded data
    audio_input_shape = (common_max_len, train_audio.shape[2])
    video_input_shape = (common_max_len, train_video.shape[2])
    
    # Create and compile the model
    model = create_model(audio_input_shape, video_input_shape)
    model.summary()
    
    # Create output directories if they don't exist
    model_dir = "models/combined"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define callbacks
    checkpoint_path = os.path.join(model_dir, "model_best.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        [train_audio, train_video],
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([val_audio, val_video], val_labels),
        callbacks=callbacks,
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
chmod +x scripts/train_combined.py

# Clear the current training log
> training.log

# Run the fixed training script using the proper conda environment
echo "Starting combined datasets training script..." > training.log
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310
echo "Conda environment activated, running script..." >> training.log
cd ~/emotion_training
nohup python scripts/train_combined.py >> training.log 2>&1 &
echo "Training restarted with combined datasets script. Check training.log for progress."
ps aux | grep python >> training.log
EOF

echo "Combined datasets training script applied."
echo "Wait a minute and then check training progress with:"
echo "cd aws-setup && ./check_training_log.sh"
