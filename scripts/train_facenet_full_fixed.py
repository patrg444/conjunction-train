#!/usr/bin/env python3
"""
Full-scale training script for Facenet video-only emotion recognition (Fixed Version)
"""
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from fixed_video_facenet_generator import FixedVideoFacenetGenerator

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facenet_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("facenet_training")

def create_model(max_seq_len, feature_dim=512, num_classes=6):
    """Create LSTM model for video-only emotion recognition."""
    model = Sequential([
        LSTM(128, input_shape=(max_seq_len, feature_dim), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def find_feature_files(data_dir):
    """Find all valid feature files in the specified directory."""
    logger.info(f"Searching for feature files in {data_dir}")
    # --- FIXED LINE ---
    feature_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    logger.info(f"Found {len(feature_files)} feature files")
    return feature_files

def get_emotion_labels(feature_files):
    """Extract emotion labels from filenames."""
    # Map emotion codes to indices
    emotions = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}

    labels = []
    valid_files = []

    for filename in feature_files:
        base = os.path.basename(filename)
        for emotion, idx in emotions.items():
            if emotion in base:
                valid_files.append(filename)
                labels.append(idx)
                break

    logger.info(f"Found {len(valid_files)} files with valid emotion labels")
    return valid_files, np.array(labels)

def train_model(data_dir="crema_d_features_facenet", batch_size=32, epochs=100):
    """Run full training on feature files."""
    # --- FIXED PATH LOGIC ---
    # Assume the script runs from facenet_full_training, so look one level up for the data
    parent_dir = os.path.dirname(os.getcwd()) # Get the parent directory (e.g., /home/ubuntu/emotion-recognition)
    full_data_path = os.path.join(parent_dir, data_dir) # Construct path like /home/ubuntu/emotion-recognition/crema_d_features_facenet
    logger.info(f"Constructed full data path: {full_data_path}")

    # Find all feature files using the corrected path
    feature_files = find_feature_files(full_data_path)

    # Get emotion labels
    valid_files, labels = get_emotion_labels(feature_files)

    if len(valid_files) == 0:
        logger.error(f"No valid files with emotion labels found in {full_data_path}!")
        sys.exit(1)

    # Split into train/val
    np.random.seed(42)
    indices = np.arange(len(valid_files))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_files = [valid_files[i] for i in train_indices]
    train_labels = labels[train_indices]

    val_files = [valid_files[i] for i in val_indices]
    val_labels = labels[val_indices]

    logger.info(f"Train samples: {len(train_files)}")
    logger.info(f"Validation samples: {len(val_files)}")

    # Calculate max sequence length from training set
    max_seq_len = 0
    sample_size = min(100, len(train_files)) # Sample up to 100 files to estimate max length
    sample_indices = np.random.choice(len(train_files), sample_size, replace=False)

    logger.info("Calculating maximum sequence length from a sample of training files...")
    for idx in sample_indices:
        try:
            file_path = train_files[idx]
            with np.load(file_path, allow_pickle=True) as data:
                if 'video_features' in data:
                    features = data['video_features']
                    if features.ndim == 2: # Check if features are correctly shaped (frames, feature_dim)
                         max_seq_len = max(max_seq_len, features.shape[0])
                    else:
                        logger.warning(f"Skipping file {file_path} due to unexpected feature dimensions: {features.shape}")
                else:
                    logger.warning(f"Skipping file {file_path} as 'video_features' key not found.")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    if max_seq_len == 0:
        logger.error("Could not determine maximum sequence length from sample files. Exiting.")
        sys.exit(1)

    logger.info(f"Determined maximum sequence length: {max_seq_len}")

    # Initialize data generators
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        normalize_features=True,
        max_seq_len=max_seq_len
    )

    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        normalize_features=True,
        max_seq_len=max_seq_len
    )

    # Setup model
    model = create_model(max_seq_len=max_seq_len)
    model.summary()

    # Create model save directory
    model_dir = f"models/facenet_lstm_{batch_size}_{epochs}"
    os.makedirs(model_dir, exist_ok=True)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"logs/facenet_lstm_{batch_size}_{epochs}",
            histogram_freq=1
        )
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        # workers=4, # Removed as it caused TypeError with this generator/TF version
        # use_multiprocessing=True, # Removed as it caused TypeError
        verbose=1
    )

    # Save final model
    model.save(os.path.join(model_dir, "final_model.h5"))
    logger.info(f"Training complete. Final model saved to {os.path.join(model_dir, 'final_model.h5')}")

    return model, history

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Facenet video-only emotion recognition model")
    parser.add_argument("--data-dir", type=str, default="crema_d_features_facenet",
                        help="Directory containing feature files (relative to parent dir)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
