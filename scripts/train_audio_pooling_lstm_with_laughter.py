#!/usr/bin/env python3
"""
Train an audio-pooling LSTM model with an auxiliary laughter detection branch.
This extends the standard emotion recognition model with an additional
output for detecting laughter in audio-visual input.

Usage:
    python train_audio_pooling_lstm_with_laughter.py [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN] [--laugh_weight LAUGH_WEIGHT]
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight

from audio_pooling_generator import AudioPoolingDataGenerator
from feature_normalizer import normalize_features, load_normalization_stats, save_normalization_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train audio-pooling LSTM model with laughter detection.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=45, help="Maximum sequence length")
    parser.add_argument("--laugh_weight", type=float, default=0.3, help="Weight for laughter loss")
    parser.add_argument("--model_dir", type=str, default="models/dynamic_padding_no_leakage", help="Model directory")
    parser.add_argument("--laugh_manifest", type=str, default="datasets_raw/manifests/laughter_v1.csv",
                      help="Path to laughter manifest file")
    return parser.parse_args()


def build_model(max_seq_len, num_emotions=8, laugh_weight=0.3):
    """
    Build audio-pooling LSTM model for emotion recognition only.
    """
    # Audio input (eGeMAPS features)
    audio_input = Input(shape=(max_seq_len, 88), name="audio_input")

    # Video input (FaceNet embeddings)
    video_input = Input(shape=(max_seq_len, 512), name="video_input")

    # Audio branch
    audio_lstm = LSTM(128, return_sequences=True, dropout=0.3, name="audio_lstm")(audio_input)

    # Video branch
    video_lstm = LSTM(128, return_sequences=True, dropout=0.3, name="video_lstm")(video_input)

    # Concatenate branches
    concat = Concatenate(axis=2, name="audio_video_concat")([audio_lstm, video_lstm])

    # Shared encoder
    shared_lstm = LSTM(256, return_sequences=False, dropout=0.3, name="shared_lstm")(concat)
    shared_dropout = Dropout(0.5, name="shared_dropout")(shared_lstm)

    # Main emotion output only
    emotion_output = Dense(num_emotions, activation="softmax", name="emotion_output")(shared_dropout)

    # Create single-output model
    # Use a dictionary for inputs so tf.data builds a valid output_signature
    model = Model(
        inputs={"audio_input": audio_input, "video_input": video_input},
        outputs=emotion_output
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def prepare_data_generators(max_seq_len, batch_size, laugh_manifest):
    """
    Prepare data generators for training.

    Args:
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        laugh_manifest: Path to laughter manifest file

    Returns:
        train_gen, val_gen, test_gen: Data generators for train/val/test
    """
    # Setup standard emotion data generator
    train_gen = AudioPoolingDataGenerator(
        "train",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=True,
        normalize_features=True,
        dynamic_padding=True,
        padding_mode="repeat_last"
    )

    val_gen = AudioPoolingDataGenerator(
        "val",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=False,
        normalize_features=True,
        dynamic_padding=True,
        padding_mode="repeat_last"
    )

    test_gen = AudioPoolingDataGenerator(
        "test",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=False,
        normalize_features=True,
        dynamic_padding=True,
        padding_mode="repeat_last"
    )

    # Laughter manifest handling removed: single-output mode (laughter labels not used)
    if os.path.exists(laugh_manifest):
        print(f"Laughter manifest {laugh_manifest} found, but laughter labels are not used in single-output mode.")
    else:
        print(f"Warning: Laughter manifest {laugh_manifest} not found. Training without laughter detection.")

    return train_gen, val_gen, test_gen


def load_laughter_manifest(manifest_path):
    """
    Load laughter manifest and convert to dictionary by split.

    Args:
        manifest_path: Path to manifest CSV file

    Returns:
        Dictionary with train/val/test splits containing {file_id: laugh_label} mappings
    """
    import pandas as pd

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Initialize result dictionaries
    laughter_data = {
        "train": {},
        "val": {},
        "test": {}
    }

    # Process each row
    for _, row in df.iterrows():
        # Create a file_id from filepath (base name without extension)
        file_id = os.path.splitext(os.path.basename(row["filepath"]))[0]

        # Add to appropriate split
        split = row["split"]
        laugh_label = int(row["laugh"])

        if split in laughter_data:
            laughter_data[split][file_id] = laugh_label

    return laughter_data


def setup_callbacks(model_dir):
    """
    Setup training callbacks.

    Args:
        model_dir: Directory to save model checkpoints

    Returns:
        List of Keras callbacks
    """
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "model_best.h5"),
            monitor="val_emotion_output_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_emotion_output_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_emotion_output_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
            histogram_freq=1,
            profile_batch=0
        )
    ]

    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Build model
    model = build_model(
        max_seq_len=args.max_seq_len,
        num_emotions=8,  # RAVDESS has 8 emotions
        laugh_weight=args.laugh_weight
    )
    model.summary()

    # Prepare data generators
    train_gen, val_gen, test_gen = prepare_data_generators(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        laugh_manifest=args.laugh_manifest
    )

    # Setup callbacks
    callbacks = setup_callbacks(args.model_dir)

    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_file = os.path.join(args.model_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history.history, f, indent=2)

    # Evaluate on test set
    test_results = model.evaluate(test_gen, verbose=1)

    # Save test results
    test_metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        test_metrics[metric_name] = float(test_results[i])

    test_file = os.path.join(args.model_dir, "test_results.json")
    with open(test_file, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Training completed. Model saved to {args.model_dir}")
    print(f"Test results: {test_metrics}")


if __name__ == "__main__":
    main()
