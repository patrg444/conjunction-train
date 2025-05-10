#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early-fusion hybrid model using Transformer Encoder on pooled audio + video features.
Audio features are pooled to match the video frame rate before concatenation.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Embedding # Removed LSTM, Bidirectional. Added Embedding
from tensorflow.keras.layers import Masking, BatchNormalization, LayerNormalization, MultiHeadAttention, Add # Added MHA, Add
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from audio_pooling_generator import AudioPoolingDataGenerator # Import the required generator

# Global variables (adjust as needed)
BATCH_SIZE = 24
EPOCHS = 125
NUM_CLASSES = 6
PATIENCE = 15
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0006 # May need adjustment for Transformers

# Model specific parameters
TRANSFORMER_BLOCKS = 2 # Number of Transformer Encoder blocks
TRANSFORMER_HEADS = 4 # Number of attention heads
TRANSFORMER_KEY_DIM = 64 # Key dimension for attention
TRANSFORMER_FF_DIM = 256 # Feed-forward network hidden dimension
TRANSFORMER_DROPOUT = 0.1
DENSE_UNITS = [256, 128] # Dense units after pooling/Transformer

print("EARLY FUSION (AUDIO POOLING) HYBRID MODEL WITH TRANSFORMER ENCODER") # Updated Title
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

class WarmUpCosineDecayScheduler(Callback):
    """
    Implements warm-up with cosine decay learning rate scheduling.
    """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# Helper Layer for Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        # Simple fixed positional embeddings (sine/cosine could also be used)
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # inputs shape: (batch, sequence_length, feature_dim)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        # Add positional embeddings to input features
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Ensure mask is preserved
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

# Helper Function/Class for Transformer Encoder Block
def transformer_encoder_block(inputs, num_heads, key_dim, ff_dim, dropout_rate, l2_reg, max_norm):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate,
        kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm)
    )(x, x) # Self-attention
    x = Add()([inputs, attention_output]) # Residual connection

    # Feed Forward Part
    ff_input = x
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(ff_dim, activation="relu", kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(inputs.shape[-1], kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm))(x) # Project back to input dim
    x = Add()([ff_input, x]) # Residual connection
    return x

def create_audio_pooling_transformer_encoder_model(combined_feature_dim, max_seq_length=200): # Renamed function, added max_seq_length
    """
    Create an early-fusion Transformer model that processes combined (pooled audio + video) features.

    Args:
        combined_feature_dim: Dimensionality of the combined features (audio_dim + video_dim)
        max_seq_length: Estimated maximum sequence length for positional embedding.

    Returns:
        Compiled Keras model
    """
    print("Creating early-fusion Transformer Encoder model (Audio Pooling):") # Updated description
    print("- Combined feature dimension:", combined_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Transformer Blocks: {TRANSFORMER_BLOCKS}")
    print(f"- Transformer Heads: {TRANSFORMER_HEADS}")
    print(f"- Transformer Key Dim: {TRANSFORMER_KEY_DIM}")
    print(f"- Transformer FF Dim: {TRANSFORMER_FF_DIM}")
    print(f"- Transformer Dropout: {TRANSFORMER_DROPOUT}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    # --- Combined Input ---
    combined_input = Input(shape=(None, combined_feature_dim), name='combined_input')
    masked_input = Masking(mask_value=0.0)(combined_input)

    # --- Positional Embedding ---
    # Note: combined_feature_dim must match embedding output_dim
    x = PositionalEmbedding(sequence_length=max_seq_length, output_dim=combined_feature_dim)(masked_input)

    # --- Transformer Encoder Stack ---
    for _ in range(TRANSFORMER_BLOCKS):
        x = transformer_encoder_block(
            x, TRANSFORMER_HEADS, TRANSFORMER_KEY_DIM, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT,
            L2_REGULARIZATION, MAX_NORM_CONSTRAINT
        )

    # --- Pooling ---
    # Pool the sequence output from the Transformer stack
    x = GlobalMaxPooling1D()(x)

    # --- Classification Head ---
    x = Dense(
        DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(
        DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(
        NUM_CLASSES, activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)

    # Create model
    model = Model(inputs=combined_input, outputs=output)

    # Optimizer
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- Data Processing (Copied from train_audio_pooling_lstm.py) ---
def process_data(path_pattern, max_files=None):
    """Load and process NPZ files containing features."""
    files = glob.glob(path_pattern)
    if max_files:
        files = files[:max_files]

    if not files:
        print("Error: No files found matching pattern: %s" % path_pattern)
        return None, None, None

    print("Processing %d files from %s" % (len(files), path_pattern))

    audio_data = []
    video_data = []
    labels = []

    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    skipped = 0
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            if '-' in filename:
                parts = filename.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None) if len(parts) >= 3 else None
            else:
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None

            if emotion_code not in emotion_map:
                skipped += 1
                continue

            data = np.load(file_path)
            if 'audio_features' not in data or 'video_features' not in data:
                skipped += 1
                continue

            audio_features = data['audio_features']
            video_features = data['video_features']

            # Basic check for minimal length
            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue

            audio_data.append(audio_features)
            video_data.append(video_features)
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)

        except Exception as e:
            print("Error processing file %s: %s" % (file_path, str(e)))
            skipped += 1

    print("Processed %d files, skipped %d files" % (len(audio_data), skipped))
    return audio_data, video_data, np.array(labels)

def normalize_features(features_list, mean=None, std=None):
    """Normalize a list of variable-length feature arrays."""
    if not features_list: # Handle empty list case
        return [], None, None

    # Filter out potentially empty arrays before stacking
    valid_features = [feat for feat in features_list if feat.shape[0] > 0]
    if not valid_features:
         return features_list, None, None # Return original list if all were empty

    if mean is None or std is None:
        all_features = np.vstack(valid_features)
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)

    std = np.where(std == 0, 1.0, std) # Avoid division by zero

    # Normalize only non-empty arrays
    normalized_list = []
    for features in features_list:
        if features.shape[0] > 0:
            normalized_list.append((features - mean) / std)
        else:
            normalized_list.append(features) # Keep empty arrays as they are

    return normalized_list, mean, std

# --- Training Function ---
def train_model():
    """Main function to train the audio pooling Transformer Encoder model."""
    print("Starting Audio Pooling Transformer Encoder model training...") # Updated print
    print(f"- Using learning rate: {LEARNING_RATE}")

    # Load data (same as before)
    ravdess_pattern = "ravdess_features_facenet/*/*.npz"
    ravdess_audio, ravdess_video, ravdess_labels = process_data(ravdess_pattern)
    cremad_pattern = "crema_d_features_facenet/*.npz"
    cremad_audio, cremad_video, cremad_labels = process_data(cremad_pattern)

    if ravdess_audio is None and cremad_audio is None:
        print("Error: Failed to load any data")
        return

    all_audio = []
    all_video = []
    all_labels = None
    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        all_labels = ravdess_labels
    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels is None: all_labels = cremad_labels
        else: all_labels = np.vstack([all_labels, cremad_labels])

    print(f"Combined: {len(all_audio)} total samples")

    # Split data (same as before)
    class_indices = [np.where(all_labels[:, i] == 1)[0] for i in range(NUM_CLASSES)]
    train_idx, val_idx = [], []
    np.random.seed(RANDOM_SEED)
    for indices in class_indices:
        np.random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        train_idx.extend(indices[:split_idx])
        val_idx.extend(indices[split_idx:])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_audio_raw = [all_audio[i] for i in train_idx]
    train_video_raw = [all_video[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_audio_raw = [all_audio[i] for i in val_idx]
    val_video_raw = [all_video[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("Train/Val split:")
    print("- Train samples:", len(train_audio_raw))
    print("- Validation samples:", len(val_audio_raw))

    # Normalize features BEFORE they go into the generator
    print("Normalizing features (training statistics only)...")
    train_audio_norm, audio_mean, audio_std = normalize_features(train_audio_raw)
    val_audio_norm, _, _ = normalize_features(val_audio_raw, mean=audio_mean, std=audio_std)
    train_video_norm, video_mean, video_std = normalize_features(train_video_raw)
    val_video_norm, _, _ = normalize_features(val_video_raw, mean=video_mean, std=video_std)
    print("Normalization complete.")

    # Get combined dimension for model creation
    if not train_audio_norm or not train_video_norm:
         print("Error: No valid training data after normalization.")
         return
    audio_feature_dim = train_audio_norm[0].shape[1] if train_audio_norm[0].shape[0] > 0 else 0
    video_feature_dim = train_video_norm[0].shape[1] if train_video_norm[0].shape[0] > 0 else 0
    if audio_feature_dim == 0 or video_feature_dim == 0:
        print("Error: Could not determine feature dimensions.")
        return
    combined_feature_dim = audio_feature_dim + video_feature_dim

    # Create data generators using NORMALIZED data
    # Need to determine max_seq_length for positional embedding
    # Let's estimate from the training video data (AudioPoolingGenerator aligns to video length)
    max_seq_length = 0
    for v in train_video_norm:
        max_seq_length = max(max_seq_length, v.shape[0])
    for v in val_video_norm:
        max_seq_length = max(max_seq_length, v.shape[0])
    print(f"Estimated max sequence length for Positional Embedding: {max_seq_length}")
    # Add some buffer
    max_seq_length += 10

    print("Creating data generators...")
    train_generator = AudioPoolingDataGenerator(
        train_video_norm, train_audio_norm, train_labels, batch_size=BATCH_SIZE, shuffle=True, max_seq_len=max_seq_length
    )
    # Use the same generator for validation, just without shuffling
    val_generator = AudioPoolingDataGenerator(
        val_video_norm, val_audio_norm, val_labels, batch_size=BATCH_SIZE, shuffle=False, max_seq_len=max_seq_length
    )

    # Create the NEW model
    model = create_audio_pooling_transformer_encoder_model(combined_feature_dim, max_seq_length) # Updated function call
    model.summary()

    # Define callbacks
    checkpoint_filename = 'best_model_audio_pooling_transformer_encoder.keras' # New checkpoint name
    checkpoint = ModelCheckpoint(
        checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min'
    )
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6
    )

    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler], verbose=1
    )
    end_time = time.time()
    print("Training finished in %.2f seconds" % (end_time - start_time))

    print("Evaluating the best model...")
    if os.path.exists(checkpoint_filename):
        print(f"Loading best weights from {checkpoint_filename}")
        model.load_weights(checkpoint_filename)
    else:
        print("Warning: Checkpoint file not found. Evaluating model with final weights.")

    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy ({checkpoint_filename}): {accuracy:.4f}")

if __name__ == '__main__':
    train_model()
