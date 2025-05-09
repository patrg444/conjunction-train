#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced hybrid model with comprehensive optimizations to address the 84.77% accuracy plateau:
1. Enhanced regularization - balanced L2, spatial dropout, weight constraints
2. Attention mechanism integration - self-attention layers to complement TCN
3. Skip connections - improved gradient flow throughout network
4. Advanced learning rate schedule - warm-up and cyclical adjustments
5. Adjusted optimization - compatibility fix for TensorFlow versions

This implementation fixes compatibility issues and adjusts hyperparameters
to push beyond the current 83.8% accuracy barrier.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.layers import Activation, Add, SpatialDropout1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from synchronized_data_generator import SynchronizedAugmentationDataGenerator
from sequence_data_generator import ValidationDataGenerator

# Global variables
BATCH_SIZE = 24
EPOCHS = 125  # Extended to 125 for better convergence with warm-up
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 15    # Increased patience for learning rate cycles
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
AUGMENTATION_FACTOR = 2.5  # Increased from 2.0 for more diverse augmentation
L2_REGULARIZATION = 0.002  # Adjusted from 0.0025 to reduce overfitting
MAX_NORM_CONSTRAINT = 3.0  # Weight constraint value
LEARNING_RATE = 0.0006     # Reduced from 0.0008 for better convergence
WEIGHT_DECAY = 0.0005      # Reduced weight decay factor

# Attention mechanism parameters
NUM_HEADS = 4
KEY_DIM = 64
ATTENTION_DROPOUT = 0.2

# Parameters for optimized model capacity - more balanced scaling
AUDIO_CONV_FILTERS = [64, 128]  # Reduced from [80, 160] to avoid overparameterization
AUDIO_LSTM_UNITS = [128, 64]    # Reduced from [160, 80] for better efficiency
VIDEO_TCN_FILTERS = 128         # Reduced from 160 to avoid overfitting
VIDEO_TCN_BLOCKS = 4            # Keep 4 blocks for focused temporal modeling
MERGED_DENSE_UNITS = [256, 128] # Reduced from [320, 160] for balanced fusion

print("ADVANCED HYBRID MODEL WITH TCN-ATTENTION ARCHITECTURE, ENHANCED REGULARIZATION, AND ADVANCED LEARNING SCHEDULE")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

class WarmUpCosineDecayScheduler(Callback):
    """
    Implements warm-up with cosine decay learning rate scheduling.

    Args:
        learning_rate_base: Base learning rate after warm-up
        total_epochs: Total number of epochs for training
        warmup_epochs: Number of epochs for linear warm-up
        min_learning_rate: Minimum learning rate at end of cosine decay
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

        # Warm-up phase
        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay after warm-up
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

def attention_block(x, num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=ATTENTION_DROPOUT, l2_reg=L2_REGULARIZATION):
    """
    Creates a multi-head self-attention block with residual connection and layer normalization.

    Args:
        x: Input tensor
        num_heads: Number of attention heads
        key_dim: Dimension of each attention head
        dropout: Dropout rate for attention
        l2_reg: L2 regularization factor

    Returns:
        Output tensor after applying attention
    """
    # Save input for residual connection
    input_tensor = x

    # Layer normalization before attention (Pre-LN pattern)
    x = LayerNormalization()(x)

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(x, x)

    # Residual connection
    x = Add()([attention_output, input_tensor])

    # Second layer normalization
    y = LayerNormalization()(x)

    # Feedforward network
    y = Dense(
        x.shape[-1] * 2,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(y)
    y = Dropout(dropout)(y)
    y = Dense(
        x.shape[-1],
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(y)

    # Second residual connection
    return Add()([x, y])

def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.3, spatial_dropout_rate=0.2, l2_reg=L2_REGULARIZATION):
    """
    Creates an enhanced TCN block with residual connection, spatial dropout, and balanced regularization.

    Args:
        x: Input tensor
        filters: Number of filters in the convolutions
        kernel_size: Size of the convolutional kernel
        dilation_rate: Dilation rate for the causal convolution
        dropout_rate: Dropout rate for regularization
        spatial_dropout_rate: Spatial dropout rate for convolutional layers
        l2_reg: L2 regularization factor

    Returns:
        Output tensor after applying the TCN block
    """
    # Save input for skip connection
    input_tensor = x

    # Apply Layer Normalization before convolutions (Pre-LN pattern)
    x = LayerNormalization()(x)

    # First dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',  # Important for causality - only looking at past
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(x)
    conv1 = BatchNormalization()(conv1)
    # Use spatial dropout which is better for convolutional layers
    conv1 = SpatialDropout1D(spatial_dropout_rate)(conv1)

    # Second dilated convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SpatialDropout1D(spatial_dropout_rate)(conv2)

    # Residual connection
    if input_tensor.shape[-1] != filters:
        # If dimensions don't match, use 1x1 conv to adapt dimensions
        input_tensor = Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=l2(l2_reg),
            kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
        )(input_tensor)

    # Add residual connection
    result = Add()([conv2, input_tensor])
    return Activation('relu')(result)

def create_enhanced_large_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim):
    """
    Create an advanced hybrid model with attention, TCN, skip connections, and
    balanced regularization for improved performance.

    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features

    Returns:
        Compiled Keras model
    """
    print("Creating advanced hybrid model with attention, TCN, skip connections, and balanced regularization:")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Attention heads: {NUM_HEADS}")
    print(f"- Audio Conv Filters: {AUDIO_CONV_FILTERS}")
    print(f"- Audio LSTM Units: {AUDIO_LSTM_UNITS}")
    print(f"- Video TCN Filters: {VIDEO_TCN_FILTERS}")
    print(f"- Video TCN Blocks: {VIDEO_TCN_BLOCKS}")
    print(f"- Merged Dense Units: {MERGED_DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")
    print(f"- Weight decay: {WEIGHT_DECAY}")

    # Audio branch with masking and skip connections
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')

    # Add masking layer to handle padding
    audio_masked = Masking(mask_value=0.0)(audio_input)

    # First conv layer with regularization
    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)  # Spatial dropout for conv layers
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    # Skip connection around second conv layer
    audio_skip = Conv1D(
        AUDIO_CONV_FILTERS[1],
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(audio_x)

    # Second conv layer
    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)

    # Add skip connection and activate
    audio_x = Add()([audio_x, audio_skip])
    audio_x = Activation('relu')(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    # Add attention block to audio branch
    audio_x = attention_block(audio_x)

    # Apply bidirectional LSTM with recurrent dropout and weight constraints
    audio_x = Bidirectional(LSTM(
        AUDIO_LSTM_UNITS[0],
        return_sequences=True,
        dropout=0.3,  # Input dropout
        recurrent_dropout=0.2,  # Recurrent state dropout
        kernel_regularizer=l2(L2_REGULARIZATION),
        recurrent_regularizer=l2(L2_REGULARIZATION/2),  # Lighter regularization for recurrent weights
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT),
        recurrent_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    ))(audio_x)
    audio_x = Dropout(0.3)(audio_x)

    audio_x = Bidirectional(LSTM(
        AUDIO_LSTM_UNITS[1],
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION),
        recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT),
        recurrent_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    ))(audio_x)

    # Add L2 regularization to Dense layer with constraint
    audio_x = Dense(
        AUDIO_LSTM_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = LayerNormalization()(audio_x)  # Add normalization
    audio_x = Dropout(0.4)(audio_x)

    # Video branch with masking, TCN and attention
    video_input = Input(shape=(None, video_feature_dim), name='video_input')

    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)

    # Initial projection layer with weight constraints
    video_x = Conv1D(
        VIDEO_TCN_FILTERS,
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(video_masked)

    # Apply TCN blocks with increasing dilation rates for wider receptive field
    # First layer
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=1)

    # Add attention after first TCN block
    video_x = attention_block(video_x)

    # Second layer with increased dilation
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=2)

    # Third layer with further increased dilation
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=4)

    # Add attention after third TCN block
    video_x = attention_block(video_x)

    # Fourth layer with even wider context
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=8)

    # Use both average and max pooling and concatenate for better feature extraction
    video_avg_pool = GlobalAveragePooling1D()(video_x)
    video_max_pool = GlobalMaxPooling1D()(video_x)
    video_x = Concatenate()([video_avg_pool, video_max_pool])

    # Dimension reduction with balanced regularization
    video_x = Dense(
        MERGED_DENSE_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(video_x)
    video_x = LayerNormalization()(video_x)  # Add normalization
    video_x = Dropout(0.4)(video_x)

    # More sophisticated fusion with skip connections
    audio_projection = Dense(
        MERGED_DENSE_UNITS[0]//2,
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(audio_x)

    video_projection = Dense(
        MERGED_DENSE_UNITS[0]//2,
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(video_x)

    # Concatenate modalities
    merged = Concatenate()([audio_projection, video_projection])

    # First merged dense layer with skip connection
    merged_skip = Dense(
        MERGED_DENSE_UNITS[0],
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(merged)

    merged = Dense(
        MERGED_DENSE_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.5)(merged)

    # Add skip connection
    merged = Add()([merged, merged_skip])
    merged = Activation('relu')(merged)

    # Second merged dense layer
    merged = Dense(
        MERGED_DENSE_UNITS[1],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.4)(merged)

    # Output layer with L2 regularization and constraint
    output = Dense(
        NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)

    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)

    # Use standard Adam optimizer - TF version compatibility fix
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compile model with optimizer
    model.compile(
        optimizer=optimizer,
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
        '02': 'NEU',  # calm (mapped to neutral)
        '03': 'HAP',  # happy
        '04': 'SAD',  # sad
        '05': 'ANG',  # angry
        '06': 'FEA',  # fearful
        '07': 'DIS',  # disgust
        # '08' (surprised) is intentionally excluded to skip these samples
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

            # Skip sequences that are too short - minimum viable length
            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue

            # Append to lists - keep original length
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

def normalize_features(features_list, mean=None, std=None):
    """
    Normalize a list of variable-length feature arrays.

    Args:
        features_list: List of numpy arrays with variable lengths
        mean: Optional pre-computed mean to use (to prevent data leakage)
        std: Optional pre-computed standard deviation to use

    Returns:
        List of normalized numpy arrays, and optionally the mean and std if not provided
    """
    if mean is None or std is None:
        # Calculate statistics from this set
        # First, concatenate all features
        all_features = np.vstack([feat for feat in features_list])

        # Calculate mean and std from the concatenated data
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    # Normalize each sequence individually using the provided stats
    normalized_list = []
    for features in features_list:
        normalized = (features - mean) / std
        normalized_list.append(normalized)

    return normalized_list, mean, std

def train_model():
    """Main function to train the model with synchronized data augmentation, balanced regularization, TCN, and adjusted learning rate."""
    print("Starting optimized balanced model training with SYNCHRONIZED DATA AUGMENTATION, BALANCED REGULARIZATION, FOCUSED TCN, AND ADJUSTED LEARNING RATE...")
    print(f"- Using learning rate: {LEARNING_RATE}")

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

    # Analyze dataset composition - per emotion
    def analyze_dataset_composition(labels, name="Dataset"):
        """Count samples per emotion class"""
        emotion_names = ['Anger (ANG)', 'Disgust (DIS)', 'Fear (FEA)', 'Happy (HAP)', 'Neutral (NEU)', 'Sad (SAD)']
        counts = np.sum(labels, axis=0).astype(int)

        print(f"\n{name} Composition:")
        for i, count in enumerate(counts):
            print(f"- {emotion_names[i]}: {count} samples")
        print(f"Total {name}: {len(labels)} samples")

        return counts

    # Combine available datasets
    all_audio = []
    all_video = []
    all_labels = None  # Initialize as None

    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        all_labels = ravdess_labels  # Just assign directly the first time
        print(f"Added RAVDESS: {len(ravdess_audio)} samples")
        analyze_dataset_composition(ravdess_labels, "RAVDESS Dataset")

    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels is None:
            all_labels = cremad_labels
        else:
            all_labels = np.vstack([all_labels, cremad_labels])
        print(f"Added CREMA-D: {len(cremad_audio)} samples")
        analyze_dataset_composition(cremad_labels, "CREMA-D Dataset")

    print(f"Combined: {len(all_audio)} total samples")
    if all_labels is not None:
        analyze_dataset_composition(all_labels, "Combined Dataset")

    print("Dataset size with synchronized augmentation:")
    print("- Number of samples:", len(all_audio))
    print("- Augmentation factor:", AUGMENTATION_FACTOR)
    print("- Effective dataset size:", len(all_audio) * AUGMENTATION_FACTOR)
    print("- Label distribution:")
    for i in range(NUM_CLASSES):
        count = np.sum(all_labels[:, i])
        print("  Class %d: %d samples (%.1f%%)" % (i, count, count/len(all_labels)*100))

    # Analyze sequence lengths
    audio_lengths = [len(seq) for seq in all_audio]
    video_lengths = [len(seq) for seq in all_video]

    print("Sequence length statistics:")
    print("- Audio: min=%d, max=%d, mean=%.1f, median=%d" % (
        min(audio_lengths), max(audio_lengths),
        np.mean(audio_lengths), np.median(audio_lengths)
    ))
    print("- Video: min=%d, max=%d, mean=%.1f, median=%d" % (
        min(video_lengths), max(video_lengths),
        np.mean(video_lengths), np.median(video_lengths)
    ))

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

    # Split the data using the indices
    train_audio = [all_audio[i] for i in train_idx]
    train_video = [all_video[i] for i in train_idx]
    train_labels = all_labels[train_idx]

    val_audio = [all_audio[i] for i in val_idx]
    val_video = [all_video[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("Train/Val split with stratification:")
    print("- Train samples:", len(train_audio))
    print("- Validation samples:", len(val_audio))

    # Normalize features for better training - FIXED APPROACH TO PREVENT DATA LEAKAGE
    print("Normalizing features (training statistics only)...")
    # Calculate normalization statistics on training data only
    train_audio, audio_mean, audio_std = normalize_features(train_audio)
    # Apply the same statistics to validation data
    val_audio, _, _ = normalize_features(val_audio, mean=audio_mean, std=audio_std)
    print("Normalization complete - leakage prevented!")

    # Note: FaceNet features are already normalized, so we skip that for video

    # Get the feature dimensionality (without sequence length)
    audio_feature_dim = train_audio[0].shape[1]
    video_feature_dim = train_video[0].shape[1]  # Fixed from shape to shape[1]

    # Create the custom data generators with synchronized augmentation
    print("Creating data generators with synchronized augmentation...")
    train_generator = SynchronizedAugmentationDataGenerator(
        train_video, train_audio, train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augmentation_factor=AUGMENTATION_FACTOR,
        augmentation_probability=0.8  # 80% chance of applying augmentation to eligible samples
    )

    val_generator = ValidationDataGenerator(
        val_video, val_audio, val_labels,
        batch_size=BATCH_SIZE
        # shuffle=False # Removed invalid argument
    )

    # Create the model
    model = create_enhanced_large_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim)
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint(
        'best_model_tcn_large_fixed_v2.keras', # Adjusted checkpoint name
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE * 2,  # Double patience for early stopping
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,  # Less aggressive reduction
        patience=PATIENCE,
        min_lr=5e-6,
        verbose=1,
        mode='min'
    )
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE,
        total_epochs=EPOCHS,
        warmup_epochs=10,
        min_learning_rate=5e-6
    )

    # Train the model
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )
    end_time = time.time()
    print("Training finished in %.2f seconds" % (end_time - start_time))

    # Evaluate the best model
    print("Evaluating the best model...")
    model.load_weights('best_model_tcn_large_fixed_v2.keras') # Adjusted checkpoint name
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print("Best model validation accuracy: %.4f" % accuracy)

if __name__ == '__main__':
    train_model()
