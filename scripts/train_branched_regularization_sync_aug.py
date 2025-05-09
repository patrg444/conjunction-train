#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script that combines three successful approaches:
1. Synchronized data augmentation - maintains temporal alignment between audio/video
2. L2 regularization - improves generalization of dense layers
3. TCN (Temporal Convolutional Network) - replaces BiLSTM in video branch for better temporal modeling

This implementation aims to combine all three improvements to potentially reach higher
validation accuracy by leveraging the advantages of TCN over BiLSTM for video data.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import time
import glob
import random
from synchronized_data_generator import SynchronizedAugmentationDataGenerator
from sequence_data_generator import ValidationDataGenerator

# Global variables
BATCH_SIZE = 24
EPOCHS = 50  # More epochs for better convergence
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 10  # Increased patience for better convergence
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility
AUGMENTATION_FACTOR = 2.0  # Double the effective dataset size through augmentation
L2_REGULARIZATION = 0.001  # Same regularization strength as in branched_regularization.py
LEARNING_RATE = 0.001  # Higher learning rate from branched_optimizer.py (instead of 0.0005)

print("IMPROVED TRAINING SCRIPT WITH SYNCHRONIZED AUGMENTATION, L2 REGULARIZATION, TCN, AND HIGHER LEARNING RATE")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2, l2_reg=0.001):
    """
    Creates a TCN (Temporal Convolutional Network) block with residual connection and L2 regularization.
    
    Args:
        x: Input tensor
        filters: Number of filters in the convolutions
        kernel_size: Size of the convolutional kernel
        dilation_rate: Dilation rate for the causal convolution
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Output tensor after applying the TCN block
    """
    # First dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',  # Important for causality - only looking at past
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Second dilated convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # Residual connection
    if x.shape[-1] != filters:
        # If dimensions don't match, use 1x1 conv to adapt dimensions
        x = Conv1D(filters=filters, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))(x)
    
    # Add residual connection
    result = Add()([conv2, x])
    return Activation('relu')(result)

def create_enhanced_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim):
    """
    Create an enhanced branched model with masking layers, L2 regularization, and TCN for video branch.
    
    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features
        
    Returns:
        Compiled Keras model
    """
    print("Creating enhanced branched model with masking layers, L2 regularization, and TCN for video:")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    
    # Audio branch with masking
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    
    # Add masking layer to handle padding
    audio_masked = Masking(mask_value=0.0)(audio_input)
    
    # Apply 1D convolutions to extract local patterns
    audio_x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    audio_x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    # Apply bidirectional LSTM for temporal features
    audio_x = Bidirectional(LSTM(128, return_sequences=True))(audio_x)
    audio_x = Dropout(0.3)(audio_x)
    audio_x = Bidirectional(LSTM(64))(audio_x)
    # Add L2 regularization to Dense layer
    audio_x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(audio_x)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch with masking - replaced BiLSTM with TCN
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # Apply TCN blocks with increasing dilation rates for wide receptive field
    # First layer
    video_x = residual_tcn_block(video_masked, filters=128, kernel_size=3, dilation_rate=1, l2_reg=L2_REGULARIZATION)
    # Second layer with increased dilation
    video_x = residual_tcn_block(video_x, filters=128, kernel_size=3, dilation_rate=2, l2_reg=L2_REGULARIZATION)
    # Third layer with further increased dilation
    video_x = residual_tcn_block(video_x, filters=128, kernel_size=3, dilation_rate=4, l2_reg=L2_REGULARIZATION)
    # Fourth layer with even wider context
    video_x = residual_tcn_block(video_x, filters=128, kernel_size=3, dilation_rate=8, l2_reg=L2_REGULARIZATION)
    
    # Global pooling to get fixed-size representation regardless of sequence length
    video_x = tf.keras.layers.GlobalAveragePooling1D()(video_x)
    video_x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(video_x)
    video_x = Dropout(0.4)(video_x)
    
    # Merge branches with more sophisticated fusion
    merged = Concatenate()([audio_x, video_x])
    # Add L2 regularization to all Dense layers in the merged branch
    merged = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer with L2 regularization
    output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(L2_REGULARIZATION))(merged)
    
    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)
    
    # Compile model with increased learning rate (0.001 instead of 0.0005)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
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
    """Main function to train the model with synchronized data augmentation, L2 regularization, TCN, and higher learning rate."""
    print("Starting enhanced model training with SYNCHRONIZED DATA AUGMENTATION, L2 REGULARIZATION, TCN, AND HIGHER LEARNING RATE...")
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
    video_feature_dim = train_video[0].shape[1]
    
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
    )
    
    # Create and compile the enhanced model with regularization, TCN, and masking layers
    model = create_enhanced_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim)
    model.summary()
    
    # Create output directories if they don't exist
    model_dir = "models/branched_regularization_sync_aug_tcn"
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
    
    # Calculate class weights to handle imbalance
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)
    
    print("Using class weights to handle imbalance:", class_weights)
    
    # Train the model
    print("Starting training with synchronized data augmentation, L2 regularization, and TCN...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
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
