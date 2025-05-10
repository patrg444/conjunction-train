#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script with temporal attention, focal loss, and no audio augmentation.
Building on the successful dynamic padding approach with proper normalization.
Audio augmentation removed to maintain audio-visual alignment.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.layers import Layer, Activation, Multiply, Permute, RepeatVector, Lambda
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import glob
import random
from sequence_data_generator import SequenceDataGenerator, ValidationDataGenerator
from scipy.signal import resample

# Global variables
BATCH_SIZE = 24
EPOCHS = 50  # More epochs for better convergence
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 10  # Increased patience for better convergence
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility
AUGMENTATION_FACTOR = 1  # No augmentation - changed from 2 to 1

print("IMPROVED TRAINING SCRIPT WITH ATTENTION AND FOCAL LOSS (NO AUGMENTATION)")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# Custom Attention Layer
class AttentionLayer(Layer):
    """
    Attention layer for sequence data that learns to focus on important timesteps.
    Enhanced to handle variable-length inputs with proper masking.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Only create weights based on the feature dimension, which is known
        self.W = self.add_weight(
            name="att_weight", 
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform", 
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Compute alignment scores using only the feature dimension weights
        # e shape: (batch_size, time_steps, 1)
        e = tf.matmul(x, self.W)
        
        # Apply tanh activation
        e = tf.tanh(e)
        
        # Create a mask for padding values
        # Check if any feature in a timestep is non-zero, if all are zero it's padding
        mask = tf.reduce_any(tf.not_equal(x, 0), axis=-1, keepdims=True)
        mask = tf.cast(mask, dtype=tf.float32)
        
        # Apply mask to attention scores (set padding scores to negative infinity)
        # This ensures padded timesteps get ~0 attention after softmax
        e = e * mask + -1e9 * (1 - mask)
        
        # Apply softmax to get attention weights (normalize over time dimension)
        a = tf.nn.softmax(e, axis=1)
        
        # The mask is applied again to ensure numerical stability
        a = a * mask
        
        # Re-normalize the weights for non-padded timesteps to sum to 1
        # This step is added for numerical stability
        a_sum = tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        a = a / a_sum
        
        # Compute weighted sum
        # a shape: (batch_size, time_steps, 1)
        # x shape: (batch_size, time_steps, features)
        # context shape: (batch_size, features)
        context = x * a
        context = tf.reduce_sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Implementation of Focal Loss
def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss implementation to focus more on hard examples.
    gamma: focusing parameter
    alpha: optional class weights
    """
    def focal_loss_fn(y_true, y_pred):
        # Clip the prediction to avoid numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Apply focusing parameter
        focal_weight = tf.pow(1 - y_pred, gamma) * y_true
        
        # Apply alpha if provided
        if alpha is not None:
            focal_weight = alpha * focal_weight
        
        loss = focal_weight * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return loss
    
    return focal_loss_fn

def create_enhanced_model_with_attention(audio_feature_dim, video_feature_dim):
    """
    Create an enhanced branched model with proper LSTM masking and attention mechanism.
    This implementation preserves masking through the entire network to handle variable-length
    sequences correctly.
    
    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features
        
    Returns:
        Compiled Keras model
    """
    print("Creating enhanced branched model with LSTM and attention:")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    
    # Audio branch with masking
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    
    # Apply masking to handle variable-length sequences
    audio_masked = Masking(mask_value=0.0)(audio_input)
    
    # Direct LSTM stack that preserves masking information
    audio_lstm1 = Bidirectional(LSTM(128, return_sequences=True, 
                                    dropout=0.2))(audio_masked)
    audio_drop1 = Dropout(0.3)(audio_lstm1)
    
    # Second LSTM layer with masking automatically preserved
    audio_lstm2 = Bidirectional(LSTM(64, return_sequences=True,
                                    dropout=0.2))(audio_drop1)
    
    # Apply attention mechanism to audio branch - attention layer preserves mask
    audio_attention = AttentionLayer()(audio_lstm2)
    audio_features = Dropout(0.4)(audio_attention)
    
    # Video branch with masking
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Apply masking to handle variable-length sequences
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # Direct LSTM stack that preserves masking information
    # FaceNet features already have high dimensionality
    video_lstm1 = Bidirectional(LSTM(256, return_sequences=True,
                                    dropout=0.2))(video_masked)
    video_drop1 = Dropout(0.3)(video_lstm1)
    
    # Second LSTM layer
    video_lstm2 = Bidirectional(LSTM(128, return_sequences=True,
                                    dropout=0.2))(video_drop1)
    
    # Apply attention mechanism to video branch
    video_attention = AttentionLayer()(video_lstm2)
    video_features = Dropout(0.4)(video_attention)
    
    # Merge branches with more sophisticated fusion
    merged = Concatenate()([audio_features, video_features])
    merged = Dense(256, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(merged)
    
    # Create model with dictionary inputs to match data generator
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)
    
    # Compile model with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy']
    )
    
    return model

# Audio augmentation function kept for reference but not used
def augment_audio(audio_features):
    """
    Apply data augmentation to audio features.
    NOTE: This function is kept for reference but not used as augmentation is disabled.
    
    Args:
        audio_features: Original audio features array
        
    Returns:
        Augmented audio features
    """
    aug_type = np.random.choice(['noise', 'pitch', 'tempo', 'all'])
    result = audio_features.copy()
    
    # Add random noise
    if aug_type in ['noise', 'all']:
        noise_level = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_level, result.shape)
        result = result + noise
    
    # Simulate pitch/speaker variation by slightly scaling feature values
    if aug_type in ['pitch', 'all']:
        pitch_factor = np.random.uniform(0.9, 1.1)
        result = result * pitch_factor
    
    # Simulate tempo variation by resampling
    if aug_type in ['tempo', 'all']:
        tempo_factor = np.random.uniform(0.9, 1.1)
        orig_len = len(result)
        new_len = int(orig_len * tempo_factor)
        # Ensure we have at least 10 frames
        new_len = max(10, new_len)
        # Resample the sequence
        temp = np.zeros((new_len, result.shape[1]))
        for i in range(result.shape[1]):
            temp[:, i] = resample(result[:, i], new_len)
        result = temp
    
    return result

def process_data(path_pattern, max_files=None, augment=False):
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
            
            # Skip sequences that are too short - minimum viable length
            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue
            
            # Append original data
            audio_data.append(audio_features)
            video_data.append(video_features)
            
            # Create one-hot encoded label
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)
            
            # Augmentation disabled - kept for reference
            if augment and False:  # Force to always be False
                # This block will never execute
                pass
            
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
    """Main function to train the model with attention, focal loss, and no augmentation."""
    print("Starting enhanced model training with ATTENTION and FOCAL LOSS (no augmentation)...")
    
    # Process RAVDESS data - note the nested pattern to find files in actor folders
    ravdess_pattern = "ravdess_features_facenet/*/*.npz"
    ravdess_audio, ravdess_video, ravdess_labels = process_data(ravdess_pattern, augment=False)
    
    # Process CREMA-D data
    cremad_pattern = "crema_d_features_facenet/*.npz"
    cremad_audio, cremad_video, cremad_labels = process_data(cremad_pattern, augment=False)
    
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
    
    print("Dataset size:")
    print("- Number of samples:", len(all_audio))
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
    
    # Create the custom data generators
    train_generator = SequenceDataGenerator(
        train_video, train_audio, train_labels,
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    val_generator = ValidationDataGenerator(
        val_video, val_audio, val_labels,
        batch_size=BATCH_SIZE
    )
    
    # Create and compile the enhanced model with attention
    model = create_enhanced_model_with_attention(audio_feature_dim, video_feature_dim)
    model.summary()
    
    # Create dataset adapters wrapped around our custom data generators
    # This helps TensorFlow properly understand the expected structure
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            {
                'video_input': tf.TensorSpec(shape=(None, None, video_feature_dim), dtype=tf.float32),
                'audio_input': tf.TensorSpec(shape=(None, None, audio_feature_dim), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    )
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_signature=(
            {
                'video_input': tf.TensorSpec(shape=(None, None, video_feature_dim), dtype=tf.float32),
                'audio_input': tf.TensorSpec(shape=(None, None, audio_feature_dim), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    )
    
    # Create output directories if they don't exist
    model_dir = "models/attention_focal_loss_no_aug"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define callbacks with more sophisticated setup
    checkpoint_path = os.path.join(model_dir, "model_best.keras")
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
    
    # Calculate class weights to handle imbalance - in addition to focal loss
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)
    
    print("Using class weights with focal loss to handle imbalance:", class_weights)
    
    # Train the model
    print("Starting training with attention and focal loss...")
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model.keras")
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
