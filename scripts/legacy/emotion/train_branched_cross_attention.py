#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script with dynamic sequence padding for emotion recognition.
This implementation allows for the full sequence length without truncation.
Version without data augmentation and with proper normalization to prevent data leakage.
Modified to add cross-modal attention between audio and video branches.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Layer
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import glob
import random
from sequence_data_generator import SequenceDataGenerator, ValidationDataGenerator

# Global variables
BATCH_SIZE = 24
EPOCHS = 50  # More epochs for better convergence
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 10  # Increased patience for better convergence
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility
# We won't set a fixed sequence length - using dynamic padding instead

print("IMPROVED TRAINING SCRIPT WITH DYNAMIC PADDING AND CROSS-MODAL ATTENTION")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# Cross-Modal Attention Layer
class CrossModalAttention(Layer):
    """
    Attention mechanism that allows one modality to attend to another.
    This helps the model focus on relevant parts of each modality.
    """
    def __init__(self, attention_dim=64, **kwargs):
        self.attention_dim = attention_dim
        super(CrossModalAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(CrossModalAttention, self).get_config()
        config.update({"attention_dim": self.attention_dim})
        return config
        
    def build(self, input_shape):
        # Input should be a list of two tensors
        assert isinstance(input_shape, list) and len(input_shape) == 2
        
        # Source and target feature dimensions
        source_dim = input_shape[0][-1]
        target_dim = input_shape[1][-1]
        
        # Weights for projecting both inputs to the same attention space
        self.W_source = self.add_weight(
            name="W_source",
            shape=(source_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.W_target = self.add_weight(
            name="W_target",
            shape=(target_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.V = self.add_weight(
            name="V",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        
        super(CrossModalAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Unpack inputs
        source, target = inputs
        
        # Project both inputs to attention space
        source_proj = tf.matmul(source, self.W_source)  # (batch, source_dim) -> (batch, att_dim)
        
        # Handle variable sequence length for target
        # target shape: (batch, seq_len, target_dim)
        target_proj = tf.einsum('bsf,fh->bsh', target, self.W_target)  # (batch, seq_len, att_dim)
        
        # Create compatibility function (additive attention)
        source_expanded = tf.expand_dims(source_proj, 1)  # (batch, 1, att_dim)
        
        # Broadcast source to each timestep of target
        compatibility = tf.tanh(source_expanded + target_proj)  # (batch, seq_len, att_dim)
        
        # Get attention weights
        scores = tf.matmul(compatibility, self.V)  # (batch, seq_len, 1)
        
        # Apply masking for padding
        mask = tf.reduce_any(tf.not_equal(target, 0), axis=-1, keepdims=True)
        mask = tf.cast(mask, dtype=tf.float32)
        scores = scores * mask + -1e9 * (1 - mask)
        
        # Get attention weights through softmax
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch, seq_len, 1)
        
        # Apply mask and re-normalize
        attention_weights = attention_weights * mask
        attention_sum = tf.reduce_sum(attention_weights, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        attention_weights = attention_weights / attention_sum
        
        # Apply attention weights to get context vector
        context = target * attention_weights  # (batch, seq_len, target_dim)
        context = tf.reduce_sum(context, axis=1)  # (batch, target_dim)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][-1])

def create_enhanced_model_with_cross_attention(audio_feature_dim, video_feature_dim):
    """
    Create an enhanced branched model with masking layers and cross-modal attention.
    
    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features
        
    Returns:
        Compiled Keras model
    """
    print("Creating enhanced branched model with masking and cross-modal attention:")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    
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
    
    # Apply bidirectional LSTM for temporal features - keep return_sequences=True for cross-attention
    audio_lstm = Bidirectional(LSTM(128, return_sequences=True))(audio_x)
    audio_lstm = Dropout(0.3)(audio_lstm)
    
    # Final audio representation (non-sequence) for fusion
    audio_final = Bidirectional(LSTM(64))(audio_lstm)
    audio_x = Dense(128, activation='relu')(audio_final)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch with masking
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # FaceNet features already have high dimensionality, so we'll use LSTM directly
    # Keep return_sequences=True for cross-attention
    video_lstm = Bidirectional(LSTM(256, return_sequences=True))(video_masked)
    video_lstm = Dropout(0.3)(video_lstm)
    
    # Final video representation (non-sequence) for fusion
    video_final = Bidirectional(LSTM(128))(video_lstm)
    video_x = Dense(256, activation='relu')(video_final)
    video_x = Dropout(0.4)(video_x)
    
    # Cross-modal attention: Audio attends to video
    audio_attends_video = CrossModalAttention()([audio_final, video_lstm])
    
    # Cross-modal attention: Video attends to audio
    video_attends_audio = CrossModalAttention()([video_final, audio_lstm])
    
    # Merge branches with more sophisticated fusion including cross-modal attention
    merged = Concatenate()([
        audio_x,             # Base audio features
        video_x,             # Base video features
        audio_attends_video, # Audio attending to video features
        video_attends_audio  # Video attending to audio features
    ])
    
    merged = Dense(256, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[video_input, audio_input], outputs=output)
    
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
    """Main function to train the model with dynamic padding and cross-modal attention."""
    print("Starting enhanced model training with DYNAMIC SEQUENCE PADDING AND CROSS-MODAL ATTENTION...")
    
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
    
    # SKIP DATA AUGMENTATION - key change from original script
    print("Skipping data augmentation...")
    
    print("Dataset size (no augmentation):")
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
    
    # Create and compile the enhanced model with cross-modal attention
    model = create_enhanced_model_with_cross_attention(audio_feature_dim, video_feature_dim)
    model.summary()
    
    # Create output directories if they don't exist
    model_dir = "models/branched_cross_attention"
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
    print("Starting training with dynamic sequence padding and cross-modal attention...")
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
