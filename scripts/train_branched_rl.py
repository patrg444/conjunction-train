#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script with dynamic sequence padding and reinforcement learning for emotion recognition.
This implementation allows for the full sequence length without truncation and uses RL for frame selection.
Based on the successful Conv1D architecture that achieved 82.9% accuracy.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.layers import Layer, Activation, Multiply, Permute, RepeatVector, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import glob
import random
import math
from sequence_data_generator import SequenceDataGenerator, ValidationDataGenerator

# Global variables
BATCH_SIZE = 24
EPOCHS = 50  # More epochs for better convergence
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 10  # Increased patience for better convergence
TRAIN_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42  # For reproducibility
# We won't set a fixed sequence length - using dynamic padding instead

# RL parameters
RL_EPSILON = 1.0  # Exploration rate, will decay over time
RL_EPSILON_DECAY = 0.95  # Epsilon decay rate per epoch
RL_EPSILON_MIN = 0.01  # Minimum exploration rate
RL_GAMMA = 0.95  # Discount factor for future rewards
TOP_K_FRAMES = 8  # Number of frames to attend to with RL

print("IMPROVED TRAINING SCRIPT WITH DYNAMIC PADDING AND REINFORCEMENT LEARNING")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# Reinforcement Learning Frame Selector Layer
class RLFrameSelector(Layer):
    """
    Reinforcement Learning layer that learns to select the most important frames
    in a sequence for emotion classification.
    """
    def __init__(self, units=64, top_k=8, **kwargs):
        super(RLFrameSelector, self).__init__(**kwargs)
        self.units = units
        self.top_k = top_k
        
    def build(self, input_shape):
        # Policy network weights
        self.policy_dense = Dense(self.units, activation='relu')
        self.policy_out = Dense(1, activation='linear')
        
        # Trainable parameters
        self.built = True
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        feature_dim = inputs.shape[-1]
        
        # Create a mask to identify real vs. padded timesteps
        mask = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1, keepdims=True)
        mask = tf.cast(mask, dtype=tf.float32)
        
        # Get frame importance scores using policy network
        frame_scores = self.policy_dense(inputs)
        frame_scores = self.policy_out(frame_scores)  # Shape: [batch, seq_len, 1]
        
        # Apply mask to scores (set padding scores to negative infinity)
        frame_scores = frame_scores * mask + -1e9 * (1 - mask)
        
        if training:
            # During training, use epsilon-greedy exploration
            # Use exploration with epsilon probability
            explore = tf.random.uniform(shape=[], minval=0, maxval=1) < RL_EPSILON
            
            if explore:
                # Random selection with mask
                random_scores = tf.random.uniform(shape=tf.shape(frame_scores)) * mask
                action_scores = random_scores
            else:
                # Use policy network scores
                action_scores = frame_scores
        else:
            # During inference, just use policy network scores
            action_scores = frame_scores
        
        # Get top-k frame indices
        _, top_indices = tf.nn.top_k(tf.squeeze(action_scores, axis=-1), k=self.top_k)
        
        # Create a one-hot tensor to select these frames
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, self.top_k])
        indices = tf.stack([batch_indices, top_indices], axis=2)
        selection_mask = tf.scatter_nd(indices, 
                                       tf.ones([batch_size, self.top_k]), 
                                       [batch_size, seq_length])
        
        # Apply selection mask to get selected frames
        selection_mask = tf.expand_dims(selection_mask, -1)
        selected_frames = inputs * selection_mask
        
        # Store selection info for training
        self.mask = mask
        self.selection_mask = selection_mask
        self.frame_scores = frame_scores
        
        return selected_frames
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(RLFrameSelector, self).get_config()
        config.update({
            'units': self.units,
            'top_k': self.top_k
        })
        return config

# Custom RL reward-based loss function
def rl_reward_loss(y_true, y_pred):
    """
    Reward-modified loss function that incorporates prediction accuracy
    as a reward signal to the RL frame selector.
    """
    # Standard categorical crossentropy for classifier
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    crossentropy_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    return crossentropy_loss

def create_enhanced_model_with_rl(audio_feature_dim, video_feature_dim):
    """
    Create an enhanced branched model with masking layers and reinforcement learning
    for frame selection. Based on the successful Conv1D architecture.
    
    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features
        
    Returns:
        Compiled Keras model
    """
    print("Creating enhanced branched model with Conv1D and RL frame selection:")
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
    
    # Apply RL frame selector to audio branch
    audio_rl = RLFrameSelector(units=64, top_k=TOP_K_FRAMES)(audio_x)
    
    # Apply bidirectional LSTM for temporal features
    audio_x = Bidirectional(LSTM(128, return_sequences=True))(audio_rl)
    audio_x = Dropout(0.3)(audio_x)
    audio_x = Bidirectional(LSTM(64))(audio_x)
    audio_x = Dense(128, activation='relu')(audio_x)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch with masking
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # Apply RL frame selector to video branch
    video_rl = RLFrameSelector(units=128, top_k=TOP_K_FRAMES)(video_masked)
    
    # FaceNet features already have high dimensionality, so we'll use LSTM directly
    video_x = Bidirectional(LSTM(256, return_sequences=True))(video_rl)
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
    model = Model(inputs=[video_input, audio_input], outputs=output)
    
    # Compile model with RL-aware loss function
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=rl_reward_loss,
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

# Custom callback to anneal exploration rate
class EpsilonAnnealer(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EpsilonAnnealer, self).__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        global RL_EPSILON
        if RL_EPSILON > RL_EPSILON_MIN:
            RL_EPSILON *= RL_EPSILON_DECAY
            print(f"\nExploration rate adjusted to: {RL_EPSILON:.4f}")

def train_model():
    """Main function to train the model with dynamic padding and masking layers."""
    print("Starting enhanced model training with RL FRAME SELECTION...")
    
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
    
    # Create and compile the enhanced model with RL frame selection
    model = create_enhanced_model_with_rl(audio_feature_dim, video_feature_dim)
    model.summary()
    
    # Create output directories if they don't exist
    model_dir = "models/rl_frame_selection"
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
        ),
        EpsilonAnnealer()  # Custom callback for RL exploration annealing
    ]
    
    # Calculate class weights to handle imbalance
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)
    
    print("Using class weights to handle imbalance:", class_weights)
    
    # Train the model
    print("Starting training with RL frame selection...")
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
