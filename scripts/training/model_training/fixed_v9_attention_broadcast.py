#!/usr/bin/env python3
"""
WAV2VEC Emotion Recognition Model with Self-Attention - Fixed Broadcasting
Adds attention mechanism to focus on important frames for emotion recognition
"""

import os
import numpy as np
import tensorflow as tf
import glob
import json
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Bidirectional, LSTM, LayerNormalization,
    GlobalAveragePooling1D, Masking
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Fixed Attention layer with built-in broadcasting support
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.use_scale = use_scale
        
    def build(self, input_shape):
        self.query_dense = Dense(input_shape[-1])
        self.key_dense = Dense(input_shape[-1])
        self.value_dense = Dense(input_shape[-1])
        self.scale_factor = 1.0
        if self.use_scale:
            self.scale_factor = tf.math.sqrt(tf.cast(input_shape[-1], tf.float32))
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # Single input version of self-attention
        query = self.query_dense(inputs)  # (batch_size, seq_len, dim)
        key = self.key_dense(inputs)      # (batch_size, seq_len, dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, dim)
        
        # Calculate attention scores
        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        if self.use_scale:
            scores = scores / self.scale_factor
            
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        context = tf.matmul(attention_weights, value)  # (batch_size, seq_len, dim)
        
        return context

class CosineDecayWithWarmup(Callback):
    """
    Combined learning rate scheduler with:
    1. Linear warmup for initial epochs
    2. ReduceLROnPlateau for middle epochs
    3. Cosine decay for final epochs
    """
    def __init__(
        self,
        min_lr=5e-6,
        max_lr=3e-4,
        warmup_epochs=8,
        cosine_decay_start=20,
        total_epochs=100,
        verbose=1
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_decay_start = cosine_decay_start
        self.total_epochs = total_epochs
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        # Start with minimum learning rate
        self.model.optimizer.learning_rate.assign(self.min_lr)
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: CosineDecayWithWarmup setting learning rate to {warmup_lr:.6f} (warmup phase).")
        elif epoch >= self.cosine_decay_start:
            # Cosine decay
            progress = (epoch - self.cosine_decay_start) / (self.total_epochs - self.cosine_decay_start)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            decayed_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            self.model.optimizer.learning_rate.assign(decayed_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: CosineDecayWithWarmup setting learning rate to {decayed_lr:.6f} (cosine decay phase).")
        
        # For middle epochs, rely on ReduceLROnPlateau

def load_wav2vec_features(data_dir, limit=None):
    """
    Load WAV2VEC features from .npz files
    """
    features = []
    labels = []
    file_paths = []
    
    # Get all .npz files
    npz_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
    
    # Apply limit if specified
    if limit and len(npz_files) > limit:
        npz_files = npz_files[:limit]
    
    print(f"Loading {len(npz_files)} WAV2VEC feature files...")
    
    for npz_file in npz_files:
        try:
            # Load features
            data = np.load(npz_file, allow_pickle=True)
            
            # Check if file uses 'features' key (older format) or 'embeddings' key (newer format)
            if 'features' in data:
                feature = data['features']
            elif 'embeddings' in data:
                feature = data['embeddings']
            else:
                # Try accessing arrays directly
                keys = list(data.keys())
                if len(keys) > 0:
                    feature = data[keys[0]]
                else:
                    print(f"Warning: No features found in {npz_file}, skipping...")
                    continue
            
            # Extract label from filename
            parts = os.path.basename(npz_file).split('_')
            label = None
            
            # Common label extraction patterns
            if len(parts) >= 7:
                # Format: XX-XX-XX-XX-XX-XX-XX_...
                # We extract the emotion code from the start sequence
                emotion_code = parts[0].split('-')[2]  # Get the 3rd element (emotion code)
                label = emotion_code
            elif len(parts) >= 3:
                # Alternative format: try to get label from 3rd part
                label = parts[2]
            
            if label is None:
                # Fallback: use directory name
                parent_dir = os.path.basename(os.path.dirname(npz_file))
                label = parent_dir
            
            # Ensure label is a string
            label = str(label)
            
            # Append data
            features.append(feature)
            labels.append(label)
            file_paths.append(npz_file)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    return features, labels, file_paths

def pad_sequences(features, max_length=None):
    """
    Pad the sequences to a fixed length
    """
    if max_length is None:
        # Get sequence lengths
        lengths = [len(f) for f in features]
        
        # Use 98th percentile length to preserve more temporal information
        max_length = int(np.percentile(lengths, 98))
    
    print(f"Padding sequences to length {max_length}")
    
    # Get statistics
    lengths = np.array([len(f) for f in features])
    print(f"Sequence length statistics:")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths)}")
    print(f"  95th percentile: {np.percentile(lengths, 95)}")
    
    # Get feature dimension
    feature_dim = features[0].shape[1]
    
    # Initialize output array
    padded_features = np.zeros((len(features), max_length, feature_dim))
    
    # Fill with actual data (truncate if needed)
    for i, feature in enumerate(features):
        seq_length = min(len(feature), max_length)
        padded_features[i, :seq_length, :] = feature[:seq_length]
    
    return padded_features, max_length

def normalize_features(features, mean=None, std=None):
    """
    Normalize features using mean and standard deviation
    """
    if mean is None or std is None:
        # Flatten all features for normalization
        all_features = np.vstack([f.reshape(-1, f.shape[-1]) for f in features])
        
        # Calculate mean and std
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-10)
    
    # Normalize each feature
    normalized_features = []
    for feature in features:
        normalized_feature = (feature - mean) / std
        normalized_features.append(normalized_feature)
    
    return normalized_features, mean, std

def build_model(input_shape, num_classes):
    """
    Build the model with attention mechanism
    """
    print(f"Building model with {num_classes} output classes...")
    
    # Input layer with masking to handle variable-length sequences
    input_layer = Input(shape=input_shape, name="input_layer")
    masked_input = Masking(mask_value=0.0)(input_layer)
    
    # Bidirectional LSTM with recurrent dropout for regularization
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25))(masked_input)
    
    # Add Layer Normalization for more stable training
    x = LayerNormalization()(x)
    
    # Self-attention mechanism to focus on important frames
    context_vector = AttentionLayer(use_scale=True)(x)
    
    # Global pooling to convert attention output to fixed-length representation
    x = GlobalAveragePooling1D()(context_vector)
    
    # Dense layers with stronger regularization
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Starting with min LR
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    return model

def main():
    # If in smoke test mode, use minimal data for quick testing
    if 'SMOKE_TEST' in os.environ:
        print("RUNNING IN SMOKE TEST MODE - using minimal dataset")
        # Generate synthetic data for testing
        num_samples = 20
        seq_length = 100
        feature_dim = 768
        num_classes = 6
        
        # Create features with varying sequence lengths
        features = []
        labels = []
        for i in range(num_samples):
            # Random sequence length between 10 and 200
            rand_length = np.random.randint(10, 200)
            feature = np.random.randn(rand_length, feature_dim).astype(np.float32)
            label = np.random.randint(0, num_classes)
            features.append(feature)
            labels.append(str(label))
        
        file_paths = [f"dummy_file_{i}.npz" for i in range(num_samples)]
    else:
        # Load real data with path flexibility
        # Try multiple potential paths for data directory
        potential_dirs = [
            "/home/ubuntu/audio_emotion/wav2vec_features",
            "/home/ubuntu/wav2vec_features",
            "/home/ubuntu/audio_emotion/features/wav2vec",
            "/home/ubuntu/features/wav2vec",
            "/data/wav2vec_features"
        ]
        
        for potential_dir in potential_dirs:
            if os.path.exists(potential_dir):
                print(f"Found data directory: {potential_dir}")
                features, labels, file_paths = load_wav2vec_features(potential_dir)
                if len(features) > 0:
                    break
        
        # If no directory worked, try a wider search
        if not 'features' in locals() or len(features) == 0:
            print("Trying wider search for .npz files...")
            npz_files = glob.glob(os.path.join("/home/ubuntu", "**/*.npz"), recursive=True)
            if npz_files:
                data_dir = os.path.dirname(npz_files[0])
                print(f"Found .npz files in {data_dir}")
                features, labels, file_paths = load_wav2vec_features(data_dir)
    
    # Check if we have data
    if len(features) == 0:
        print("Error: No features loaded. Please check the data directory.")
        return
    
    print(f"Loaded {len(features)} samples with {len(set(labels))} unique labels.")
    
    # Split into train and validation with minimum test size
    n_classes = len(set(labels))
    min_test_size = max(0.2, n_classes / len(features))  # At least 20% or enough for one sample per class
    
    print(f"Using test_size of {min_test_size:.2f} to accommodate {n_classes} classes")
    
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=min_test_size, random_state=42, stratify=labels
    )
    
    print(f"Train set: {len(train_features)} samples")
    print(f"Validation set: {len(val_features)} samples")
    
    # Normalize features
    train_features_normalized, mean, std = normalize_features(train_features)
    val_features_normalized, _, _ = normalize_features(val_features, mean, std)
    
    # Save normalization parameters
    np.save("/home/ubuntu/audio_emotion/audio_mean_v9.npy", mean)
    np.save("/home/ubuntu/audio_emotion/audio_std_v9.npy", std)
    
    # Pad sequences
    X_train, max_length = pad_sequences(train_features_normalized)
    X_val, _ = pad_sequences(val_features_normalized, max_length)
    
    print(f"Padded train shape: {X_train.shape}")
    print(f"Padded validation shape: {X_val.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    
    # Save label classes
    np.save("/home/ubuntu/audio_emotion/checkpoints/label_classes_v9.npy", label_encoder.classes_)
    
    # Print class distribution
    for i, label in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        print(f"Class {label}: {train_count} train, {val_count} validation")
    
    # Build model
    input_shape = (max_length, X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    
    # Prepare callbacks
    os.makedirs("/home/ubuntu/audio_emotion/checkpoints", exist_ok=True)
    
    checkpoint_best = ModelCheckpoint(
        "/home/ubuntu/audio_emotion/checkpoints/best_model_v9.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    checkpoint_final = ModelCheckpoint(
        "/home/ubuntu/audio_emotion/checkpoints/final_model_v9.h5",
        monitor='val_accuracy',
        save_best_only=False,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    lr_scheduler = CosineDecayWithWarmup(
        min_lr=5e-6,
        max_lr=3e-4,
        warmup_epochs=8,
        cosine_decay_start=20,
        total_epochs=100
    )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint_best, checkpoint_final, early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )
    
    # Evaluate on validation set
    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    print("Training completed.")
    
    return history

if __name__ == "__main__":
    main()
