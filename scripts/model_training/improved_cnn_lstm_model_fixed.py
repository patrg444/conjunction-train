#!/usr/bin/env python3
"""
Improved CNN-LSTM model with enhanced regularization to prevent overfitting.
Key changes:
1. Increased regularization (dropout and L2)
2. Lower initial learning rate and modified schedule
3. Added attention mechanism
4. Improved class balancing with stratified split
5. Enhanced validation stability
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import Masking, LayerNormalization, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
import time
import datetime
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# CNN audio directories
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed"
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"

# Model parameters
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 20  # Increased patience for early stopping
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Enhanced regularization to combat overfitting
L2_REGULARIZATION = 0.001  # Increased L2 regularization
DROPOUT_RATE = 0.5  # Increased dropout rate
RECURRENT_DROPOUT = 0.3  # Specific recurrent dropout

# Learning rate setup - starting with a lower rate
LEARNING_RATE = 0.0003  # Lower initial learning rate (3e-4)
LR_DECAY_FACTOR = 0.5
LR_PATIENCE = 12  # Increased patience for learning rate reduction

print("IMPROVED CNN-LSTM MODEL WITH ENHANCED REGULARIZATION")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version}")


class FeatureNormalizer:
    """Normalizes features based on computed statistics."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, features_list):
        """Compute mean and std from a list of feature arrays."""
        # Concatenate all features along the time dimension
        all_features = np.vstack([feat for feat in features_list if feat.shape[0] > 0])
        self.mean = np.mean(all_features, axis=0, keepdims=True)
        self.std = np.std(all_features, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1.0, self.std)  # Avoid division by zero
        self.is_fitted = True
        print(f"Fitted normalizer on {len(features_list)} samples, feature shape: {all_features.shape}")
        print(f"Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"Std range: [{self.std.min():.4f}, {self.std.max():.4f}]")
        return self

    def transform(self, features):
        """Normalize features using mean and std."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Handle potential input types (numpy array or tensor)
        if hasattr(features, 'numpy'):
            features_np = features.numpy()
        else:
            features_np = features

        # Apply normalization
        normalized = (features_np - self.mean) / self.std
        return normalized


class DataGenerator(tf.keras.utils.Sequence):
    """Generator that handles CNN audio features with proper normalization."""

    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, normalizer=None):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalizer = normalizer
        self.indexes = np.arange(len(self.file_paths))

        # Get feature dimension from first sample
        print("Determining CNN audio feature dimension from first valid sample...")
        for path in self.file_paths:
            try:
                sample = np.load(path)
                self.feature_dim = sample.shape[1]
                print(f"  Detected CNN feature shape: {sample.shape} -> Dim: {self.feature_dim}")
                break
            except Exception as e:
                print(f"Warning: Could not load {path} - {e}")
                continue

        print(f"\nCreated DataGenerator:")
        print(f"- Samples: {len(self.file_paths)}")
        print(f"- CNN Audio Dim: {self.feature_dim}")
        print(f"- Batch Size: {self.batch_size}")

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize batch data
        batch_x = []
        batch_y = []
        sequence_lengths = []

        # Load and preprocess the data
        for i in batch_indexes:
            try:
                # Load features
                features = np.load(self.file_paths[i])

                # Apply normalizer if provided
                if self.normalizer is not None and self.normalizer.is_fitted:
                    # Normalize each time step independently
                    normalized_features = self.normalizer.transform(features)
                    batch_x.append(normalized_features)
                else:
                    batch_x.append(features)

                batch_y.append(self.labels[i])

                # Track sequence length for each sample
                sequence_lengths.append(features.shape[0])

            except Exception as e:
                print(f"Error loading {self.file_paths[i]}: {e}")
                # Skip this sample or use zero padding
                continue

        # If no valid samples were loaded, create a dummy batch
        if len(batch_x) == 0:
            dummy_x = np.zeros((1, 1, self.feature_dim))
            dummy_y = np.zeros((1, NUM_CLASSES))
            return dummy_x, dummy_y

        # Pad sequences to the maximum length in this batch
        max_len = max(sequence_lengths)
        padded_batch_x = []

        for i, features in enumerate(batch_x):
            # Create padded version of the features
            padded_features = np.zeros((max_len, self.feature_dim))
            # Copy actual features to the beginning
            padded_features[:features.shape[0], :] = features
            padded_batch_x.append(padded_features)

        return np.array(padded_batch_x), np.array(batch_y)

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def load_data_paths_and_labels():
    """Finds CNN audio feature files and extracts labels."""
    # Find CNN audio feature files
    cnn_audio_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "*", "*.npy")) + \
                      glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))

    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None

        # Extract label
        try:
            if "Actor_" in cnn_audio_file:  # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
            else:  # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found.")

    return valid_cnn_audio_files, np.array(labels)


def compute_class_weights(labels):
    """Compute class weights to address class imbalance."""
    # Convert one-hot encoded labels to class indices
    y_indices = np.argmax(labels, axis=1)
    
    # Compute class weights using sklearn's class_weight utility
    class_weights_dict = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_indices),
        y=y_indices
    )
    
    # Convert to dictionary format expected by Keras
    weights = {i: weight for i, weight in enumerate(class_weights_dict)}
    
    print("Class weights (balanced):")
    for i, weight in weights.items():
        print(f"  Class {i}: {weight:.3f}")
        
    return weights


def create_lstm_attention_model(input_dim):
    """Create a regularized model with attention mechanism to combat overfitting."""
    print(f"Creating improved CNN-LSTM model with attention and stronger regularization:")
    print(f"- Input dimension: {input_dim}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Dropout rate: {DROPOUT_RATE}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")

    # Input layer
    inputs = Input(shape=(None, input_dim), name="cnn_features_input")

    # Masking layer to handle variable-length sequences
    x = Masking(mask_value=0.0)(inputs)

    # Batch normalization helps stabilize training
    x = BatchNormalization()(x)

    # Bidirectional LSTM with stronger regularization
    lstm_layer = Bidirectional(LSTM(
        128,  # Increased units 
        return_sequences=True,  # Return sequences for attention
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=l2(L2_REGULARIZATION),
        recurrent_regularizer=l2(L2_REGULARIZATION/2)
    ))(x)
    
    # Apply attention mechanism to focus on relevant parts of the sequence
    context_vector = Attention()([lstm_layer, lstm_layer])
    
    # Global context representation
    x = LayerNormalization()(context_vector)
    
    # FIXED: Add GlobalAveragePooling1D to remove the time dimension
    x = GlobalAveragePooling1D()(x)
    
    # First dense layer with regularization
    x = Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(x)
    
    # Strong dropout
    x = Dropout(DROPOUT_RATE)(x)
    
    # Second dense layer
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION/2)
    )(x)

    # Final dropout
    x = Dropout(DROPOUT_RATE)(x)

    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model with reduced learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def create_alternate_model(input_dim):
    """Create an alternate model focused on global features with regularization."""
    print(f"Creating alternate model with global pooling and regularization:")
    print(f"- Input dimension: {input_dim}")

    # Input layer
    inputs = Input(shape=(None, input_dim), name="cnn_features_input")

    # Masking layer
    x = Masking(mask_value=0.0)(inputs)

    # Time-distributed dense to transform features before pooling
    x = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION)))(x)
    x = TimeDistributed(Dropout(DROPOUT_RATE/2))(x)
    
    # Global average pooling - simplifies temporal dimension completely
    pooled = GlobalAveragePooling1D()(x)

    # Batch normalization
    x = BatchNormalization()(pooled)

    # First dense layer with stronger regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Second dense layer
    x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION/2))(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create and compile model with reduced learning rate
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def train_model():
    """Main function to train the improved model."""
    print("Starting improved CNN-LSTM model with regularization...\n")

    # Load data paths and labels
    try:
        cnn_audio_files, all_labels = load_data_paths_and_labels()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Compute class weights
    class_weights = compute_class_weights(all_labels)

    # Fit feature normalizer on a subset of data
    print("Training feature normalizer...")
    sample_features = []
    for i in tqdm(range(min(500, len(cnn_audio_files))), desc="Loading samples for normalization"):
        try:
            features = np.load(cnn_audio_files[i])
            sample_features.append(features)
        except Exception as e:
            print(f"Error loading file {cnn_audio_files[i]}: {e}")

    normalizer = FeatureNormalizer().fit(sample_features)

    # Get class indices for stratified splitting
    class_indices = np.argmax(all_labels, axis=1)
    
    # Use sklearn's stratified split to maintain class distribution
    train_idx, val_idx = train_test_split(
        np.arange(len(cnn_audio_files)),
        test_size=1-TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=class_indices  # This ensures balanced classes in both train and val
    )

    # Create file lists for train/val
    train_cnn_audio_files = [cnn_audio_files[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_cnn_audio_files = [cnn_audio_files[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    # Report class distribution in splits
    train_class_dist = np.sum(train_labels, axis=0)
    val_class_dist = np.sum(val_labels, axis=0)
    print("\nTrain/Val split with class distribution:")
    print(f"- Train samples: {len(train_cnn_audio_files)}")
    for i, count in enumerate(train_class_dist):
        print(f"  Class {i}: {count} samples ({count/sum(train_class_dist)*100:.1f}%)")
    print(f"- Validation samples: {len(val_cnn_audio_files)}")
    for i, count in enumerate(val_class_dist):
        print(f"  Class {i}: {count} samples ({count/sum(val_class_dist)*100:.1f}%)")

    # Create data generators
    print("\nCreating data generators...")
    train_generator = DataGenerator(
        train_cnn_audio_files,
        train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        normalizer=normalizer
    )

    val_generator = DataGenerator(
        val_cnn_audio_files,
        val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        normalizer=normalizer
    )

    # Get input dimension from the first batch
    for batch_x, _ in train_generator:
        input_dim = batch_x.shape[2]
        break

    # Create both model architectures
    lstm_model = create_lstm_attention_model(input_dim)
    lstm_model.summary()

    alternate_model = create_alternate_model(input_dim)
    alternate_model.summary()

    # Define callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = f"models/improved_cnn_lstm_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)

    lstm_callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'lstm_attention_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_DECAY_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
    ]

    alt_callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'pooling_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_DECAY_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
    ]

    # Train the LSTM model
    print("\n=== Training LSTM Model with Attention ===")
    lstm_history = lstm_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=lstm_callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Train the alternate model
    print("\n=== Training Pooling Model ===")
    alt_history = alternate_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=alt_callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Compare models
    print("\n=== Model Performance Comparison ===")
    lstm_val_acc = max(lstm_history.history['val_accuracy'])
    alt_val_acc = max(alt_history.history['val_accuracy'])

    print(f"LSTM with Attention Validation Accuracy: {lstm_val_acc:.4f}")
    print(f"Pooling Model Validation Accuracy: {alt_val_acc:.4f}")

    print(f"\nBest models saved to {models_dir}")


if __name__ == "__main__":
    train_model()
