#!/usr/bin/env python3
"""
Final CNN-LSTM model with fixed sequence length and attention.
Key improvements:
1. Fixed sequence length padding to ensure consistent batch shapes
2. Simplified model architecture with GlobalAveragePooling1D
3. Enhanced regularization (dropout and L2)
4. Lower initial learning rate with adaptive scheduling
5. Stratified train/validation split
6. Class weighting to handle imbalance
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
MAX_SEQ_LENGTH = 20  # Set a fixed maximum sequence length for all batches
EPOCHS = 100
PATIENCE = 20
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Enhanced regularization to combat overfitting
L2_REGULARIZATION = 0.001
DROPOUT_RATE = 0.5
RECURRENT_DROPOUT = 0.3

# Learning rate setup
LEARNING_RATE = 0.0003
LR_DECAY_FACTOR = 0.5
LR_PATIENCE = 12

print("FINAL CNN-LSTM MODEL WITH FIXED SEQUENCE LENGTH")
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


class FixedLengthDataGenerator(tf.keras.utils.Sequence):
    """Generator that handles CNN audio features with fixed sequence length."""

    def __init__(self, file_paths, labels, batch_size=32, max_seq_length=MAX_SEQ_LENGTH, shuffle=True, normalizer=None):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length  # Fixed maximum sequence length
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

        print(f"\nCreated FixedLengthDataGenerator:")
        print(f"- Samples: {len(self.file_paths)}")
        print(f"- CNN Audio Dim: {self.feature_dim}")
        print(f"- Fixed Max Sequence Length: {self.max_seq_length}")
        print(f"- Batch Size: {self.batch_size}")

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data with fixed sequence length."""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize batch data
        batch_x = []
        batch_y = []

        # Load and preprocess the data
        for i in batch_indexes:
            try:
                # Load features
                features = np.load(self.file_paths[i])

                # Apply normalizer if provided
                if self.normalizer is not None and self.normalizer.is_fitted:
                    normalized_features = self.normalizer.transform(features)
                    features = normalized_features

                # Prepare a fixed-length sequence
                if features.shape[0] > self.max_seq_length:
                    # If sequence is longer than max_seq_length, truncate it
                    fixed_length_features = features[:self.max_seq_length, :]
                else:
                    # If sequence is shorter, pad with zeros
                    fixed_length_features = np.zeros((self.max_seq_length, self.feature_dim))
                    fixed_length_features[:features.shape[0], :] = features

                batch_x.append(fixed_length_features)
                batch_y.append(self.labels[i])

            except Exception as e:
                print(f"Error loading {self.file_paths[i]}: {e}")
                # Skip this sample
                continue

        # If no valid samples were loaded, create a dummy batch
        if len(batch_x) == 0:
            dummy_x = np.zeros((1, self.max_seq_length, self.feature_dim))
            dummy_y = np.zeros((1, NUM_CLASSES))
            return dummy_x, dummy_y

        return np.array(batch_x), np.array(batch_y)

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


def create_lstm_attention_model(input_dim, sequence_length=MAX_SEQ_LENGTH):
    """Create a regularized model with attention mechanism and fixed input shape."""
    print(f"Creating CNN-LSTM model with attention and fixed sequence length:")
    print(f"- Input dimension: {input_dim}")
    print(f"- Fixed sequence length: {sequence_length}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Dropout rate: {DROPOUT_RATE}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")

    # Input layer with fixed sequence length
    inputs = Input(shape=(sequence_length, input_dim), name="cnn_features_input")

    # Masking layer to handle padding
    x = Masking(mask_value=0.0)(inputs)

    # Batch normalization helps stabilize training
    x = BatchNormalization()(x)

    # Bidirectional LSTM with regularization
    lstm_layer = Bidirectional(LSTM(
        128,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=l2(L2_REGULARIZATION),
        recurrent_regularizer=l2(L2_REGULARIZATION/2)
    ))(x)
    
    # Apply attention mechanism to focus on important parts
    context_vector = Attention()([lstm_layer, lstm_layer])
    
    # Global pooling to collapse the time dimension
    x = GlobalAveragePooling1D()(context_vector)
    
    # Dense layers with regularization
    x = Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(x)
    
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION/2)
    )(x)

    x = Dropout(DROPOUT_RATE)(x)

    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def train_model():
    """Main function to train the fixed sequence length model."""
    print("Starting final CNN-LSTM model with fixed sequence length...\n")

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
    
    # Use sklearn's stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(cnn_audio_files)),
        test_size=1-TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=class_indices
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

    # Create fixed-length data generators
    print("\nCreating fixed-length data generators...")
    train_generator = FixedLengthDataGenerator(
        train_cnn_audio_files,
        train_labels,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        shuffle=True,
        normalizer=normalizer
    )

    val_generator = FixedLengthDataGenerator(
        val_cnn_audio_files,
        val_labels,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        shuffle=False,
        normalizer=normalizer
    )

    # Determine the input dimension
    feature_dim = train_generator.feature_dim
    
    # Create LSTM model with fixed sequence length
    model = create_lstm_attention_model(feature_dim, MAX_SEQ_LENGTH)
    model.summary()

    # Define callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = f"models/fixed_cnn_lstm_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'best_model.h5'),
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

    # Train the model
    print("\n=== Training CNN-LSTM Model with Fixed Sequence Length ===")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Display results
    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.4f}")
    print(f"Model saved to {os.path.join(models_dir, 'best_model.h5')}")


if __name__ == "__main__":
    train_model()
