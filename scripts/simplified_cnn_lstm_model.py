#!/usr/bin/env python3
"""
Simplified CNN-LSTM model with fixed sequence length.
Key improvements:
1. Fixed sequence length padding to ensure consistent batch shapes
2. Simplified architecture without the problematic Attention layer
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
from tensorflow.keras.layers import Masking, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import datetime
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
LR_PATIENCE = 10

print("SIMPLIFIED CNN-LSTM MODEL WITH FIXED SEQUENCE LENGTH")
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
        return (features - self.mean) / self.std
    
    def save(self, mean_file="audio_mean.npy", std_file="audio_std.npy"):
        """Save normalizer parameters to files."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before saving")
        np.save(mean_file, self.mean)
        np.save(std_file, self.std)
        print(f"Saved normalizer parameters to {mean_file} and {std_file}")
        
    def load(self, mean_file="audio_mean.npy", std_file="audio_std.npy"):
        """Load normalizer parameters from files."""
        self.mean = np.load(mean_file)
        self.std = np.load(std_file)
        self.is_fitted = True
        print(f"Loaded normalizer parameters from {mean_file} and {std_file}")
        print(f"Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
        print(f"Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"Std range: [{self.std.min():.4f}, {self.std.max():.4f}]")
        return self


def pad_or_truncate(sequence, max_length):
    """Pad or truncate sequence to fixed length."""
    if sequence.shape[0] > max_length:
        return sequence[:max_length]
    elif sequence.shape[0] < max_length:
        # Create padding of zeros with the same feature dimension
        padding = np.zeros((max_length - sequence.shape[0], sequence.shape[1]))
        return np.vstack([sequence, padding])
    else:
        return sequence


def load_cnn_audio_files():
    """Load CNN audio features from disk with padded sequences."""
    print("Loading CNN audio features...")
    
    # Get all npz files
    ravdess_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "*.npz"))
    crema_d_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npz"))
    
    cnn_audio_files = ravdess_files + crema_d_files
    print(f"Found {len(cnn_audio_files)} total CNN audio files:")
    print(f" - RAVDESS: {len(ravdess_files)} files")
    print(f" - CREMA-D: {len(crema_d_files)} files")
    
    # Extract features and labels
    features = []
    labels = []
    
    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        try:
            npz = np.load(cnn_audio_file)
            
            # Extract audio features and labels
            feature = npz['cnn_features']
            emotion = npz['emotion']
            
            # Validate data
            if feature.shape[0] == 0 or emotion.shape[0] == 0:
                print(f"Skipping empty file: {cnn_audio_file}")
                continue
                
            # Convert emotion to appropriate class index (0-5)
            # We assume emotions are coded as 1-based indices (1-6)
            emotion_idx = int(emotion[0]) - 1
            
            # Skip if emotion is out of range
            if emotion_idx < 0 or emotion_idx >= NUM_CLASSES:
                print(f"Skipping file with invalid emotion {emotion[0]}: {cnn_audio_file}")
                continue
                
            # Add to dataset
            features.append(feature)
            labels.append(emotion_idx)
            
        except Exception as e:
            print(f"Error loading {cnn_audio_file}: {e}")
    
    print(f"Successfully loaded {len(features)} valid samples")
    
    # Check for class imbalance
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        print(f" - Class {label} (Emotion {label+1}): {count} samples ({count/len(labels)*100:.1f}%)")
    
    return features, np.array(labels)


def create_padded_dataset(features, labels):
    """Create a padded dataset with fixed length sequences."""
    print("\nCreating padded dataset...")
    
    # Normalize features
    normalizer = FeatureNormalizer()
    normalized_features = []
    
    print("Fitting feature normalizer...")
    normalizer.fit(features)
    normalizer.save()  # Save for later use
    
    print("Normalizing and padding features...")
    for feature in tqdm(features):
        normalized_feature = normalizer.transform(feature)
        padded_feature = pad_or_truncate(normalized_feature, MAX_SEQ_LENGTH)
        normalized_features.append(padded_feature)
    
    # Convert to numpy arrays
    X = np.array(normalized_features)
    y = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=1-TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels  # Ensure balanced classes in split
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nClass weights to handle imbalance:")
    for cls, weight in class_weight_dict.items():
        print(f" - Class {cls}: {weight:.4f}")
    
    return X_train, X_val, y_train, y_val, class_weight_dict


def create_simplified_lstm_model(input_dim, sequence_length):
    """Create a simplified LSTM model without attention."""
    print("\nCreating simplified LSTM model without attention:")
    print(f"- Input dimension: {input_dim}")
    print(f"- Fixed sequence length: {sequence_length}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Dropout rate: {DROPOUT_RATE}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")
    
    # Input for CNN features
    inputs = Input(shape=(sequence_length, input_dim), name="cnn_features_input")
    
    # Masking layer to handle padding
    masked = Masking(mask_value=0.0)(inputs)
    
    # Batch normalization
    norm = BatchNormalization()(masked)
    
    # Bidirectional LSTM with regularization
    lstm = Bidirectional(LSTM(
        128,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=l2(L2_REGULARIZATION)
    ))(norm)
    
    # Global pooling instead of attention
    pooled = GlobalAveragePooling1D()(lstm)
    
    # Dense layers with dropout
    dense1 = Dense(
        256, 
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(pooled)
    dropout1 = Dropout(DROPOUT_RATE)(dense1)
    
    dense2 = Dense(
        128, 
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(dropout1)
    dropout2 = Dropout(DROPOUT_RATE)(dense2)
    
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(dropout2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, class_weight_dict):
    """Train the model with callbacks for early stopping and learning rate reduction."""
    print("\nTraining model...")
    
    # Create model checkpoint callback
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"cnn_lstm_simplified_{timestamp}" + "_{epoch:02d}_{val_accuracy:.4f}.h5"
    )
    
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_DECAY_FACTOR,
        patience=LR_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("\nTraining completed!")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"final_cnn_lstm_simplified_{timestamp}.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return history, final_model_path


def test_model_with_dummy_data():
    """Run a quick test of the model with synthetic data."""
    print("\nTesting model with synthetic data...")
    
    # Define feature dimension
    feature_dim = 2048  # This should match your CNN features
    
    # Create a model
    model = create_simplified_lstm_model(feature_dim, MAX_SEQ_LENGTH)
    
    # Generate synthetic batch
    dummy_batch = np.random.randn(BATCH_SIZE, MAX_SEQ_LENGTH, feature_dim)
    dummy_labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    dummy_y = tf.keras.utils.to_categorical(dummy_labels, NUM_CLASSES)
    
    # Make a prediction
    predictions = model.predict(dummy_batch, verbose=1)
    print(f"Predictions shape: {predictions.shape}")
    
    # Test training on dummy data
    history = model.fit(
        dummy_batch, 
        dummy_y,
        epochs=2,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("Model test completed successfully!")
    return True


def main():
    """Main function to run the CNN-LSTM training pipeline."""
    # Start by testing the model with dummy data
    test_success = test_model_with_dummy_data()
    if not test_success:
        print("Model test failed. Exiting...")
        return
    
    # Load and preprocess the data
    features, labels = load_cnn_audio_files()
    X_train, X_val, y_train, y_val, class_weight_dict = create_padded_dataset(features, labels)
    
    # Get feature dimension from the data
    feature_dim = X_train.shape[2]
    
    # Create and train the model
    model = create_simplified_lstm_model(feature_dim, MAX_SEQ_LENGTH)
    history, _ = train_model(model, X_train, y_train, X_val, y_val, class_weight_dict)
    
    # Print final validation accuracy
    final_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_with_dummy_data()
    else:
        main()
