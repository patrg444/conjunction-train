#!/usr/bin/env python3
# Fixed Wav2Vec Emotion Recognition Script
# This version properly handles dataset-specific emotion coding with correct NPZ keys

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Directory setup
data_dir = "/home/ubuntu/audio_emotion"
wav2vec_dir = os.path.join(data_dir, "models/wav2vec")
checkpoint_dir = os.path.join(data_dir, "checkpoints")
logs_dir = os.path.join(data_dir, "logs")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Define emotion mapping with continuous indices
emotion_to_index = {
    'neutral': 0,  # Neutral and calm are combined as class 0
    'calm': 0,     # Mapped to neutral (same as index 0)
    'happy': 1,    # Was 2, now 1 - continuous indexing
    'sad': 2,      # Was 3, now 2
    'angry': 3,    # Was 4, now 3
    'fear': 4,     # Was 5, now 4
    'disgust': 5,  # Was 6, now 5
    'surprise': 6  # Was 7, now 6
}

# Dataset-specific emotion code mappings
# CREMA-D uses 3-letter codes
cremad_code_to_emotion = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# RAVDESS uses numeric codes (3rd position in filename)
ravdess_code_to_emotion = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

print("Using emotion mapping:")
for emotion, index in sorted(emotion_to_index.items(), key=lambda x: x[1]):
    print(f"  {emotion} -> {index}")

# Learning rate scheduler that fixes ResourceVariable issue
class WarmUpReduceLROnPlateau(Callback):
    def __init__(self, warmup_epochs=5, reduce_factor=0.5, patience=5, min_lr=1e-6, verbose=1):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.reduce_factor = reduce_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # During warmup, gradually increase learning rate
            lr = self.model.optimizer.learning_rate
            warmup_lr = float(lr) * ((epoch + 1) / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: WarmUpReduceLROnPlateau setting learning rate to {warmup_lr}.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        
        if val_loss is None:
            return
            
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.reduce_factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch+1}: WarmUpReduceLROnPlateau reducing learning rate to {new_lr}.")
                    self.wait = 0

def build_model(input_shape, num_classes):
    """Build LSTM model with wav2vec embeddings"""
    input_layer = Input(shape=input_shape, name='input_layer')
    
    # Bidirectional LSTM for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    
    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer for classification
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def main():
    """Main training function"""
    # Find all available wav2vec feature files
    feature_files = [os.path.join(wav2vec_dir, f) for f in os.listdir(wav2vec_dir) if f.endswith('.npz')]
    
    # Lists to hold valid file paths and labels
    valid_files = []
    labels = []

    skipped_files = 0
    emotion_counts = {}

    for file_path in feature_files:
        base_name = os.path.basename(file_path)

        # Try to extract label from filename
        try:
            emotion = None
            
            # Handle CREMA-D dataset (cremad_1001_DFA_ANG_XX.npz)
            if base_name.startswith('cremad_'):
                parts = base_name.split('_')
                if len(parts) >= 4:
                    emotion_code = parts[3]  # ANG, DIS, FEA, etc.
                    emotion = cremad_code_to_emotion.get(emotion_code)
            
            # Handle RAVDESS dataset (ravdess_01-01-01-01-01-01-01.npz)
            elif base_name.startswith('ravdess_'):
                parts = base_name[8:].split('-')  # Remove 'ravdess_' prefix
                if len(parts) >= 3:
                    emotion_code = parts[2]  # Third digit is emotion
                    emotion = ravdess_code_to_emotion.get(emotion_code)
            
            # Check if we identified a valid emotion
            if emotion:
                emotion_idx = emotion_to_index.get(emotion)
                if emotion_idx is not None:
                    # Append to valid lists only if we have a valid emotion
                    labels.append(emotion_idx)
                    valid_files.append(file_path)
                    # Count emotions for distribution stats
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                else:
                    skipped_files += 1
                    print(f"Skipping file with unknown emotion mapping: {base_name}, emotion: {emotion}")
            else:
                skipped_files += 1
                print(f"Couldn't extract emotion from filename: {base_name}")
        except Exception as e:
            print(f"Error parsing filename {base_name}: {e}")
            skipped_files += 1
            continue

    print(f"Skipped {skipped_files} files due to parsing errors or excluded emotions")
    print(f"Using {len(valid_files)} valid files")

    # Print distribution of emotions before combining
    print("\nEmotion distribution in dataset:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[0]):
        print(f"  {emotion}: {count} samples")

    if len(valid_files) == 0:
        raise ValueError("No valid files found after parsing!")

    # Load normalization statistics if available
    mean_path = os.path.join(data_dir, "audio_mean.npy")
    std_path = os.path.join(data_dir, "audio_std.npy")

    if os.path.exists(mean_path) and os.path.exists(std_path):
        print("Loading existing normalization statistics")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        print("Computing normalization statistics...")
        # Load a subset of files to compute statistics
        sample_size = min(500, len(valid_files))
        sample_files = random.sample(valid_files, sample_size)
        
        # Load and concatenate features
        sample_features = []
        for file in sample_files:
            data = np.load(file)
            sample_features.append(data['wav2vec_features'])  # Fixed key name
        
        # Compute mean and std
        sample_features = np.vstack(sample_features)
        mean = np.mean(sample_features, axis=0)
        std = np.std(sample_features, axis=0)
        
        # Save for future use
        np.save(mean_path, mean)
        np.save(std_path, std)

    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Print unique values in labels before encoding
    print("Original unique label values:", np.unique(labels))
    
    # One-hot encode labels
    num_classes = len(np.unique(labels))
    encoded_labels = to_categorical(labels, num_classes=num_classes)
    
    # Print information about encoded shape
    print(f"Number of classes after encoding: {num_classes}")
    
    # Split data into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        valid_files, encoded_labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Check for overlap between train and validation
    train_set = set(train_files)
    val_set = set(val_files)
    overlap = train_set.intersection(val_set)
    if overlap:
        print(f"Warning: {len(overlap)} files overlap between training and validation sets!")
    else:
        print("No overlap detected between training and validation sets.")
    
    # Determine sequence length distribution
    print("Determining sequence length distribution...")
    
    # Calculate class weights to handle imbalance
    class_weights = {}
    for i in range(num_classes):
        class_count = np.sum(labels == i)
        weight = 1.0 / (class_count / len(labels)) if class_count > 0 else 0
        class_weights[i] = weight
        print(f"  Class {i}: {class_count} samples (weight: {weight:.4f})")

    # Define generators for training and validation
    def data_generator(file_list, label_list, batch_size=32):
        num_samples = len(file_list)
        while True:
            # Shuffle at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_features = []
                batch_labels = []
                
                for idx in batch_indices:
                    # Load the features
                    data = np.load(file_list[idx])
                    features = data['wav2vec_features']  # Fixed key name
                    
                    # Normalize
                    features = (features - mean) / (std + 1e-8)
                    
                    batch_features.append(features)
                    batch_labels.append(label_list[idx])
                
                # Pad sequences to the same length
                max_len = max(feat.shape[0] for feat in batch_features)
                padded_features = np.zeros((len(batch_features), max_len, batch_features[0].shape[1]))
                
                for i, feat in enumerate(batch_features):
                    padded_features[i, :feat.shape[0], :] = feat
                
                yield padded_features, np.array(batch_labels)

    # Get a sample file to determine feature dimensions
    sample_data = np.load(valid_files[0])
    sample_features = sample_data['wav2vec_features']  # Fixed key name
    feature_dim = sample_features.shape[1]
    
    # Determine a reasonable sequence length from the data
    # For this example, we'll use a fixed length based on the first file
    seq_length = sample_features.shape[0]  # Use the sequence length from the first file
    
    # Build the model
    print(f"Building model with {num_classes} output classes...")
    input_shape = (None, feature_dim)  # Variable sequence length
    model = build_model(input_shape, num_classes)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, "wav2vec_six_classes_best.weights.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        WarmUpReduceLROnPlateau(
            warmup_epochs=5,
            reduce_factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # Set up data generators
    batch_size = 32
    train_generator = data_generator(train_files, train_labels, batch_size)
    val_generator = data_generator(val_files, val_labels, batch_size)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // batch_size
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Train the model - REMOVED class_weight parameter which is incompatible with generators
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set:")
    val_loss, val_accuracy = model.evaluate(val_generator, steps=validation_steps)
    print(f"Validation loss: {val_loss:.4f} Validation accuracy: {val_accuracy:.4f}")
    
    print("Training complete. Best weights saved to", checkpoint_path)

if __name__ == "__main__":
    main()
