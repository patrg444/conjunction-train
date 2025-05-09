#!/usr/bin/env python3
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import re
import sys

# Import the custom generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_video_generator import VideoOnlyFacenetGenerator

# Set paths and parameters
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Reverse mapping for class indices
LABEL_TO_INDEX = {label: i for i, label in enumerate(sorted(set(EMOTION_LABELS.values())))}

def get_emotion_from_filename(filename):
    """Extract emotion code from filename."""
    try:
        # Extract the emotion code (01-08) from the filename
        basename = os.path.basename(filename)
        # Extract the emotion code based on the standard naming pattern
        match = re.search(r'-(\d{2})-', basename)
        if match:
            emotion_code = match.group(1)
            if emotion_code in EMOTION_LABELS:
                return EMOTION_LABELS[emotion_code]
        
        # If we couldn't extract the code using the pattern, try looking for it directly
        for code, emotion in EMOTION_LABELS.items():
            if f"-{code}-" in basename:
                return emotion
        
        return None
    except Exception as e:
        print(f"Error extracting emotion from {filename}: {e}")
        return None

def extract_labels_from_files(files):
    """Extract emotion labels from file names."""
    labels = []
    valid_files = []
    
    for file_path in tqdm(files, desc="Extracting labels"):
        emotion = get_emotion_from_filename(file_path)
        if emotion is not None and emotion in LABEL_TO_INDEX:
            # Convert to one-hot encoding
            label_idx = LABEL_TO_INDEX[emotion]
            one_hot = np.zeros(len(LABEL_TO_INDEX))
            one_hot[label_idx] = 1
            labels.append(one_hot)
            valid_files.append(file_path)
        else:
            print(f"Warning: Could not extract valid emotion from {file_path}")
    
    return valid_files, labels

def find_facenet_files(facenet_dir="/home/ubuntu/emotion-recognition/crema_d_features_facenet"):
    """Find all Facenet feature files."""
    # Using glob to find all npz files
    facenet_files = glob.glob(os.path.join(facenet_dir, "*.npz"))
    
    if not facenet_files:
        facenet_files = glob.glob(os.path.join(facenet_dir, "**", "*.npz"), recursive=True)
    
    print(f"Found {len(facenet_files)} precomputed Facenet files. Extracting labels...")
    
    # Extract labels and filter out files with invalid labels
    valid_files, labels = extract_labels_from_files(facenet_files)
    
    print(f"Found {len(valid_files)} Facenet files with valid labels and features.")
    print(f"Skipped {len(facenet_files) - len(valid_files)} due to label parsing or feature issues.")
    
    return valid_files, labels

def get_max_seq_length(files):
    """Find the maximum sequence length from the feature files."""
    max_len = 0
    for file in tqdm(files, desc="Checking lengths"):
        try:
            features = np.load(file)
            max_len = max(max_len, features.shape[0])
        except Exception as e:
            print(f"Error checking length of {file}: {e}")
    return max_len

def split_data(files, labels, val_split=0.2):
    """Split data into training and validation sets."""
    # Convert to numpy arrays for easier manipulation
    files = np.array(files)
    labels = np.array(labels)
    
    # Generate indices and shuffle
    indices = np.arange(len(files))
    np.random.shuffle(indices)
    
    # Calculate split point
    split_idx = int(len(indices) * (1 - val_split))
    
    # Split the data
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation sets
    train_files = files[train_indices].tolist()
    train_labels = labels[train_indices].tolist()
    val_files = files[val_indices].tolist()
    val_labels = labels[val_indices].tolist()
    
    print("\nTrain/Val split (Full):")
    print(f"- Train samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    return train_files, train_labels, val_files, val_labels

def build_lstm_model(seq_length, feature_dim, num_classes, learning_rate=0.001):
    """Build and compile the LSTM model for video features."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, feature_dim)),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the Video-Only LSTM model on Facenet features."""
    print("VIDEO ONLY FACENET LSTM (KEY-FIXED VERSION)")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    
    # Parameters
    batch_size = 32
    epochs = 100
    learning_rate = 0.0005
    
    print("Starting Precomputed Facenet LSTM model training (Video-Only)...")
    print(f"- Using learning rate: {learning_rate}")
    
    # Find video facenet feature files
    facenet_dir = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    video_files, video_labels = find_facenet_files(facenet_dir)
    
    # Split the data
    train_video_files, train_video_labels, val_video_files, val_video_labels = split_data(
        video_files, video_labels
    )
    
    # Calculate maximum sequence length for the facenet features
    print("\nCalculating maximum sequence length from training Facenet features...")
    max_seq_len = get_max_seq_length(train_video_files)
    print(f"Determined max_seq_len for video: {max_seq_len}")
    
    # Create data generators
    print("\nCreating data generators (Precomputed Facenet - Video Only)...")
    train_generator = VideoOnlyFacenetGenerator(
        video_feature_files=train_video_files,
        labels=train_video_labels,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=True
    )
    
    val_generator = VideoOnlyFacenetGenerator(
        video_feature_files=val_video_files,
        labels=val_video_labels,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=False
    )
    
    # Build and compile the model
    feature_dim = 512  # Facenet feature dimension
    num_classes = len(LABEL_TO_INDEX)
    model = build_lstm_model(max_seq_len, feature_dim, num_classes, learning_rate)
    
    print("\nModel Summary:")
    model.summary()
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"/home/ubuntu/emotion-recognition/models/video_only_facenet_lstm_{timestamp}"
    
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_filepath = os.path.join(model_dir, "best_model_video_only_facenet_lstm.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, early_stopping, reduce_lr]
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model_video_only_facenet_lstm.keras")
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to: {final_model_path}")
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    history_plot_path = os.path.join(model_dir, "training_history.png")
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to: {history_plot_path}")
    
    # Save history to CSV
    history_path = os.path.join(model_dir, "training_history.csv")
    with open(history_path, 'w') as f:
        f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1},{history.history['loss'][i]},{history.history['accuracy'][i]},{history.history['val_loss'][i]},{history.history['val_accuracy'][i]}\n")
    
    print(f"Training history saved to: {history_path}")
    
    # Print final accuracy
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print("\nTraining Results:")
    print(f"- Final training accuracy: {final_train_acc:.4f}")
    print(f"- Final validation accuracy: {final_val_acc:.4f}")
    print(f"- Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

# Main execution
if __name__ == "__main__":
    train_model()
