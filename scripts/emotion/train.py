#!/usr/bin/env python3
"""
Training script for the multimodal emotion recognition model.
"""

import os
import sys
import glob
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from .synchronize_test import parse_ravdess_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)

def load_data(data_dir, dataset_name):
    """Loads preprocessed data and labels from a directory.

    Args:
        data_dir: Directory containing the preprocessed .npz files.
        dataset_name: name of the dataset ('RAVDESS' or 'CREMA-D')

    Returns:
        Tuple: (video_sequences, audio_sequences, labels, filenames)
               - video_sequences is a list of video feature sequences
               - audio_sequences is a list of audio feature sequences
               - labels is a list of corresponding emotion labels (numeric)
               - filenames is a list of the base filenames
    """
    video_sequences = []
    audio_sequences = []
    labels = []
    filenames = []
    
    file_pattern = os.path.join(data_dir, "*.npz")
    files = glob.glob(file_pattern)

    if not files:
        logging.error(f"No .npz files found in {data_dir}")
        return None, None, None, None

    for file_path in files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Check for new format with separate video and audio sequences
            if 'video_sequences' in data and 'audio_sequences' in data:
                # Get video sequences
                video_seqs = data['video_sequences']
                # Get audio sequences
                audio_seqs = data['audio_sequences']
                
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # Get emotion label
                if 'emotion_label' in data:
                    emotion_labels = data['emotion_label']
                    if not isinstance(emotion_labels, np.ndarray):
                        # Convert scalar to array if needed
                        emotion_labels = np.array([emotion_labels])
                else:
                    # Try to parse from filename if not in the file
                    emotion_info = None
                    if dataset_name == 'RAVDESS':
                        emotion_info = parse_ravdess_filename(file_path)
                    
                    if emotion_info:
                        emotion_labels = np.array([emotion_info['numeric']] * len(video_seqs))
                    else:
                        logging.warning(f"Could not find emotion label for {file_path}, skipping.")
                        continue
                
                # Add data to our collections
                for i in range(len(video_seqs)):
                    video_sequences.append(video_seqs[i])
                    
                    # Handle audio sequences
                    # For audio, we need to ensure each sequence has the same length
                    # This is for demo - in practice, you might use more sophisticated
                    # sequence alignment techniques
                    audio_sequences.append(audio_seqs[i])
                    
                    # Add label and filename for each sequence
                    if len(emotion_labels) == 1:
                        # If there's only one label for all sequences
                        labels.append(emotion_labels[0])
                    else:
                        # If there's a label per sequence
                        labels.append(emotion_labels[i])
                    
                    filenames.append(base_filename)
            
            else:
                logging.warning(f"{file_path} does not contain 'video_sequences' and 'audio_sequences'. Trying legacy format...")
                
                # Try legacy format (fallback for older files)
                if 'sequences' in data and 'sequence_lengths' in data:
                    logging.warning(f"Found legacy format in {file_path}. This won't work with the dual-stream model.")
                else:
                    logging.error(f"Unrecognized data format in {file_path}, skipping.")
                continue

        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    logging.info(f"Loaded {len(video_sequences)} video sequences and {len(audio_sequences)} audio sequences")
    
    return video_sequences, audio_sequences, labels, filenames

def create_datasets(features, labels, test_size=0.1, val_size=0.1, random_state=42):
    """Creates training, validation, and test datasets.

    Args:
        features: List of feature arrays.
        labels: List of corresponding labels.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if len(features) == 0:
        logging.error("No features provided to create_datasets.")
        return None, None, None, None, None, None

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Split into training and temporary set (test + validation)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )

    # Split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def define_model(video_input_shape, audio_input_shape, num_classes):
    """Defines the dual-stream LSTM-based model architecture.

    Args:
        video_input_shape: Shape of the video input data (sequence_length, feature_dim).
        audio_input_shape: Shape of the audio input data (sequence_length, feature_dim).
        num_classes: Number of emotion classes.

    Returns:
        Compiled Keras model.
    """
    # Define inputs for both streams
    video_input = Input(shape=video_input_shape, name='video_input')
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    
    # Video stream
    video_lstm1 = LSTM(128, return_sequences=True, name='video_lstm1')(video_input)
    video_lstm2 = LSTM(64, name='video_lstm2')(video_lstm1)
    
    # Audio stream
    audio_lstm1 = LSTM(128, return_sequences=True, name='audio_lstm1')(audio_input)
    audio_lstm2 = LSTM(64, name='audio_lstm2')(audio_lstm1)
    
    # Late fusion (concatenate the outputs of both streams)
    fusion = concatenate([video_lstm2, audio_lstm2], name='fusion')
    
    # Additional dense layers after fusion
    dense1 = Dense(128, activation='relu', name='dense1')(fusion)
    dropout = tf.keras.layers.Dropout(0.5)(dense1)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax', name='output')(dropout)
    
    # Create the dual-stream model
    model = Model(inputs=[video_input, audio_input], outputs=output_layer)
    
    return model

def compile_model(model, learning_rate=0.001):
    """Compiles the Keras model.

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for the optimizer.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            # F1-score can be calculated from precision and recall
        ]
    )

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=32,
    model_save_path='models/'
):
    """Trains the model.

    Args:
        model: Compiled Keras model.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        epochs: Number of training epochs.
        batch_size: Batch size.
        model_save_path: Directory to save model checkpoints

    Returns:
        Trained Keras model.
    """
    # Create callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # Save entire model
        mode='min'  # Save the model when validation loss is minimized
    )
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: Test labels.
    """
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score:.4f}")

    # Confusion matrix can be added here using sklearn.metrics

def create_multimodal_datasets(video_sequences, audio_sequences, labels, test_size=0.1, val_size=0.1, random_state=42):
    """Creates training, validation, and test datasets for multimodal data.

    Args:
        video_sequences: List of video feature sequences.
        audio_sequences: List of audio feature sequences.
        labels: List of corresponding labels.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing training, validation, and test data and labels.
    """
    if len(video_sequences) == 0 or len(audio_sequences) == 0:
        logging.error("Empty video or audio sequences provided.")
        return None
    
    if len(video_sequences) != len(audio_sequences) or len(video_sequences) != len(labels):
        logging.error(f"Mismatched data lengths: {len(video_sequences)} video, {len(audio_sequences)} audio, {len(labels)} labels")
        return None

    # Convert to numpy arrays
    video_data = np.array(video_sequences)
    audio_data = np.array(audio_sequences)
    labels_array = np.array(labels)

    # Create indices for splitting
    indices = np.arange(len(video_sequences))
    
    # Split into train and temp (val+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(test_size + val_size), random_state=random_state, 
        stratify=labels_array
    )
    
    # Split temp into val and test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_size / (test_size + val_size)), 
        random_state=random_state, stratify=labels_array[temp_idx]
    )
    
    # Training data
    X_train_video = video_data[train_idx]
    X_train_audio = audio_data[train_idx]
    y_train = labels_array[train_idx]
    
    # Validation data
    X_val_video = video_data[val_idx]
    X_val_audio = audio_data[val_idx]
    y_val = labels_array[val_idx]
    
    # Test data
    X_test_video = video_data[test_idx]
    X_test_audio = audio_data[test_idx]
    y_test = labels_array[test_idx]
    
    # Handle variable-length sequences (if needed)
    # Pad all sequences to the same length or use a more advanced approach
    # For simplicity, we'll assume the sequences are already properly padded
    
    return (X_train_video, X_train_audio, y_train, 
            X_val_video, X_val_audio, y_val,
            X_test_video, X_test_audio, y_test)

def train_multimodal_model(
    model,
    X_train_video,
    X_train_audio,
    y_train,
    X_val_video,
    X_val_audio,
    y_val,
    epochs=100,
    batch_size=32,
    model_save_path='models/'
):
    """Trains the dual-stream model.

    Args:
        model: Compiled Keras model.
        X_train_video, X_train_audio: Training features for video and audio.
        y_train: Training labels.
        X_val_video, X_val_audio: Validation features for video and audio.
        y_val: Validation labels.
        epochs: Number of training epochs.
        batch_size: Batch size.
        model_save_path: Directory to save model checkpoints

    Returns:
        Trained Keras model and training history.
    """
    # Create callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'multimodal_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Train the model
    history = model.fit(
        [X_train_video, X_train_audio],  # Input is a list of arrays for the two input branches
        y_train,
        validation_data=([X_val_video, X_val_audio], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history

def evaluate_multimodal_model(model, X_test_video, X_test_audio, y_test):
    """Evaluates the dual-stream model on the test set.

    Args:
        model: Trained Keras model.
        X_test_video, X_test_audio: Test features for video and audio.
        y_test: Test labels.
    """
    loss, accuracy, precision, recall = model.evaluate(
        [X_test_video, X_test_audio], y_test, verbose=1
    )
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1_score:.4f}")

    # Calculate and print confusion matrix
    y_pred = model.predict([X_test_video, X_test_audio])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

def main():
    # --- Configuration ---
    DATA_DIR = 'processed_features_test'  # Assuming data is in this directory
    DATASET_NAME = 'RAVDESS'
    NUM_CLASSES = 8  # Number of emotions in RAVDESS
    
    # --- Data Loading ---
    video_sequences, audio_sequences, labels, filenames = load_data(DATA_DIR, DATASET_NAME)
    
    if video_sequences is None or audio_sequences is None or len(video_sequences) == 0 or len(audio_sequences) == 0:
        logging.error("Data loading failed or returned empty sequences.")
        sys.exit(1)
    
    # Convert labels to categorical format (one-hot encoding)
    labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)
    
    # --- Dataset Creation ---
    dataset = create_multimodal_datasets(video_sequences, audio_sequences, labels_categorical)
    
    if dataset is None:
        logging.error("Failed to create datasets.")
        sys.exit(1)
    
    X_train_video, X_train_audio, y_train, X_val_video, X_val_audio, y_val, X_test_video, X_test_audio, y_test = dataset
    
    logging.info(f"Training set size: {len(X_train_video)}")
    logging.info(f"Validation set size: {len(X_val_video)}")
    logging.info(f"Test set size: {len(X_test_video)}")
    
    # --- Padding Video Sequences ---
    # Find the maximum sequence length in video sequences
    video_max_len = max(seq.shape[0] for seq in X_train_video)
    video_dim = X_train_video[0].shape[1]
    
    # Pad all video sequences
    X_train_video_padded = np.zeros((len(X_train_video), video_max_len, video_dim))
    X_val_video_padded = np.zeros((len(X_val_video), video_max_len, video_dim))
    X_test_video_padded = np.zeros((len(X_test_video), video_max_len, video_dim))
    
    for i, seq in enumerate(X_train_video):
        X_train_video_padded[i, :seq.shape[0]] = seq
    for i, seq in enumerate(X_val_video):
        X_val_video_padded[i, :seq.shape[0]] = seq
    for i, seq in enumerate(X_test_video):
        X_test_video_padded[i, :seq.shape[0]] = seq
    
    # --- Padding Audio Sequences ---
    # Find the maximum sequence length in audio sequences
    audio_max_len = max(seq.shape[0] for seq in X_train_audio)
    audio_dim = X_train_audio[0].shape[1]
    
    # Pad all audio sequences
    X_train_audio_padded = np.zeros((len(X_train_audio), audio_max_len, audio_dim))
    X_val_audio_padded = np.zeros((len(X_val_audio), audio_max_len, audio_dim))
    X_test_audio_padded = np.zeros((len(X_test_audio), audio_max_len, audio_dim))
    
    for i, seq in enumerate(X_train_audio):
        X_train_audio_padded[i, :seq.shape[0]] = seq
    for i, seq in enumerate(X_val_audio):
        X_val_audio_padded[i, :seq.shape[0]] = seq
    for i, seq in enumerate(X_test_audio):
        X_test_audio_padded[i, :seq.shape[0]] = seq
    
    # --- Model Definition ---
    video_input_shape = (video_max_len, video_dim)
    audio_input_shape = (audio_max_len, audio_dim)
    
    model = define_model(video_input_shape, audio_input_shape, NUM_CLASSES)
    
    # --- Model Compilation ---
    compile_model(model)
    model.summary()
    
    # --- Training ---
    model, history = train_multimodal_model(
        model, 
        X_train_video_padded, X_train_audio_padded, y_train,
        X_val_video_padded, X_val_audio_padded, y_val
    )
    
    # --- Evaluation ---
    evaluate_multimodal_model(model, X_test_video_padded, X_test_audio_padded, y_test)

if __name__ == "__main__":
    main()
