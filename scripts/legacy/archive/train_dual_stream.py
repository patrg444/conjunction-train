#!/usr/bin/env python3
"""
Training script for the dual-stream LSTM model for multimodal emotion recognition.
This implements the exact architecture specified in the design document:
- Separate LSTM branches for video and audio
- Early fusion through concatenation
- MLP layers post-fusion
- Output layer for emotion classification
"""

import os
import sys
import glob
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from synchronize_test import parse_ravdess_filename
from visualize_model import plot_training_history, visualize_model_architecture, visualize_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_dual_stream.log"),
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
                    
                    # Handle scalar emotion label (single integer)
                    if np.isscalar(emotion_labels) or (isinstance(emotion_labels, np.ndarray) and emotion_labels.size == 1):
                        # Convert scalar to array with the value repeated for each sequence
                        emotion_label_value = emotion_labels.item() if isinstance(emotion_labels, np.ndarray) else emotion_labels
                        emotion_labels = np.array([emotion_label_value] * len(video_seqs))
                    elif not isinstance(emotion_labels, np.ndarray):
                        # Convert other types to numpy array
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

def define_dual_stream_model(video_input_shape, audio_input_shape, num_classes):
    """Defines the dual-stream LSTM model architecture.

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
    
    # --- Video Stream ---
    # Using a stacked LSTM architecture for video
    video_lstm1 = LSTM(128, return_sequences=True, name='video_lstm1')(video_input)
    video_lstm1 = Dropout(0.3, name='video_dropout1')(video_lstm1)
    video_lstm2 = LSTM(64, return_sequences=False, name='video_lstm2')(video_lstm1)
    video_lstm2 = Dropout(0.3, name='video_dropout2')(video_lstm2)
    
    # --- Audio Stream ---
    # Using a stacked LSTM architecture for audio
    audio_lstm1 = LSTM(128, return_sequences=True, name='audio_lstm1')(audio_input)
    audio_lstm1 = Dropout(0.3, name='audio_dropout1')(audio_lstm1)
    audio_lstm2 = LSTM(64, return_sequences=False, name='audio_lstm2')(audio_lstm1)
    audio_lstm2 = Dropout(0.3, name='audio_dropout2')(audio_lstm2)
    
    # --- Early Fusion (Concatenation) ---
    fusion = concatenate([video_lstm2, audio_lstm2], name='fusion')
    
    # --- MLP Post-Fusion ---
    dense1 = Dense(128, activation='relu', name='dense1')(fusion)
    dropout1 = Dropout(0.5, name='dropout_dense1')(dense1)
    dense2 = Dense(64, activation='relu', name='dense2')(dropout1)
    dropout2 = Dropout(0.3, name='dropout_dense2')(dense2)
    
    # --- Output Layer ---
    # For multi-class emotion classification, use softmax activation
    output_layer = Dense(num_classes, activation='softmax', name='output')(dropout2)
    
    # Create the model
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
            tf.keras.metrics.Recall(name='recall')
        ]
    )

def create_multimodal_datasets(video_sequences, audio_sequences, labels, filenames, test_size=0.2, val_size=0.1, random_state=42):
    """Creates training, validation, and test datasets for multimodal data with video-level separation.

    Args:
        video_sequences: List of video feature sequences.
        audio_sequences: List of audio feature sequences.
        labels: List of corresponding labels.
        filenames: List of source video filenames for each segment.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing training, validation, and test data and labels.
    """
    if len(video_sequences) == 0 or len(audio_sequences) == 0:
        logging.error("Empty video or audio sequences provided.")
        return None
    
    if len(video_sequences) != len(audio_sequences) or len(video_sequences) != len(labels) or len(video_sequences) != len(filenames):
        logging.error(f"Mismatched data lengths: {len(video_sequences)} video, {len(audio_sequences)} audio, "
                     f"{len(labels)} labels, {len(filenames)} filenames")
        return None

    # Convert labels to numpy array
    labels_array = np.array(labels)
    
    # Group segments by source video
    unique_videos = {}  # Dictionary: video_name -> list of indices
    for i, fname in enumerate(filenames):
        if fname not in unique_videos:
            unique_videos[fname] = []
        unique_videos[fname].append(i)
    
    logging.info(f"Data contains segments from {len(unique_videos)} unique videos")
    
    # Special handling for very small datasets
    if len(unique_videos) < 3:
        logging.warning(f"Not enough unique videos ({len(unique_videos)}) for proper train/val/test split.")
        logging.warning("Using all data for all splits to continue with testing.")
        
        # Use all data for all splits
        all_indices = list(range(len(video_sequences)))
        return (
            [video_sequences[i] for i in all_indices],
            [audio_sequences[i] for i in all_indices],
            labels_array[all_indices],
            [video_sequences[i] for i in all_indices],
            [audio_sequences[i] for i in all_indices],
            labels_array[all_indices],
            [video_sequences[i] for i in all_indices],
            [audio_sequences[i] for i in all_indices],
            labels_array[all_indices]
        )
    
    # Get list of unique video names
    video_names = list(unique_videos.keys())
    
    # Random state for reproducibility
    np.random.seed(random_state)
    np.random.shuffle(video_names)
    
    # Calculate split points
    n_videos = len(video_names)
    n_test = max(1, int(n_videos * test_size))
    n_val = max(1, int(n_videos * val_size))
    n_train = n_videos - n_test - n_val
    
    # Split at the video level
    train_videos = video_names[:n_train]
    val_videos = video_names[n_train:n_train+n_val]
    test_videos = video_names[n_train+n_val:]
    
    logging.info(f"Video-level split: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
    
    # Collect indices for each split from the grouped videos
    train_idx = []
    for v in train_videos:
        train_idx.extend(unique_videos[v])
    
    val_idx = []
    for v in val_videos:
        val_idx.extend(unique_videos[v])
    
    test_idx = []
    for v in test_videos:
        test_idx.extend(unique_videos[v])
    
    logging.info(f"Segment counts: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Verify no overlap between sets
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    if len(train_set.intersection(val_set)) > 0 or len(train_set.intersection(test_set)) > 0 or len(val_set.intersection(test_set)) > 0:
        logging.error("Data leakage detected! Segments appear in multiple splits.")
    else:
        logging.info("No data leakage detected. Train/val/test splits are properly separated.")
        
    # Report distribution of emotion labels in each split
    if len(set(labels)) <= 10:  # Only for small number of classes
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        test_labels = [labels[i] for i in test_idx]
        
        for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
            label_counts = {}
            for lbl in split_labels:
                if lbl not in label_counts:
                    label_counts[lbl] = 0
                label_counts[lbl] += 1
            logging.info(f"{split_name} label distribution: {label_counts}")
    
    # Training data - keep as lists to handle variable lengths
    X_train_video = [video_sequences[i] for i in train_idx]
    X_train_audio = [audio_sequences[i] for i in train_idx]
    y_train = labels_array[train_idx]
    
    # Validation data
    X_val_video = [video_sequences[i] for i in val_idx]
    X_val_audio = [audio_sequences[i] for i in val_idx]
    y_val = labels_array[val_idx]
    
    # Test data
    X_test_video = [video_sequences[i] for i in test_idx]
    X_test_audio = [audio_sequences[i] for i in test_idx]
    y_test = labels_array[test_idx]
    
    return (X_train_video, X_train_audio, y_train, 
            X_val_video, X_val_audio, y_val,
            X_test_video, X_test_audio, y_test)

def train_model(
    model,
    X_train_video,
    X_train_audio,
    y_train,
    X_val_video,
    X_val_audio,
    y_val,
    epochs=100,
    batch_size=32,
    model_save_path='models/dual_stream/'
):
    """Trains the dual-stream LSTM model.

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
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
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
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    # Train the model
    history = model.fit(
        [X_train_video, X_train_audio],
        y_train,
        validation_data=([X_val_video, X_val_audio], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test_video, X_test_audio, y_test, class_names, output_dir="model_evaluation"):
    """Evaluates the dual-stream LSTM model on the test set.

    Args:
        model: Trained Keras model.
        X_test_video, X_test_audio: Test features for video and audio.
        y_test: Test labels.
        class_names: Names of the emotion classes.
        output_dir: Directory to save evaluation results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the model
    loss, accuracy, precision, recall = model.evaluate(
        [X_test_video, X_test_audio], y_test, verbose=1
    )
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print and save evaluation metrics
    evaluation_text = (
        f"Test Loss: {loss:.4f}\n"
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Test Precision: {precision:.4f}\n"
        f"Test Recall: {recall:.4f}\n"
        f"Test F1-score: {f1_score:.4f}\n"
    )
    
    print(evaluation_text)
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(evaluation_text)
    
    # Calculate and print confusion matrix
    y_pred = model.predict([X_test_video, X_test_audio])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Try to save confusion matrix visualization, but continue if it fails
    try:
        logging.info("Attempting to visualize confusion matrix...")
        visualize_confusion_matrix(y_test, y_pred, class_names, 
                                  output_file=os.path.join(output_dir, "confusion_matrix.png"))
        logging.info("Confusion matrix visualization saved.")
    except Exception as e:
        logging.warning(f"Could not visualize confusion matrix: {str(e)}")
        logging.warning("Continuing without confusion matrix visualization.")
    
    # Print and save classification report
    try:
        # Get the unique classes in the test set
        unique_classes = sorted(set(y_true_classes) | set(y_pred_classes))
        if len(unique_classes) < len(class_names):
            logging.warning(f"Only {len(unique_classes)} out of {len(class_names)} classes present in test set")
            
            # Filter class names to only include those present in the data
            present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
            
            # Generate report with only the present classes
            report = classification_report(
                y_true_classes, y_pred_classes, 
                labels=unique_classes,
                target_names=present_class_names,
                zero_division=0
            )
        else:
            report = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=class_names,
                zero_division=0
            )
            
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
    except Exception as e:
        logging.warning(f"Could not generate classification report: {str(e)}")

def main():
    """Main function to train and evaluate the dual-stream LSTM model."""
    
    # --- Configuration ---
    DATA_DIR = 'processed_features'  # Directory containing processed features
    DATASET_NAME = 'RAVDESS'  # Dataset name ('RAVDESS' or 'CREMA-D')
    NUM_CLASSES = 8  # Number of emotion classes
    EPOCHS = 2  # Number of training epochs (reduced for testing)
    BATCH_SIZE = 2  # Batch size (reduced for testing)
    MODEL_SAVE_PATH = 'models/dual_stream/'  # Directory to save model checkpoints
    EVALUATION_DIR = 'model_evaluation/dual_stream/'  # Directory to save evaluation results
    TEST_SIZE = 0.2  # Proportion of data for testing
    VAL_SIZE = 0.1  # Proportion of data for validation
    
    # RAVDESS emotion classes
    CLASS_NAMES = [
        'neutral', 'calm', 'happy', 'sad', 
        'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    # --- Data Loading ---
    logging.info("Loading data...")
    video_sequences, audio_sequences, labels, filenames = load_data(DATA_DIR, DATASET_NAME)
    
    if video_sequences is None or audio_sequences is None or len(video_sequences) == 0 or len(audio_sequences) == 0:
        logging.error("Data loading failed or returned empty sequences.")
        sys.exit(1)
    
    # Convert labels to categorical format (one-hot encoding)
    labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)
    
    # --- Dataset Creation ---
    logging.info("Creating datasets...")
    dataset = create_multimodal_datasets(
        video_sequences, audio_sequences, labels_categorical, filenames,
        test_size=TEST_SIZE, val_size=VAL_SIZE
    )
    
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
    
    logging.info(f"Video sequence max length: {video_max_len}, dimension: {video_dim}")
    
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
    
    # Check if all audio sequences have the same feature dimension
    audio_dims = [seq.shape[1] for seq in X_train_audio + X_val_audio + X_test_audio]
    if len(set(audio_dims)) == 1:
        # All sequences have the same dimension
        audio_dim = audio_dims[0]
        logging.info(f"Audio sequence max length: {audio_max_len}, consistent dimension: {audio_dim}")
        
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
    else:
        # Variable dimensions - we'll need to standardize
        logging.info(f"Audio sequences have variable dimensions: {set(audio_dims)}")
        
        # Use the most common dimension
        from collections import Counter
        most_common_dim = Counter(audio_dims).most_common(1)[0][0]
        logging.info(f"Using most common dimension: {most_common_dim}")
        
        # Initialize padded arrays with zeros
        X_train_audio_padded = np.zeros((len(X_train_audio), audio_max_len, most_common_dim))
        X_val_audio_padded = np.zeros((len(X_val_audio), audio_max_len, most_common_dim))
        X_test_audio_padded = np.zeros((len(X_test_audio), audio_max_len, most_common_dim))
        
        # Copy and potentially reshape data
        for i, seq in enumerate(X_train_audio):
            if seq.shape[1] == most_common_dim:
                X_train_audio_padded[i, :seq.shape[0]] = seq
            else:
                logging.warning(f"Training sequence {i} has dimension {seq.shape[1]}, truncating/padding to {most_common_dim}")
                for j in range(min(seq.shape[0], audio_max_len)):
                    X_train_audio_padded[i, j, :min(seq.shape[1], most_common_dim)] = seq[j, :min(seq.shape[1], most_common_dim)]
        
        for i, seq in enumerate(X_val_audio):
            if seq.shape[1] == most_common_dim:
                X_val_audio_padded[i, :seq.shape[0]] = seq
            else:
                logging.warning(f"Validation sequence {i} has dimension {seq.shape[1]}, truncating/padding to {most_common_dim}")
                for j in range(min(seq.shape[0], audio_max_len)):
                    X_val_audio_padded[i, j, :min(seq.shape[1], most_common_dim)] = seq[j, :min(seq.shape[1], most_common_dim)]
        
        for i, seq in enumerate(X_test_audio):
            if seq.shape[1] == most_common_dim:
                X_test_audio_padded[i, :seq.shape[0]] = seq
            else:
                logging.warning(f"Test sequence {i} has dimension {seq.shape[1]}, truncating/padding to {most_common_dim}")
                for j in range(min(seq.shape[0], audio_max_len)):
                    X_test_audio_padded[i, j, :min(seq.shape[1], most_common_dim)] = seq[j, :min(seq.shape[1], most_common_dim)]
    
    # Update audio_dim for the model
    audio_dim = X_train_audio_padded.shape[2]
    logging.info(f"Final audio feature dimension: {audio_dim}")
    
    # --- Model Definition ---
    logging.info("Defining the dual-stream LSTM model...")
    video_input_shape = (video_max_len, video_dim)
    audio_input_shape = (audio_max_len, audio_dim)
    
    model = define_dual_stream_model(video_input_shape, audio_input_shape, NUM_CLASSES)
    
    # --- Model Compilation ---
    logging.info("Compiling the model...")
    compile_model(model, learning_rate=0.001)
    model.summary()
    
    # Create evaluation directory
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    # Try to visualize model architecture, but continue if it fails
    try:
        logging.info("Attempting to visualize model architecture...")
        visualize_model_architecture(model, output_file=os.path.join(EVALUATION_DIR, "model_architecture.png"))
        logging.info("Model architecture visualization saved.")
    except ImportError as e:
        logging.warning(f"Could not visualize model architecture: {str(e)}")
        logging.warning("Continuing without visualization. To enable visualization, install pydot and graphviz.")
    except Exception as e:
        logging.warning(f"Could not visualize model architecture: {str(e)}")
        logging.warning("Continuing without visualization.")
    
    # --- Training ---
    logging.info("Training the model...")
    model, history = train_model(
        model, 
        X_train_video_padded, X_train_audio_padded, y_train,
        X_val_video_padded, X_val_audio_padded, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Try to plot training history, but continue if it fails
    try:
        logging.info("Attempting to plot training history...")
        plot_training_history(history, output_file=os.path.join(EVALUATION_DIR, "training_history.png"))
        logging.info("Training history plot saved.")
    except Exception as e:
        logging.warning(f"Could not plot training history: {str(e)}")
        logging.warning("Continuing without training history visualization.")
    
    # --- Evaluation ---
    logging.info("Evaluating the model...")
    evaluate_model(
        model, 
        X_test_video_padded, X_test_audio_padded, y_test,
        CLASS_NAMES,
        output_dir=EVALUATION_DIR
    )
    
    # Save the final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.h5')
    model.save(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    
    logging.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
