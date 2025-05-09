#!/usr/bin/env python3
"""
Training script for the multimodal emotion recognition model with branched LSTM architecture.
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
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from synchronize_test import parse_ravdess_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_branched.log"),
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

def define_branched_model(video_input_shape, audio_input_shape, num_classes):
    """Defines the branched LSTM model architecture with attention mechanisms.

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
    
    # ---- Video Branch 1: Standard LSTM ----
    video_branch1 = LSTM(128, return_sequences=True, name='video_branch1_lstm1')(video_input)
    video_branch1 = Dropout(0.3)(video_branch1)
    video_branch1 = LSTM(64, name='video_branch1_lstm2')(video_branch1)
    
    # ---- Video Branch 2: Bidirectional LSTM ----
    video_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='video_branch2_bilstm1')(video_input)
    video_branch2_seq = Dropout(0.3)(video_branch2_seq)
    
    # Add self-attention to video branch 2
    try:
        # Using TensorFlow 2.x MultiHeadAttention if available
        video_branch2_attn = MultiHeadAttention(
            num_heads=4, key_dim=32, name="video_self_attention"
        )(video_branch2_seq, video_branch2_seq)
        video_branch2_seq = video_branch2_attn + video_branch2_seq  # Residual connection
    except Exception as e:
        logging.warning(f"Could not use MultiHeadAttention: {str(e)}. Using regular LSTM instead.")
        pass  # Fall back to regular processing if MultiHeadAttention fails

    video_branch2 = Bidirectional(LSTM(64), name='video_branch2_bilstm2')(video_branch2_seq)
    
    # ---- Audio Branch 1: Standard LSTM ----
    audio_branch1 = LSTM(128, return_sequences=True, name='audio_branch1_lstm1')(audio_input)
    audio_branch1 = Dropout(0.3)(audio_branch1)
    audio_branch1 = LSTM(64, name='audio_branch1_lstm2')(audio_branch1)
    
    # ---- Audio Branch 2: Bidirectional LSTM ----
    audio_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='audio_branch2_bilstm1')(audio_input)
    audio_branch2_seq = Dropout(0.3)(audio_branch2_seq)
    
    # Add self-attention to audio branch 2
    try:
        # Using TensorFlow 2.x MultiHeadAttention if available
        audio_branch2_attn = MultiHeadAttention(
            num_heads=4, key_dim=32, name="audio_self_attention"
        )(audio_branch2_seq, audio_branch2_seq)
        audio_branch2_seq = audio_branch2_attn + audio_branch2_seq  # Residual connection
    except Exception as e:
        logging.warning(f"Could not use MultiHeadAttention: {str(e)}. Using regular LSTM instead.")
        pass  # Fall back to regular processing if MultiHeadAttention fails
        
    audio_branch2 = Bidirectional(LSTM(64), name='audio_branch2_bilstm2')(audio_branch2_seq)
    
    # ---- Audio Branch 3: Conv1D + LSTM ----
    audio_branch3 = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='audio_branch3_conv1')(audio_input)
    audio_branch3 = MaxPooling1D(pool_size=2, name='audio_branch3_pool1')(audio_branch3)
    audio_branch3 = Dropout(0.3)(audio_branch3)
    audio_branch3 = LSTM(64, name='audio_branch3_lstm')(audio_branch3)
    
    # ---- Video modality fusion ----
    video_fusion = concatenate([video_branch1, video_branch2], name='video_fusion')
    video_fusion = Dense(128, activation='relu', name='video_fusion_dense')(video_fusion)
    video_fusion = Dropout(0.4)(video_fusion)
    
    # ---- Audio modality fusion ----
    audio_fusion = concatenate([audio_branch1, audio_branch2, audio_branch3], name='audio_fusion')
    audio_fusion = Dense(128, activation='relu', name='audio_fusion_dense')(audio_fusion)
    audio_fusion = Dropout(0.4)(audio_fusion)
    
    # ---- Cross-modal fusion with attention (instead of simple concatenation) ----
    try:
        # Reshape for compatibility with attention mechanism
        video_fusion_reshaped = tf.expand_dims(video_fusion, axis=1)  # [batch, 1, features]
        audio_fusion_reshaped = tf.expand_dims(audio_fusion, axis=1)  # [batch, 1, features]
        
        # Cross-modal attention (video attending to audio)
        video_attends_audio = Attention(name='video_audio_attention')([video_fusion_reshaped, audio_fusion_reshaped])
        video_attends_audio = tf.squeeze(video_attends_audio, axis=1)
        
        # Cross-modal attention (audio attending to video)
        audio_attends_video = Attention(name='audio_video_attention')([audio_fusion_reshaped, video_fusion_reshaped])
        audio_attends_video = tf.squeeze(audio_attends_video, axis=1)
        
        # Weighted combination with original features
        video_with_context = concatenate([video_fusion, audio_attends_video], name='video_with_audio_context')
        audio_with_context = concatenate([audio_fusion, video_attends_audio], name='audio_with_video_context')
        
        # Final multimodal fusion
        cross_modal_fusion = concatenate([video_with_context, audio_with_context], name='cross_modal_fusion')
    except Exception as e:
        logging.warning(f"Could not use cross-modal attention: {str(e)}. Falling back to simple concatenation.")
        # Fallback to standard concatenation if attention fails
        cross_modal_fusion = concatenate([video_fusion, audio_fusion], name='cross_modal_fusion_fallback')
    
    # ---- Final classification layers ----
    dense1 = Dense(256, activation='relu', name='dense1')(cross_modal_fusion)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(128, activation='relu', name='dense2')(dropout)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax', name='output')(dropout2)
    
    # Create the branched model
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

def create_multimodal_datasets(video_sequences, audio_sequences, labels, filenames, test_size=0.1, val_size=0.1, random_state=42):
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
    # Convert to scalar integer labels if one-hot encoded
    scalar_labels = []
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # Convert one-hot encoded labels to scalar
        scalar_labels = np.argmax(labels, axis=1)
    else:
        # Already scalar labels
        scalar_labels = labels.flatten() if hasattr(labels, 'flatten') else np.array(labels)
    
    unique_labels = set(scalar_labels.tolist())
    if len(unique_labels) <= 10:  # Only for small number of classes
        train_labels = [scalar_labels[i] for i in train_idx]
        val_labels = [scalar_labels[i] for i in val_idx]
        test_labels = [scalar_labels[i] for i in test_idx]
        
        for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
            label_counts = {}
            for lbl in split_labels:
                if lbl not in label_counts:
                    label_counts[lbl] = 0
                label_counts[lbl] += 1
            logging.info(f"{split_name} label distribution: {label_counts}")
    
    # Create the split datasets
    X_train_video = [video_sequences[i] for i in train_idx]
    X_train_audio = [audio_sequences[i] for i in train_idx]
    y_train = labels_array[train_idx]
    
    X_val_video = [video_sequences[i] for i in val_idx]
    X_val_audio = [audio_sequences[i] for i in val_idx]
    y_val = labels_array[val_idx]
    
    X_test_video = [video_sequences[i] for i in test_idx]
    X_test_audio = [audio_sequences[i] for i in test_idx]
    y_test = labels_array[test_idx]
    
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
    model_save_path='models/branched/'
):
    """Trains the branched LSTM model.

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
        filepath=os.path.join(model_save_path, 'branched_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

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

def evaluate_multimodal_model(model, X_test_video, X_test_audio, y_test, class_names=None, output_dir="model_evaluation/branched"):
    """Evaluates the branched LSTM model on the test set with enhanced visualization.

    Args:
        model: Trained Keras model.
        X_test_video, X_test_audio: Test features for video and audio.
        y_test: Test labels.
        class_names: Names of the emotion classes (optional).
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
    
    # Try to save confusion matrix visualization if possible
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        # If class names are provided, use them
        if class_names:
            # Get the unique classes in the test set
            unique_classes = sorted(set(y_true_classes) | set(y_pred_classes))
            if len(unique_classes) < len(class_names):
                # Filter class names to only include those present
                present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=present_class_names, yticklabels=present_class_names)
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save the confusion matrix visualization
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        logging.info(f"Saved confusion matrix visualization to {confusion_matrix_path}")
    except Exception as e:
        logging.warning(f"Could not create confusion matrix visualization: {str(e)}")
    
    # Generate classification report
    try:
        # Get the unique classes in the test set
        unique_classes = sorted(set(y_true_classes) | set(y_pred_classes))
        
        if class_names and len(unique_classes) < len(class_names):
            # Filter class names to only include those present in the data
            present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
            
            # Generate report with only the present classes
            report = classification_report(
                y_true_classes, y_pred_classes, 
                labels=unique_classes,
                target_names=present_class_names,
                zero_division=0
            )
        elif class_names:
            report = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=class_names,
                zero_division=0
            )
        else:
            report = classification_report(
                y_true_classes, y_pred_classes,
                zero_division=0
            )
            
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
    except Exception as e:
        logging.warning(f"Could not generate classification report: {str(e)}")

def main():
    """Main function to train and evaluate the branched LSTM model."""
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train the branched LSTM model for emotion recognition')
    parser.add_argument('--data_dir', type=str, default='processed_features',
                        help='Directory containing processed features (default: processed_features)')
    parser.add_argument('--dataset', type=str, default='RAVDESS',
                        help='Dataset name: RAVDESS or CREMA-D (default: RAVDESS)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--model_dir', type=str, default='models/branched/',
                        help='Directory to save model checkpoints (default: models/branched/)')
    parser.add_argument('--eval_dir', type=str, default='model_evaluation/branched/',
                        help='Directory to save evaluation results (default: model_evaluation/branched/)')
    
    args = parser.parse_args()
    
    # --- Configuration ---
    DATA_DIR = args.data_dir
    DATASET_NAME = args.dataset
    NUM_CLASSES = 8  # Number of emotions in RAVDESS
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_SAVE_PATH = args.model_dir  # Directory to save model checkpoints
    EVALUATION_DIR = args.eval_dir  # Directory to save evaluation results
    
    # RAVDESS emotion classes
    CLASS_NAMES = [
        'neutral', 'calm', 'happy', 'sad', 
        'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    # --- Data Loading ---
    video_sequences, audio_sequences, labels, filenames = load_data(DATA_DIR, DATASET_NAME)
    
    if video_sequences is None or audio_sequences is None or len(video_sequences) == 0 or len(audio_sequences) == 0:
        logging.error("Data loading failed or returned empty sequences.")
        sys.exit(1)
    
    # Convert labels to categorical format (one-hot encoding)
    labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)
    
    # --- Dataset Creation ---
    dataset = create_multimodal_datasets(video_sequences, audio_sequences, labels_categorical, filenames)
    
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
    video_input_shape = (video_max_len, video_dim)
    audio_input_shape = (audio_max_len, audio_dim)
    
    try:
        model = define_branched_model(video_input_shape, audio_input_shape, NUM_CLASSES)
        
        # --- Model Compilation ---
        compile_model(model, learning_rate=0.001)
        model.summary()
        
        # Create directories
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        
        # Visualize model architecture if possible
        try:
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file=os.path.join(EVALUATION_DIR, 'model_architecture.png'), 
                      show_shapes=True, show_layer_names=True)
            logging.info(f"Model architecture visualization saved to {EVALUATION_DIR}/model_architecture.png")
        except Exception as e:
            logging.warning(f"Could not visualize model architecture: {str(e)}")
            logging.warning("Continuing without visualization. To enable, install pydot and graphviz.")
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    # --- Training ---
    try:
        model, history = train_multimodal_model(
            model, 
            X_train_video_padded, X_train_audio_padded, y_train,
            X_val_video_padded, X_val_audio_padded, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_path=MODEL_SAVE_PATH
        )
        
        # Plot training history if possible
        try:
            import matplotlib.pyplot as plt
            
            # Plot accuracy
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            history_path = os.path.join(EVALUATION_DIR, 'training_history.png')
            plt.savefig(history_path)
            plt.close()
            logging.info(f"Training history plot saved to {history_path}")
        except Exception as e:
            logging.warning(f"Could not create training history plot: {str(e)}")
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    # --- Evaluation ---
    try:
        evaluate_multimodal_model(
            model, 
            X_test_video_padded, X_test_audio_padded, y_test,
            class_names=CLASS_NAMES,
            output_dir=EVALUATION_DIR
        )
        
        # Save the trained model
        final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_branched_model.h5')
        model.save(final_model_path)
        logging.info(f"Final model saved to '{final_model_path}'")
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
