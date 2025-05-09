#!/usr/bin/env python3
"""
Training script for the multimodal emotion recognition model with branched LSTM architecture,
specifically adapted for single-segment (full video) processing rather than windowed segments.
"""

import os
import sys
import glob
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout, Bidirectional, Masking
from tensorflow.keras.layers import Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_branched_single_segment.log"),
        logging.StreamHandler()
    ]
)

def load_single_segment_data(data_dir):
    """Loads preprocessed data from single-segment NPZ files.

    Args:
        data_dir: Directory containing the preprocessed .npz files.

    Returns:
        Tuple: (video_features, audio_features, labels, filenames, valid_frames_masks)
               - video_features is a list of video feature arrays (each is frames x features)
               - audio_features is a list of audio feature arrays (each is frames x features)
               - labels is a list of corresponding emotion labels (numeric)
               - filenames is a list of the base filenames
               - valid_frames_masks is a list of boolean masks for valid video frames
    """
    video_features = []
    audio_features = []
    labels = []
    filenames = []
    valid_frames_masks = []

    file_pattern = os.path.join(data_dir, "*.npz")
    files = glob.glob(file_pattern)

    if not files:
        logging.error(f"No .npz files found in {data_dir}")
        return None, None, None, None, None

    for file_path in files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Check if this is a single-segment file
            is_single_segment = False
            if 'params' in data:
                params = data['params'].item()
                if isinstance(params, dict) and params.get('is_single_segment', False):
                    is_single_segment = True
            
            # Process single-segment file format
            if is_single_segment and 'video_features' in data and 'audio_features' in data:
                # Get video features
                video_feat = data['video_features']
                
                # Get audio features
                audio_feat = data['audio_features']
                
                # Get valid frames mask (or create one if not present)
                if 'valid_frames' in data:
                    valid_mask = data['valid_frames']
                else:
                    valid_mask = np.ones(len(video_feat), dtype=bool)
                
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # Get emotion label
                if 'emotion_label' in data:
                    emotion_label = data['emotion_label']
                    if isinstance(emotion_label, np.ndarray) and emotion_label.size == 1:
                        emotion_label = emotion_label.item()
                else:
                    # Try to parse from filename (format: 01-01-06-01-02-01-16)
                    # Third segment is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
                    parts = base_filename.split('-')
                    if len(parts) >= 3:
                        try:
                            emotion_code = int(parts[2])
                            # Convert to 0-based index
                            emotion_label = emotion_code - 1
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse emotion from filename {base_filename}")
                            continue
                    else:
                        logging.warning(f"Could not find emotion label for {file_path}, skipping.")
                        continue
                
                # Add data to our collections
                video_features.append(video_feat)
                audio_features.append(audio_feat)
                labels.append(emotion_label)
                filenames.append(base_filename)
                valid_frames_masks.append(valid_mask)
            
            else:
                # Legacy format or incompatible file
                logging.warning(f"File {file_path} is not in single-segment format or is missing required features. Skipping.")
                continue
                
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    logging.info(f"Loaded {len(video_features)} videos with valid features")
    
    # Log some statistics
    video_lengths = [v.shape[0] for v in video_features]
    audio_lengths = [a.shape[0] for a in audio_features]
    
    logging.info(f"Video frames per file: min={min(video_lengths)}, max={max(video_lengths)}, avg={np.mean(video_lengths):.1f}")
    logging.info(f"Audio frames per file: min={min(audio_lengths)}, max={max(audio_lengths)}, avg={np.mean(audio_lengths):.1f}")

    return video_features, audio_features, labels, filenames, valid_frames_masks

def define_branched_model(video_input_shape, audio_input_shape, num_classes, use_masking=True):
    """Defines the branched LSTM model architecture with attention mechanisms,
    adapted for variable-length single-segment input.

    Args:
        video_input_shape: Shape of the video input data (max_seq_length, feature_dim).
        audio_input_shape: Shape of the audio input data (max_seq_length, feature_dim).
        num_classes: Number of emotion classes.
        use_masking: Whether to use masking for invalid frames.

    Returns:
        Compiled Keras model.
    """
    # Define inputs for both streams
    video_input = Input(shape=video_input_shape, name='video_input')
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    
    # Optional masking layers to handle variable-length sequences and invalid frames
    if use_masking:
        # For video, we'll mask zeros (represents padding or invalid frames)
        video_masked = Masking(mask_value=0.0, name='video_masking')(video_input)
        # For audio, we'll mask zeros (represents padding)
        audio_masked = Masking(mask_value=0.0, name='audio_masking')(audio_input)
    else:
        video_masked = video_input
        audio_masked = audio_input

    # ---- Video Branch 1: Standard LSTM ----
    video_branch1 = LSTM(128, return_sequences=True, name='video_branch1_lstm1')(video_masked)
    video_branch1 = Dropout(0.3)(video_branch1)
    video_branch1 = LSTM(64, name='video_branch1_lstm2')(video_branch1)

    # ---- Video Branch 2: Bidirectional LSTM ----
    video_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='video_branch2_bilstm1')(video_masked)
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
    audio_branch1 = LSTM(128, return_sequences=True, name='audio_branch1_lstm1')(audio_masked)
    audio_branch1 = Dropout(0.3)(audio_branch1)
    audio_branch1 = LSTM(64, name='audio_branch1_lstm2')(audio_branch1)

    # ---- Audio Branch 2: Bidirectional LSTM ----
    audio_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='audio_branch2_bilstm1')(audio_masked)
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

    # ---- Audio Branch 3: LSTM with Attention ----
    # Using simpler LSTM branch instead of Conv1D to better handle variable length
    audio_branch3 = LSTM(128, return_sequences=True, name='audio_branch3_lstm1')(audio_masked)
    audio_branch3 = Dropout(0.3)(audio_branch3)
    audio_branch3 = LSTM(64, name='audio_branch3_lstm2')(audio_branch3)

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
        ]
    )

def create_dataset_split(video_features, audio_features, labels, filenames, valid_frames_masks, 
                        test_size=0.2, val_size=0.1, random_state=42):
    """Split the dataset into training, validation, and test sets.
    
    Args:
        video_features: List of video feature arrays.
        audio_features: List of audio feature arrays.
        labels: List of emotion labels.
        filenames: List of file names.
        valid_frames_masks: List of boolean masks for valid frames.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of training, validation, and test data.
    """
    # Convert labels to categorical format (one-hot encoding)
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)
    labels_categorical = to_categorical([unique_labels.index(l) for l in labels], num_classes=num_classes)
    
    # First split: training + validation vs. test
    train_val_indices, test_indices = train_test_split(
        range(len(video_features)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Stratify by emotion label to ensure balanced split
    )
    
    # Second split: training vs. validation
    # Calculate validation size as a proportion of the training + validation set
    val_proportion = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_proportion,
        random_state=random_state,
        stratify=[labels[i] for i in train_val_indices]  # Stratify by emotion label
    )
    
    # Create the split datasets
    X_train_video = [video_features[i] for i in train_indices]
    X_train_audio = [audio_features[i] for i in train_indices]
    y_train = labels_categorical[train_indices]
    train_filenames = [filenames[i] for i in train_indices]
    train_masks = [valid_frames_masks[i] for i in train_indices]
    
    X_val_video = [video_features[i] for i in val_indices]
    X_val_audio = [audio_features[i] for i in val_indices]
    y_val = labels_categorical[val_indices]
    val_filenames = [filenames[i] for i in val_indices]
    val_masks = [valid_frames_masks[i] for i in val_indices]
    
    X_test_video = [video_features[i] for i in test_indices]
    X_test_audio = [audio_features[i] for i in test_indices]
    y_test = labels_categorical[test_indices]
    test_filenames = [filenames[i] for i in test_indices]
    test_masks = [valid_frames_masks[i] for i in test_indices]
    
    logging.info(f"Split dataset: {len(X_train_video)} train, {len(X_val_video)} validation, {len(X_test_video)} test")
    
    # Report distribution of emotion labels in each split
    train_label_indices = np.argmax(y_train, axis=1)
    val_label_indices = np.argmax(y_val, axis=1)
    test_label_indices = np.argmax(y_test, axis=1)
    
    train_counts = {i: np.sum(train_label_indices == i) for i in range(num_classes)}
    val_counts = {i: np.sum(val_label_indices == i) for i in range(num_classes)}
    test_counts = {i: np.sum(test_label_indices == i) for i in range(num_classes)}
    
    logging.info(f"Train label distribution: {train_counts}")
    logging.info(f"Validation label distribution: {val_counts}")
    logging.info(f"Test label distribution: {test_counts}")
    
    return (X_train_video, X_train_audio, y_train, train_filenames, train_masks,
            X_val_video, X_val_audio, y_val, val_filenames, val_masks,
            X_test_video, X_test_audio, y_test, test_filenames, test_masks)

def pad_variable_length_sequences(sequences, max_length=None, mask=None):
    """Pad sequences to the same length.
    
    Args:
        sequences: List of NumPy arrays of varying lengths.
        max_length: Maximum sequence length. If None, use the longest sequence.
        mask: Optional mask to apply to the sequences (for handling invalid frames).
        
    Returns:
        Padded sequences as a single NumPy array.
    """
    if not sequences:
        return None
    
    # Determine feature dimension
    feature_dim = sequences[0].shape[1]
    
    # Determine max length if not provided
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    # Initialize padded array
    padded = np.zeros((len(sequences), max_length, feature_dim))
    
    # Fill with actual data
    for i, seq in enumerate(sequences):
        if mask is not None and i < len(mask):
            # Apply mask: zero out invalid frames
            for j in range(min(seq.shape[0], max_length)):
                if j < len(mask[i]) and mask[i][j]:
                    padded[i, j, :] = seq[j, :]
        else:
            # No mask, just copy the sequence
            length = min(seq.shape[0], max_length)
            padded[i, :length, :] = seq[:length, :]
    
    return padded

def prepare_data_for_training(video_features, audio_features, valid_frames_masks=None, 
                             max_video_length=None, max_audio_length=None):
    """Prepare data for training by padding the sequences.
    
    Args:
        video_features: List of video feature arrays.
        audio_features: List of audio feature arrays.
        valid_frames_masks: List of boolean masks for valid frames.
        max_video_length: Maximum length for video features (if None, compute from data).
        max_audio_length: Maximum length for audio features (if None, compute from data).
        
    Returns:
        Tuple of (padded_video, padded_audio).
    """
    # Get max lengths if not provided
    if max_video_length is None:
        max_video_length = max(v.shape[0] for v in video_features)
    if max_audio_length is None:
        max_audio_length = max(a.shape[0] for a in audio_features)
    
    logging.info(f"Padding video sequences to length {max_video_length}")
    logging.info(f"Padding audio sequences to length {max_audio_length}")
    
    # Pad video features, applying mask if provided
    padded_video = pad_variable_length_sequences(
        video_features, 
        max_length=max_video_length, 
        mask=valid_frames_masks
    )
    
    # Pad audio features (no masking needed for audio)
    padded_audio = pad_variable_length_sequences(
        audio_features,
        max_length=max_audio_length
    )
    
    return padded_video, padded_audio

def train_model(model, train_data, val_data, model_dir='models/branched_single_segment', 
                epochs=100, batch_size=32, class_names=None):
    """Train the model and evaluate it on the validation set.
    
    Args:
        model: Compiled Keras model.
        train_data: Tuple of (X_train_video, X_train_audio, y_train).
        val_data: Tuple of (X_val_video, X_val_audio, y_val).
        model_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        class_names: List of class names for reporting.
        
    Returns:
        Tuple of (trained_model, history).
    """
    X_train_video, X_train_audio, y_train = train_data
    X_val_video, X_val_audio, y_val = val_data
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        [X_train_video, X_train_audio],
        y_train,
        validation_data=([X_val_video, X_val_audio], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    try:
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
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
        history_path = os.path.join(plots_dir, 'training_history.png')
        plt.savefig(history_path)
        plt.close()
        logging.info(f"Training history plot saved to {history_path}")
    except Exception as e:
        logging.warning(f"Could not create training history plot: {str(e)}")
    
    return model, history

def evaluate_model(model, test_data, class_names=None, output_dir='model_evaluation/branched_single_segment'):
    """Evaluate the model on the test set.
    
    Args:
        model: Trained Keras model.
        test_data: Tuple of (X_test_video, X_test_audio, y_test).
        class_names: List of class names for reporting.
        output_dir: Directory to save evaluation results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    X_test_video, X_test_audio, y_test = test_data
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the model
    logging.info("Evaluating model on test set...")
    metrics = model.evaluate([X_test_video, X_test_audio], y_test, verbose=1)
    
    # Get metric names
    metric_names = model.metrics_names
    
    # Create dictionary of metrics
    metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
    
    # Calculate F1 score if precision and recall are available
    if 'precision' in metrics_dict and 'recall' in metrics_dict:
        precision = metrics_dict['precision']
        recall = metrics_dict['recall']
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics_dict['f1'] = f1
    
    # Print metrics
    for name, value in metrics_dict.items():
        logging.info(f"Test {name}: {value:.4f}")
    
    # Save metrics to file
    metrics_text = "\n".join([f"{name}: {value:.4f}" for name, value in metrics_dict.items()])
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(metrics_text)
    
    # Generate predictions
    y_pred = model.predict([X_test_video, X_test_audio])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate confusion matrix and classification report
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Generate classification report
        if class_names:
            report = classification_report(y_true_classes, y_pred_classes, 
                                          target_names=class_names, zero_division=0)
        else:
            report = classification_report(y_true_classes, y_pred_classes, zero_division=0)
        
        # Save classification report
        report_file = os.path.join(output_dir, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Plot confusion matrix
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            
            if class_names:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            cm_file = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_file)
            plt.close()
        except Exception as e:
            logging.warning(f"Could not create confusion matrix plot: {str(e)}")
    
    except Exception as e:
        logging.warning(f"Could not generate classification metrics: {str(e)}")
    
    return metrics_dict

def main():
    """Main function to train and evaluate the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a branched LSTM model on single-segment data')
    parser.add_argument('--data_dir', type=str, default='processed_ravdess_single_segment',
                       help='Directory containing processed single-segment data')
    parser.add_argument('--model_dir', type=str, default='models/branched_single_segment',
                       help='Directory to save model checkpoints')
    parser.add_argument('--eval_dir', type=str, default='model_evaluation/branched_single_segment',
                       help='Directory to save evaluation results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--use_masking', action='store_true',
                       help='Use masking for invalid frames')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # RAVDESS emotion classes
    CLASS_NAMES = [
        'neutral', 'calm', 'happy', 'sad',
        'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    # Load data
    print("Loading data...")
    video_features, audio_features, labels, filenames, valid_frames_masks = load_single_segment_data(args.data_dir)
    
    if video_features is None or len(video_features) == 0:
        logging.error("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Split data into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    dataset_split = create_dataset_split(
        video_features, audio_features, labels, filenames, valid_frames_masks,
        test_size=0.2, val_size=0.1, random_state=args.random_seed
    )
    
    (X_train_video, X_train_audio, y_train, train_filenames, train_masks,
     X_val_video, X_val_audio, y_val, val_filenames, val_masks,
     X_test_video, X_test_audio, y_test, test_filenames, test_masks) = dataset_split
    
    # Compute the global maximum lengths for video and audio across all splits
    print("Preparing data for training...")
    global_max_video_length = max(
        max(len(seq) for seq in X_train_video),
        max(len(seq) for seq in X_val_video),
        max(len(seq) for seq in X_test_video)
    )
    global_max_audio_length = max(
        max(len(seq) for seq in X_train_audio),
        max(len(seq) for seq in X_val_audio),
        max(len(seq) for seq in X_test_audio)
    )
    
    logging.info(f"Global maximum video length: {global_max_video_length}")
    logging.info(f"Global maximum audio length: {global_max_audio_length}")
    
    # Pad all splits with the same global maximum lengths
    X_train_video_padded, X_train_audio_padded = prepare_data_for_training(
        X_train_video, X_train_audio, train_masks,
        max_video_length=global_max_video_length,
        max_audio_length=global_max_audio_length
    )
    X_val_video_padded, X_val_audio_padded = prepare_data_for_training(
        X_val_video, X_val_audio, val_masks,
        max_video_length=global_max_video_length,
        max_audio_length=global_max_audio_length
    )
    X_test_video_padded, X_test_audio_padded = prepare_data_for_training(
        X_test_video, X_test_audio, test_masks,
        max_video_length=global_max_video_length,
        max_audio_length=global_max_audio_length
    )
    
    # Define model architecture
    print("Defining model architecture...")
    video_input_shape = X_train_video_padded.shape[1:]
    audio_input_shape = X_train_audio_padded.shape[1:]
    num_classes = y_train.shape[1]
    
    logging.info(f"Video input shape: {video_input_shape}")
    logging.info(f"Audio input shape: {audio_input_shape}")
    logging.info(f"Number of classes: {num_classes}")
    
    model = define_branched_model(
        video_input_shape, audio_input_shape, num_classes,
        use_masking=args.use_masking
    )
    
    # Compile the model
    compile_model(model, learning_rate=args.learning_rate)
    model.summary()
    
    # Train the model
    print("Training model...")
    model, history = train_model(
        model,
        (X_train_video_padded, X_train_audio_padded, y_train),
        (X_val_video_padded, X_val_audio_padded, y_val),
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_names=CLASS_NAMES
    )
    
    # Evaluate the model on the test set
    print("Evaluating model...")
    metrics = evaluate_model(
        model,
        (X_test_video_padded, X_test_audio_padded, y_test),
        class_names=CLASS_NAMES,
        output_dir=args.eval_dir
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
