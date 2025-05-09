#!/usr/bin/env python3
"""
Training script for the multimodal emotion recognition model with branched LSTM architecture
using a combination of RAVDESS and CREMA-D datasets with a unified 6-class emotion schema.

The 6 classes are:
0: Neutral/Calm (RAVDESS Neutral+Calm, CREMA-D Neutral)
1: Happy (RAVDESS Happy, CREMA-D Happy)
2: Sad (RAVDESS Sad, CREMA-D Sad)
3: Angry (RAVDESS Angry, CREMA-D Angry)
4: Fearful (RAVDESS Fearful, CREMA-D Fear)
5: Disgust (RAVDESS Disgust, CREMA-D Disgust)

NOTE: This version EXCLUDES the "Surprised" class (RAVDESS emotion 7)

IMPROVEMENTS:
- Added F1 score metric for better performance evaluation
- Increased dropout rates to combat overfitting
- Added L2 regularization to dense layers
- Implemented cosine annealing learning rate schedule
- Enhanced evaluation with misclassification visualization
"""

import os
import sys
import glob
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout, Bidirectional, Masking
from tensorflow.keras.layers import Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import math
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_branched_6class_improved.log"),
        logging.StreamHandler()
    ]
)

# Define emotion mappings - NOTE: No mapping for emotion 7 (Surprised)
RAVDESS_EMOTION_MAPPING = {
    0: 0,  # Neutral → Neutral/Calm (0)
    1: 0,  # Calm → Neutral/Calm (0)
    2: 1,  # Happy → Happy (1)
    3: 2,  # Sad → Sad (2)
    4: 3,  # Angry → Angry (3)
    5: 4,  # Fearful → Fearful (4)
    6: 5   # Disgust → Disgust (5)
    # Note: No mapping for 7 (Surprised) - these samples will be excluded
}

# Emotion class names for the 6-class model (no Surprised)
EMOTION_LABELS = [
    'Neutral/Calm', 'Happy', 'Sad',
    'Angry', 'Fearful', 'Disgust'
]

def load_ravdess_data(ravdess_dir):
    """Load RAVDESS data and apply emotion remapping, excluding Surprised samples.

    Args:
        ravdess_dir: Directory containing processed RAVDESS data

    Returns:
        Tuple of lists: (video_features, audio_features, remapped_labels, filenames, valid_frames_masks)
    """
    video_features = []
    audio_features = []
    labels = []
    filenames = []
    valid_frames_masks = []
    original_labels = []

    # Find all .npz files
    file_pattern = os.path.join(ravdess_dir, "*.npz")
    files = sorted(glob.glob(file_pattern))

    if not files:
        logging.error(f"No .npz files found in {ravdess_dir}")
        return None, None, None, None, None, None

    logging.info(f"Loading RAVDESS data from {ravdess_dir}")

    # Load each file
    for file_path in tqdm(files, desc="Loading RAVDESS data"):
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
                    orig_emotion = data['emotion_label'].item()
                    
                    # Skip Surprised samples (emotion 7)
                    if orig_emotion == 7:
                        logging.info(f"Skipping Surprised sample: {base_filename}")
                        continue
                    
                    # Remap emotion according to our 6-class schema
                    if orig_emotion in RAVDESS_EMOTION_MAPPING:
                        remapped_emotion = RAVDESS_EMOTION_MAPPING[orig_emotion]
                    else:
                        logging.warning(f"Unknown emotion {orig_emotion} in {file_path}, skipping.")
                        continue
                else:
                    # Try to parse from filename (format: 01-01-06-01-02-01-16)
                    # Third segment is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
                    parts = base_filename.split('-')
                    if len(parts) >= 3:
                        try:
                            emotion_code = int(parts[2])
                            
                            # Skip Surprised samples (emotion 8 in filename, 7 in 0-based index)
                            if emotion_code == 8:
                                logging.info(f"Skipping Surprised sample: {base_filename}")
                                continue
                                
                            # Convert to 0-based index
                            orig_emotion = emotion_code - 1
                            
                            # Remap emotion
                            if orig_emotion in RAVDESS_EMOTION_MAPPING:
                                remapped_emotion = RAVDESS_EMOTION_MAPPING[orig_emotion]
                            else:
                                logging.warning(f"Unknown emotion {orig_emotion} in {file_path}, skipping.")
                                continue
                                
                        except (ValueError, IndexError, KeyError):
                            logging.warning(f"Could not parse emotion from filename {base_filename}")
                            continue
                    else:
                        logging.warning(f"Could not find emotion label for {file_path}, skipping.")
                        continue

                # Add data to our collections
                video_features.append(video_feat)
                audio_features.append(audio_feat)
                labels.append(remapped_emotion)
                original_labels.append(orig_emotion)
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

    logging.info(f"Loaded {len(video_features)} RAVDESS videos with valid features")

    # Log some statistics
    video_lengths = [v.shape[0] for v in video_features]
    audio_lengths = [a.shape[0] for a in audio_features]

    logging.info(f"RAVDESS video frames per file: min={min(video_lengths)}, max={max(video_lengths)}, avg={np.mean(video_lengths):.1f}")
    logging.info(f"RAVDESS audio frames per file: min={min(audio_lengths)}, max={max(audio_lengths)}, avg={np.mean(audio_lengths):.1f}")

    # Log remapping statistics
    orig_counts = {}
    remapped_counts = {}
    for o, r in zip(original_labels, labels):
        orig_counts[o] = orig_counts.get(o, 0) + 1
        remapped_counts[r] = remapped_counts.get(r, 0) + 1

    logging.info("RAVDESS original emotion distribution:")
    ravdess_original_emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    for i in range(8):
        count = orig_counts.get(i, 0)
        logging.info(f"  {ravdess_original_emotions[i]}: {count}")

    logging.info("RAVDESS remapped emotion distribution:")
    for i in range(6):  # 6-class model
        count = remapped_counts.get(i, 0)
        logging.info(f"  {EMOTION_LABELS[i]}: {count}")

    return video_features, audio_features, labels, original_labels, filenames, valid_frames_masks

def load_crema_d_data(crema_d_dir):
    """Load CREMA-D data which is already mapped to our emotion schema.

    Args:
        crema_d_dir: Directory containing processed CREMA-D data

    Returns:
        Tuple of lists: (video_features, audio_features, labels, filenames, valid_frames_masks)
    """
    video_features = []
    audio_features = []
    labels = []
    filenames = []
    valid_frames_masks = []
    original_labels = []

    # Find all .npz files
    file_pattern = os.path.join(crema_d_dir, "*.npz")
    files = sorted(glob.glob(file_pattern))

    if not files:
        logging.error(f"No .npz files found in {crema_d_dir}")
        return None, None, None, None, None, None

    logging.info(f"Loading CREMA-D data from {crema_d_dir}")

    # Load each file
    for file_path in tqdm(files, desc="Loading CREMA-D data"):
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
                    emotion_label = data['emotion_label'].item()
                    # Get original emotion if available
                    orig_emotion = data['original_emotion'].item() if 'original_emotion' in data else None
                    
                    # Verify the emotion is within our 6-class range
                    if emotion_label >= 6:
                        logging.warning(f"Invalid emotion {emotion_label} in {file_path}, skipping.")
                        continue
                else:
                    # Parse from filename (format: 1001_DFA_ANG_XX.flv)
                    # Third segment is emotion code: NEU=0, HAP=1, SAD=2, ANG=3, FEA=4, DIS=5
                    parts = base_filename.split('_')
                    if len(parts) >= 3:
                        try:
                            emotion_code = parts[2]
                            emotion_map = {
                                'NEU': 0,  # Neutral
                                'HAP': 1,  # Happy
                                'SAD': 2,  # Sad
                                'ANG': 3,  # Angry
                                'FEA': 4,  # Fearful
                                'DIS': 5   # Disgust
                            }
                            if emotion_code in emotion_map:
                                emotion_label = emotion_map[emotion_code]
                                orig_emotion = emotion_code
                            else:
                                logging.warning(f"Unknown emotion code {emotion_code} in {file_path}, skipping.")
                                continue
                        except (ValueError, IndexError, KeyError):
                            logging.warning(f"Could not parse emotion from filename {base_filename}")
                            continue
                    else:
                        logging.warning(f"Could not find emotion label for {file_path}, skipping.")
                        continue

                # Add data to our collections
                video_features.append(video_feat)
                audio_features.append(audio_feat)
                labels.append(emotion_label)
                original_labels.append(orig_emotion)
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

    logging.info(f"Loaded {len(video_features)} CREMA-D videos with valid features")

    # Log some statistics
    video_lengths = [v.shape[0] for v in video_features]
    audio_lengths = [a.shape[0] for a in audio_features]

    logging.info(f"CREMA-D video frames per file: min={min(video_lengths)}, max={max(video_lengths)}, avg={np.mean(video_lengths):.1f}")
    logging.info(f"CREMA-D audio frames per file: min={min(audio_lengths)}, max={max(audio_lengths)}, avg={np.mean(audio_lengths):.1f}")

    # Log emotion distribution
    emotion_counts = {}
    for l in labels:
        emotion_counts[l] = emotion_counts.get(l, 0) + 1

    logging.info("CREMA-D emotion distribution:")
    for i in range(6):  # 6-class model
        count = emotion_counts.get(i, 0)
        percent = count / len(labels) * 100 if labels else 0
        logging.info(f"  {EMOTION_LABELS[i]}: {count} ({percent:.1f}%)")

    return video_features, audio_features, labels, original_labels, filenames, valid_frames_masks

def combine_datasets(ravdess_data, crema_d_data):
    """Combine RAVDESS and CREMA-D datasets.

    Args:
        ravdess_data: Tuple of (video_features, audio_features, labels, original_labels, filenames, valid_frames_masks)
        crema_d_data: Tuple of (video_features, audio_features, labels, original_labels, filenames, valid_frames_masks)

    Returns:
        Tuple of combined data: (video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks)
    """
    if ravdess_data is None or crema_d_data is None:
        logging.error("One or both datasets are missing. Cannot combine datasets.")
        return None

    # Extract data
    rav_video, rav_audio, rav_labels, _, rav_filenames, rav_masks = ravdess_data
    cre_video, cre_audio, cre_labels, _, cre_filenames, cre_masks = crema_d_data

    # Create dataset flags (0 = RAVDESS, 1 = CREMA-D)
    rav_flags = [0] * len(rav_labels)
    cre_flags = [1] * len(cre_labels)

    # Combine data
    combined_video = rav_video + cre_video
    combined_audio = rav_audio + cre_audio
    combined_labels = rav_labels + cre_labels
    combined_flags = rav_flags + cre_flags
    combined_filenames = rav_filenames + cre_filenames
    combined_masks = rav_masks + cre_masks

    logging.info(f"Combined dataset: {len(combined_labels)} samples")
    logging.info(f"  - RAVDESS: {len(rav_labels)} samples")
    logging.info(f"  - CREMA-D: {len(cre_labels)} samples")

    # Log combined emotion distribution
    emotion_counts = {}
    for l in combined_labels:
        emotion_counts[l] = emotion_counts.get(l, 0) + 1

    logging.info("Combined emotion distribution:")
    for i in range(6):  # 6-class model
        count = emotion_counts.get(i, 0)
        percent = count / len(combined_labels) * 100 if combined_labels else 0
        logging.info(f"  {EMOTION_LABELS[i]}: {count} ({percent:.1f}%)")

    return (combined_video, combined_audio, combined_labels, combined_flags, combined_filenames, combined_masks)

def define_branched_model(video_input_shape, audio_input_shape, num_classes=6, use_masking=True):
    """Defines the branched LSTM model architecture with attention mechanisms
    adapted for variable-length single-segment input.

    Args:
        video_input_shape: Shape of the video input data (max_seq_length, feature_dim).
        audio_input_shape: Shape of the audio input data (max_seq_length, feature_dim).
        num_classes: Number of emotion classes (default: 6 for our combined schema)
        use_masking: Whether to use masking for invalid frames.

    Returns:
        Compiled Keras model.
    """
    # Define inputs for both streams
    video_input = Input(shape=video_input_shape, name='video_input')
    audio_input = Input(shape=audio_input_shape, name='audio_input')

    # Optional masking layers to handle variable-length sequences and invalid frames
    if use_masking:
        # For video we'll mask zeros (represents padding or invalid frames)
        video_masked = Masking(mask_value=0.0, name='video_masking')(video_input)
        # For audio we'll mask zeros (represents padding)
        audio_masked = Masking(mask_value=0.0, name='audio_masking')(audio_input)
    else:
        video_masked = video_input
        audio_masked = audio_input

    # ---- Video Branch 1: Standard LSTM ----
    video_branch1 = LSTM(128, return_sequences=True, name='video_branch1_lstm1')(video_masked)
    video_branch1 = Dropout(0.4)(video_branch1)  # Increased dropout from 0.3 to 0.4
    video_branch1 = LSTM(64, name='video_branch1_lstm2')(video_branch1)

    # ---- Video Branch 2: Bidirectional LSTM ----
    video_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='video_branch2_bilstm1')(video_masked)
    video_branch2_seq = Dropout(0.4)(video_branch2_seq)  # Increased dropout from 0.3 to 0.4

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
    audio_branch1 = Dropout(0.4)(audio_branch1)  # Increased dropout from 0.3 to 0.4
    audio_branch1 = LSTM(64, name='audio_branch1_lstm2')(audio_branch1)

    # ---- Audio Branch 2: Bidirectional LSTM ----
    audio_branch2_seq = Bidirectional(LSTM(128, return_sequences=True), name='audio_branch2_bilstm1')(audio_masked)
    audio_branch2_seq = Dropout(0.4)(audio_branch2_seq)  # Increased dropout from 0.3 to 0.4

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
    audio_branch3 = Dropout(0.4)(audio_branch3)  # Increased dropout from 0.3 to 0.4
    audio_branch3 = LSTM(64, name='audio_branch3_lstm2')(audio_branch3)

    # ---- Video modality fusion ----
    video_fusion = concatenate([video_branch1, video_branch2], name='video_fusion')
    video_fusion = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='video_fusion_dense')(video_fusion)  # Added L2
    video_fusion = Dropout(0.5)(video_fusion)  # Increased dropout from 0.4 to 0.5

    # ---- Audio modality fusion ----
    audio_fusion = concatenate([audio_branch1, audio_branch2, audio_branch3], name='audio_fusion')
    audio_fusion = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='audio_fusion_dense')(audio_fusion)  # Added L2
    audio_fusion = Dropout(0.5)(audio_fusion)  # Increased dropout from 0.4 to 0.5

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
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense1')(cross_modal_fusion)  # Added L2
    dropout = Dropout(0.6)(dense1)  # Increased dropout from 0.5 to 0.6
    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense2')(dropout)  # Added L2
    dropout2 = Dropout(0.4)(dense2)  # Increased dropout from 0.3 to 0.4

    # Output layer - NOTE: num_classes=6 for the 6-class model
    output_layer = Dense(num_classes, activation='softmax', name='output')(dropout2)

    # Create the branched model
    model = Model(inputs=[video_input, audio_input], outputs=output_layer)

    return model

def f1_score(y_true, y_pred):
    """Custom F1 metric for multi-class classification"""
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

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
            f1_score  # Added F1 score metric
        ]
    )

def create_dataset_split(video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks,
                         test_size=0.2, val_size=0.1, random_state=42, stratify_by_dataset=True):
    """Split the dataset into training, validation, and test sets.

    Args:
        video_features: List of video feature arrays.
        audio_features: List of audio feature arrays.
        labels: List of emotion labels.
        dataset_flags: List of dataset flags (0=RAVDESS, 1=CREMA-D)
        filenames: List of file names.
        valid_frames_masks: List of boolean masks for valid frames.
        test_size: Proportion of data for testing.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.
        stratify_by_dataset: Whether to stratify splits by dataset and emotion

    Returns:
        Tuple of training, validation, and test data.
    """
    # Convert labels to categorical format (one-hot encoding)
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)
    labels_categorical = to_categorical([l for l in labels], num_classes=num_classes)

    # Create stratification array that combines dataset and emotion
    if stratify_by_dataset:
        # Combine dataset and emotion: dataset*10 + emotion
        # This ensures we maintain dataset and emotion distribution
        stratify_labels = [dataset_flags[i] * 10 + labels[i] for i in range(len(labels))]
    else:
        # Stratify by emotion only
        stratify_labels = labels

    # First split: training + validation vs. test
    train_val_indices, test_indices = train_test_split(
        range(len(video_features)),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels  # Stratify by combined dataset+emotion
    )

    # Second split: training vs. validation
    # Calculate validation size as a proportion of the training + validation set
    val_proportion = val_size / (1 - test_size)

    # Get stratification labels for training + validation split
    if stratify_by_dataset:
        stratify_train_val = [stratify_labels[i] for i in train_val_indices]
    else:
        stratify_train_val = [labels[i] for i in train_val_indices]

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_proportion,
        random_state=random_state,
        stratify=stratify_train_val  # Stratify by combined dataset+emotion
    )

    # Create the split datasets
    X_train_video = [video_features[i] for i in train_indices]
    X_train_audio = [audio_features[i] for i in train_indices]
    y_train = labels_categorical[train_indices]
    train_filenames = [filenames[i] for i in train_indices]
    train_dataset_flags = [dataset_flags[i] for i in train_indices]
    train_valid_frames = [valid_frames_masks[i] for i in train_indices] if valid_frames_masks else None
    
    X_val_video = [video_features[i] for i in val_indices]
    X_val_audio = [audio_features[i] for i in val_indices]
    y_val = labels_categorical[val_indices]
    val_filenames = [filenames[i] for i in val_indices]
    val_dataset_flags = [dataset_flags[i] for i in val_indices]
    val_valid_frames = [valid_frames_masks[i] for i in val_indices] if valid_frames_masks else None
    
    X_test_video = [video_features[i] for i in test_indices]
    X_test_audio = [audio_features[i] for i in test_indices]
    y_test = labels_categorical[test_indices]
    test_filenames = [filenames[i] for i in test_indices]
    test_dataset_flags = [dataset_flags[i] for i in test_indices]
    test_valid_frames = [valid_frames_masks[i] for i in test_indices] if valid_frames_masks else None
    
    # Log split statistics
    logging.info(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation, {len(test_indices)} test")
    
    # Check emotion distribution in splits
    train_emotion_counts = {}
    val_emotion_counts = {}
    test_emotion_counts = {}
    
    for i in train_indices:
        emotion = labels[i]
        train_emotion_counts[emotion] = train_emotion_counts.get(emotion, 0) + 1
        
    for i in val_indices:
        emotion = labels[i]
        val_emotion_counts[emotion] = val_emotion_counts.get(emotion, 0) + 1
        
    for i in test_indices:
        emotion = labels[i]
        test_emotion_counts[emotion] = test_emotion_counts.get(emotion, 0) + 1
    
    logging.info("Emotion distribution in splits:")
    for i in range(6):  # 6-class model
        train_count = train_emotion_counts.get(i, 0)
        val_count = val_emotion_counts.get(i, 0)
        test_count = test_emotion_counts.get(i, 0)
        logging.info(f"  {EMOTION_LABELS[i]}: train={train_count}, val={val_count}, test={test_count}")
    
    return (
        (X_train_video, X_train_audio, y_train, train_filenames, train_dataset_flags, train_valid_frames),
        (X_val_video, X_val_audio, y_val, val_filenames, val_dataset_flags, val_valid_frames),
        (X_test_video, X_test_audio, y_test, test_filenames, test_dataset_flags, test_valid_frames)
    )

def pad_sequences(sequences, max_length=None, padding_value=0.0):
    """Pad sequences to the same length.
    
    Args:
        sequences: List of numpy arrays.
        max_length: Maximum length to pad to. If None, use the longest sequence.
        padding_value: Value to pad with.
        
    Returns:
        Numpy array of padded sequences.
    """
    if not sequences:
        return np.array([])
    
    # Get sequence lengths
    lengths = [len(s) for s in sequences]
    
    # Determine max length
    if max_length is None:
        max_length = max(lengths)
    
    # Get feature dimension from first sequence
    feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    
    # Create padded array
    padded = np.ones((len(sequences), max_length, feature_dim)) * padding_value
    
    # Copy data
    for i, seq in enumerate(sequences):
        length = min(lengths[i], max_length)
        padded[i, :length] = seq[:length]
    
    return padded

def cosine_annealing_schedule(epoch, lr, total_epochs=100, min_lr=1e-5, cycle_epochs=5):
    """Cosine annealing learning rate schedule with warm restarts.
    
    Args:
        epoch: Current epoch.
        lr: Current learning rate.
        total_epochs: Total number of epochs.
        min_lr: Minimum learning rate.
        cycle_epochs: Number of epochs per cycle.
        
    Returns:
        New learning rate.
    """
    # Calculate current cycle and position within cycle
    cycle = epoch // cycle_epochs
    cycle_position = epoch % cycle_epochs
    
    # Apply cosine annealing within current cycle
    cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_position / cycle_epochs))
    new_lr = min_lr + (lr - min_lr) * cosine_decay
    
    logging.info(f"Epoch {epoch}: Learning rate = {new_lr:.6f}")
    return new_lr

def train_model(model, train_data, val_data, batch_size=32, epochs=100, patience=15, 
               model_save_path="models/branched_6class_model",
               apply_class_weights=True, use_lr_schedule=True):
    """Train the model using the provided data.
    
    Args:
        model: Keras model to train.
        train_data: Tuple of (X_train_video, X_train_audio, y_train, train_filenames, train_dataset_flags, train_valid_frames)
        val_data: Tuple of (X_val_video, X_val_audio, y_val, val_filenames, val_dataset_flags, val_valid_frames)
        batch_size: Training batch size.
        epochs: Maximum number of epochs to train for.
        patience: Early stopping patience.
        model_save_path: Path to save the trained model.
        apply_class_weights: Whether to apply class weights to handle class imbalance.
        use_lr_schedule: Whether to use cosine annealing learning rate schedule.
        
    Returns:
        Training history.
    """
    # Extract data
    X_train_video, X_train_audio, y_train, train_filenames, train_dataset_flags, train_valid_frames = train_data
    X_val_video, X_val_audio, y_val, val_filenames, val_dataset_flags, val_valid_frames = val_data
    
    # Determine maximum sequence lengths
    max_video_length = max(max(len(v) for v in X_train_video), max(len(v) for v in X_val_video))
    max_audio_length = max(max(len(a) for a in X_train_audio), max(len(a) for a in X_val_audio))
    
    logging.info(f"Maximum sequence lengths: video={max_video_length}, audio={max_audio_length}")
    
    # Pad sequences
    X_train_video_padded = pad_sequences(X_train_video, max_length=max_video_length)
    X_train_audio_padded = pad_sequences(X_train_audio, max_length=max_audio_length)
    X_val_video_padded = pad_sequences(X_val_video, max_length=max_video_length)
    X_val_audio_padded = pad_sequences(X_val_audio, max_length=max_audio_length)
    
    # Apply class weights if needed
    class_weights = None
    if apply_class_weights:
        # Get original labels (not one-hot encoded)
        train_labels = np.argmax(y_train, axis=1)
        
        # Calculate class weights
        weights = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        
        # Convert to dictionary
        class_weights = {i: weights[i] for i in range(len(weights))}
        logging.info(f"Class weights: {class_weights}")
    
    # Create model callbacks
    callbacks = []
    
    # Model checkpointing
    model_checkpoint = ModelCheckpoint(
        filepath=f"{model_save_path}.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate scheduling
    if use_lr_schedule:
        # Cosine annealing with warm restarts
        lr_scheduler = LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_schedule(
                epoch, 
                lr, 
                total_epochs=epochs,
                min_lr=1e-6,
                cycle_epochs=10
            ),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    else:
        # Standard reduce on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        [X_train_video_padded, X_train_audio_padded],
        y_train,
        validation_data=([X_val_video_padded, X_val_audio_padded], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the trained model
    model.save(f"{model_save_path}_final.h5")
    logging.info(f"Model saved to {model_save_path}_final.h5")
    
    return history

def evaluate_model(model, test_data, batch_size=32, model_save_path="models/branched_6class_model", visualize=True):
    """Evaluate the trained model on the test set.
    
    Args:
        model: Trained Keras model.
        test_data: Tuple of (X_test_video, X_test_audio, y_test, test_filenames, test_dataset_flags, test_valid_frames)
        batch_size: Batch size for evaluation.
        model_save_path: Path to save evaluation results.
        visualize: Whether to generate visualizations.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Extract data
    X_test_video, X_test_audio, y_test, test_filenames, test_dataset_flags, test_valid_frames = test_data
    
    # Determine maximum sequence lengths
    max_video_length = max(len(v) for v in X_test_video)
    max_audio_length = max(len(a) for a in X_test_audio)
    
    # Pad sequences
    X_test_video_padded = pad_sequences(X_test_video, max_length=max_video_length)
    X_test_audio_padded = pad_sequences(X_test_audio, max_length=max_audio_length)
    
    # Evaluate the model
    logging.info("Evaluating model on test set...")
    test_metrics = model.evaluate(
        [X_test_video_padded, X_test_audio_padded],
        y_test,
        batch_size=batch_size,
        verbose=1
    )
    
    # Get metric names
    metric_names = model.metrics_names
    results = {metric_names[i]: test_metrics[i] for i in range(len(metric_names))}
    
    # Log metrics
    for name, value in results.items():
        logging.info(f"Test {name}: {value:.4f}")
    
    # Get predictions
    predictions = model.predict([X_test_video_padded, X_test_audio_padded], batch_size=batch_size)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Compute classification report
    class_report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=EMOTION_LABELS,
        digits=4
    )
    logging.info(f"Classification Report:\n{class_report}")
    
    # Save classification report to file
    with open(f"{model_save_path}_classification_report.txt", "w") as f:
        f.write(class_report)
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Optionally visualize results
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(EMOTION_LABELS))
        plt.xticks(tick_marks, EMOTION_LABELS, rotation=45)
        plt.yticks(tick_marks, EMOTION_LABELS)
        
        # Add normalized values to cells
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"{model_save_path}_confusion_matrix.png")
        
        # Create misclassification visualization
        visualize_misclassifications(true_classes, predicted_classes, test_dataset_flags, test_filenames, model_save_path)
    
    # Collect metrics for different dataset sources
    results_by_dataset = {}
    
    # RAVDESS results
    ravdess_indices = [i for i, flag in enumerate(test_dataset_flags) if flag == 0]
    if ravdess_indices:
        ravdess_true = [true_classes[i] for i in ravdess_indices]
        ravdess_pred = [predicted_classes[i] for i in ravdess_indices]
        ravdess_report = classification_report(
            ravdess_true,
            ravdess_pred,
            target_names=EMOTION_LABELS,
            output_dict=True
        )
        results_by_dataset['RAVDESS'] = {
            'accuracy': ravdess_report['accuracy'],
            'macro_f1': ravdess_report['macro avg']['f1-score']
        }
        logging.info(f"RAVDESS Accuracy: {ravdess_report['accuracy']:.4f}")
        logging.info(f"RAVDESS Macro F1: {ravdess_report['macro avg']['f1-score']:.4f}")
    
    # CREMA-D results
    crema_d_indices = [i for i, flag in enumerate(test_dataset_flags) if flag == 1]
    if crema_d_indices:
        crema_d_true = [true_classes[i] for i in crema_d_indices]
        crema_d_pred = [predicted_classes[i] for i in crema_d_indices]
        crema_d_report = classification_report(
            crema_d_true,
            crema_d_pred,
            target_names=EMOTION_LABELS,
            output_dict=True
        )
        results_by_dataset['CREMA-D'] = {
            'accuracy': crema_d_report['accuracy'],
            'macro_f1': crema_d_report['macro avg']['f1-score']
        }
        logging.info(f"CREMA-D Accuracy: {crema_d_report['accuracy']:.4f}")
        logging.info(f"CREMA-D Macro F1: {crema_d_report['macro avg']['f1-score']:.4f}")
    
    # Add dataset-specific results
    results['by_dataset'] = results_by_dataset
    
    return results

def visualize_misclassifications(true_classes, predicted_classes, dataset_flags, filenames, output_path):
    """Create visualizations of misclassified samples.
    
    Args:
        true_classes: True class labels.
        predicted_classes: Predicted class labels.
        dataset_flags: Dataset source flags (0=RAVDESS, 1=CREMA-D).
        filenames: List of sample filenames.
        output_path: Path to save visualization.
    """
    # Find misclassified samples
    misclassified_indices = np.where(true_classes != predicted_classes)[0]
    
    if len(misclassified_indices) == 0:
        logging.info("No misclassified samples found.")
        return
    
    # Get misclassification details
    misclassified_true = [true_classes[i] for i in misclassified_indices]
    misclassified_pred = [predicted_classes[i] for i in misclassified_indices]
    misclassified_dataset = [dataset_flags[i] for i in misclassified_indices]
    misclassified_files = [filenames[i] for i in misclassified_indices]
    
    # Create dataframe of misclassifications
    df = pd.DataFrame({
        'Filename': misclassified_files,
        'Dataset': ['RAVDESS' if d == 0 else 'CREMA-D' for d in misclassified_dataset],
        'True Emotion': [EMOTION_LABELS[t] for t in misclassified_true],
        'Predicted Emotion': [EMOTION_LABELS[p] for p in misclassified_pred]
    })
    
    # Save to CSV
    df.to_csv(f"{output_path}_misclassifications.csv", index=False)
    logging.info(f"Saved misclassification details to {output_path}_misclassifications.csv")
    
    # Create confusion flow diagram (which emotions get confused for which)
    plt.figure(figsize=(12, 10))
    
    # Count misclassification pairs
    pairs = {}
    for t, p in zip(misclassified_true, misclassified_pred):
        pair = (EMOTION_LABELS[t], EMOTION_LABELS[p])
        pairs[pair] = pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    
    # Create bar chart of most common misclassifications
    labels = [f"{t} → {p}" for (t, p), _ in sorted_pairs[:15]]  # Top 15
    values = [count for _, count in sorted_pairs[:15]]
    
    plt.figure(figsize=(12, 8))
    plt.barh(labels, values, color='salmon')
    plt.xlabel('Count')
    plt.ylabel('Misclassification')
    plt.title('Most Common Misclassifications')
    plt.tight_layout()
    plt.savefig(f"{output_path}_misclassification_types.png")
    
    # Analyze misclassifications by dataset
    ravdess_misclass = sum(1 for d in misclassified_dataset if d == 0)
    crema_d_misclass = sum(1 for d in misclassified_dataset if d == 1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(['RAVDESS', 'CREMA-D'], [ravdess_misclass, crema_d_misclass], color=['blue', 'green'])
    plt.ylabel('Number of Misclassifications')
    plt.title('Misclassifications by Dataset')
    plt.savefig(f"{output_path}_misclassification_by_dataset.png")

def plot_training_history(history, output_path="models/branched_6class_model"):
    """Plot training and validation metrics history.
    
    Args:
        history: Keras history object.
        output_path: Path to save plots.
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(14, 5))
    
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
    plt.savefig(f"{output_path}_training_history.png")
    
    # Plot F1 score
    if 'f1_score' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['f1_score'])
        plt.plot(history.history['val_f1_score'])
        plt.title('Model F1 Score')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(f"{output_path}_f1_score_history.png")

def main():
    """Main function to run the training and evaluation process."""
    parser = argparse.ArgumentParser(description="Train and evaluate 6-class branched model")
    parser.add_argument("--ravdess-dir", default="ravdess_features", help="Directory containing RAVDESS .npz files")
    parser.add_argument("--crema-d-dir", default="crema_d_features", help="Directory containing CREMA-D .npz files")
    parser.add_argument("--output-dir", default="models", help="Directory to save model and results")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--no-lr-schedule", action="store_true", help="Disable cosine annealing learning rate schedule")
    parser.add_argument("--test-only", action="store_true", help="Skip training and only test a saved model")
    parser.add_argument("--model-path", default=None, help="Path to saved model for testing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model save path
    model_save_path = os.path.join(args.output_dir, "branched_6class_model")
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    if not args.test_only:
        # Load datasets
        logging.info("Loading RAVDESS dataset...")
        ravdess_data = load_ravdess_data(args.ravdess_dir)
        
        logging.info("Loading CREMA-D dataset...")
        crema_d_data = load_crema_d_data(args.crema_d_dir)
        
        # Combine datasets
        logging.info("Combining datasets...")
        combined_data = combine_datasets(ravdess_data, crema_d_data)
        
        if combined_data is None:
            logging.error("Failed to combine datasets. Exiting.")
            return
        
        # Split dataset
        logging.info("Splitting dataset into train/val/test sets...")
        video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks = combined_data
        
        train_data, val_data, test_data = create_dataset_split(
            video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks,
            test_size=0.2, val_size=0.1, random_state=42, stratify_by_dataset=True
        )
        
        # Get input shapes
        max_video_length = max(max(len(v) for v in video_features), 100)  # Minimum 100 frames
        max_audio_length = max(max(len(a) for a in audio_features), 500)  # Minimum 500 frames
        
        video_feature_dim = video_features[0].shape[1] if len(video_features[0].shape) > 1 else 1
        audio_feature_dim = audio_features[0].shape[1] if len(audio_features[0].shape) > 1 else 1
        
        video_input_shape = (max_video_length, video_feature_dim)
        audio_input_shape = (max_audio_length, audio_feature_dim)
        
        logging.info(f"Video input shape: {video_input_shape}")
        logging.info(f"Audio input shape: {audio_input_shape}")
        
        # Define and compile model
        logging.info("Creating model...")
        model = define_branched_model(video_input_shape, audio_input_shape, num_classes=6, use_masking=True)
        
        logging.info("Compiling model...")
        compile_model(model, learning_rate=0.001)
        
        # Print model summary
        model.summary(print_fn=logging.info)
        
        # Train model
        logging.info("Training model...")
        history = train_model(
            model,
            train_data,
            val_data,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            model_save_path=model_save_path,
            apply_class_weights=not args.no_class_weights,
            use_lr_schedule=not args.no_lr_schedule
        )
        
        # Plot training history
        logging.info("Plotting training history...")
        plot_training_history(history, output_path=model_save_path)
        
        # Evaluate on test set
        logging.info("Evaluating model on test set...")
        results = evaluate_model(
            model,
            test_data,
            batch_size=args.batch_size,
            model_save_path=model_save_path,
            visualize=True
        )
    else:
        # Load model for testing only
        if args.model_path is None:
            logging.error("Model path must be provided for test-only mode. Use --model-path.")
            return
        
        logging.info(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path, custom_objects={'f1_score': f1_score})
        
        # Load datasets
        logging.info("Loading RAVDESS dataset...")
        ravdess_data = load_ravdess_data(args.ravdess_dir)
        
        logging.info("Loading CREMA-D dataset...")
        crema_d_data = load_crema_d_data(args.crema_d_dir)
        
        # Combine datasets
        logging.info("Combining datasets...")
        combined_data = combine_datasets(ravdess_data, crema_d_data)
        
        # Split dataset (we only need the test set for evaluation)
        logging.info("Splitting dataset to get test set...")
        video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks = combined_data
        
        _, _, test_data = create_dataset_split(
            video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks,
            test_size=0.2, val_size=0.1, random_state=42, stratify_by_dataset=True
        )
        
        # Evaluate on test set
        logging.info("Evaluating model on test set...")
        results = evaluate_model(
            model,
            test_data,
            batch_size=args.batch_size,
            model_save_path=model_save_path,
            visualize=True
        )
    
    logging.info("Done!")

if __name__ == "__main__":
    main()
