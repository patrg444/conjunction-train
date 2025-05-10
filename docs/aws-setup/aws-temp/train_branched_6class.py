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
from tensorflow.keras.layers import Attention, MultiHeadAttention, BatchNormalization, ActivityRegularization
from tensorflow.keras.layers import Conv1D, GaussianNoise, GaussianDropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Import custom data generators and dynamic padding functions
from sequence_data_generator import SequenceDataGenerator, ValidationDataGenerator
from train_branched_dynamic_funcs import evaluate_model_with_generator, train_model_with_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_branched_6class.log"),
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

    # Find all .npz files in Actor subdirectories
    files = []
    actor_dirs = glob.glob(os.path.join(ravdess_dir, "Actor_*"))
    
    if actor_dirs:
        # If Actor directories exist, get files from each actor directory
        for actor_dir in actor_dirs:
            actor_files = glob.glob(os.path.join(actor_dir, "*.npz"))
            files.extend(actor_files)
    else:
        # Fallback to direct file search (for backward compatibility)
        file_pattern = os.path.join(ravdess_dir, "*.npz")
        files = glob.glob(file_pattern)
        
        # If still no files found, try recursive search
        if not files:
            file_pattern = os.path.join(ravdess_dir, "**", "*.npz")
            files = glob.glob(file_pattern, recursive=True)
    
    files = sorted(files)

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

def define_branched_model(video_input_shape, audio_input_shape, num_classes=6, use_masking=True,
                          regularization_strength=0.001, add_noise=True, reduce_capacity=False):
    """Defines the branched LSTM model architecture with attention mechanisms
    adapted for variable-length single-segment input with anti-overfitting measures.

    Args:
        video_input_shape: Shape of the video input data (max_seq_length, feature_dim).
        audio_input_shape: Shape of the audio input data (max_seq_length, feature_dim).
        num_classes: Number of emotion classes (default: 6 for our combined schema)
        use_masking: Whether to use masking for invalid frames.
        regularization_strength: L2 regularization coefficient (default: 0.001)
        add_noise: Whether to add Gaussian noise for regularization
        reduce_capacity: Whether to reduce model capacity to prevent overfitting

    Returns:
        Compiled Keras model.
    """
    # Define inputs for both streams
    video_input = Input(shape=video_input_shape, name='video_input')
    audio_input = Input(shape=audio_input_shape, name='audio_input')

    # Set LSTM units based on capacity setting
    lstm_units = 64 if reduce_capacity else 128
    lstm_units_2 = 32 if reduce_capacity else 64

    # Optional masking layers to handle variable-length sequences and invalid frames
    if use_masking:
        # For video we'll mask zeros (represents padding or invalid frames)
        video_masked = Masking(mask_value=0.0, name='video_masking')(video_input)
        # For audio we'll mask zeros (represents padding)
        audio_masked = Masking(mask_value=0.0, name='audio_masking')(audio_input)
    else:
        video_masked = video_input
        audio_masked = audio_input
    
    # Gaussian noise removed to reduce regularization
    # We're skipping the noise addition to reduce regularization

    # ---- Video Branch 1: Standard LSTM with regularization ----
    video_branch1 = LSTM(lstm_units, return_sequences=True, 
                         kernel_regularizer=l2(regularization_strength),
                         recurrent_regularizer=l2(regularization_strength),
                         name='video_branch1_lstm1')(video_masked)
    video_branch1 = BatchNormalization(name='video_branch1_bn1')(video_branch1)
    # Dropout removed to reduce regularization
    video_branch1 = LSTM(lstm_units_2, 
                         kernel_regularizer=l2(regularization_strength),
                         name='video_branch1_lstm2')(video_branch1)
    video_branch1 = BatchNormalization(name='video_branch1_bn2')(video_branch1)

    # ---- Video Branch 2: Bidirectional LSTM with regularization ----
    video_branch2_seq = Bidirectional(LSTM(lstm_units, return_sequences=True,
                                          kernel_regularizer=l2(regularization_strength)),
                                     name='video_branch2_bilstm1')(video_masked)
    video_branch2_seq = BatchNormalization(name='video_branch2_bn1')(video_branch2_seq)
    # Dropout removed to reduce regularization

    # Add self-attention to video branch 2
    try:
        # Using TensorFlow 2.x MultiHeadAttention if available
        video_branch2_attn = MultiHeadAttention(
            num_heads=4, key_dim=32, name="video_self_attention"
        )(video_branch2_seq, video_branch2_seq)
        video_branch2_seq = video_branch2_attn + video_branch2_seq  # Residual connection
        video_branch2_seq = BatchNormalization(name='video_branch2_bn_attn')(video_branch2_seq)
    except Exception as e:
        logging.warning(f"Could not use MultiHeadAttention: {str(e)}. Using regular LSTM instead.")
        pass  # Fall back to regular processing if MultiHeadAttention fails

    video_branch2 = Bidirectional(LSTM(lstm_units_2, 
                                      kernel_regularizer=l2(regularization_strength)),
                                 name='video_branch2_bilstm2')(video_branch2_seq)
    video_branch2 = BatchNormalization(name='video_branch2_bn2')(video_branch2)

    # ---- Audio Branch 1: Standard LSTM with regularization ----
    audio_branch1 = LSTM(lstm_units, return_sequences=True, 
                         kernel_regularizer=l2(regularization_strength),
                         name='audio_branch1_lstm1')(audio_masked)
    audio_branch1 = BatchNormalization(name='audio_branch1_bn1')(audio_branch1)
    # Dropout removed to reduce regularization
    audio_branch1 = LSTM(lstm_units_2, 
                         kernel_regularizer=l2(regularization_strength),
                         name='audio_branch1_lstm2')(audio_branch1)
    audio_branch1 = BatchNormalization(name='audio_branch1_bn2')(audio_branch1)

    # ---- Audio Branch 2: Bidirectional LSTM with regularization ----
    audio_branch2_seq = Bidirectional(LSTM(lstm_units, return_sequences=True,
                                          kernel_regularizer=l2(regularization_strength)),
                                     name='audio_branch2_bilstm1')(audio_masked)
    audio_branch2_seq = BatchNormalization(name='audio_branch2_bn1')(audio_branch2_seq)
    # Dropout removed to reduce regularization

    # Add self-attention to audio branch 2
    try:
        # Using TensorFlow 2.x MultiHeadAttention if available
        audio_branch2_attn = MultiHeadAttention(
            num_heads=4, key_dim=32, name="audio_self_attention"
        )(audio_branch2_seq, audio_branch2_seq)
        audio_branch2_seq = audio_branch2_attn + audio_branch2_seq  # Residual connection
        audio_branch2_seq = BatchNormalization(name='audio_branch2_bn_attn')(audio_branch2_seq)
    except Exception as e:
        logging.warning(f"Could not use MultiHeadAttention: {str(e)}. Using regular LSTM instead.")
        pass  # Fall back to regular processing if MultiHeadAttention fails

    audio_branch2 = Bidirectional(LSTM(lstm_units_2, 
                                      kernel_regularizer=l2(regularization_strength)),
                                 name='audio_branch2_bilstm2')(audio_branch2_seq)
    audio_branch2 = BatchNormalization(name='audio_branch2_bn2')(audio_branch2)

    # ---- Audio Branch 3: LSTM with regularization ----
    # Using simpler LSTM branch instead of Conv1D to better handle variable length
    audio_branch3 = LSTM(lstm_units, return_sequences=True, 
                         kernel_regularizer=l2(regularization_strength),
                         name='audio_branch3_lstm1')(audio_masked)
    audio_branch3 = BatchNormalization(name='audio_branch3_bn1')(audio_branch3)
    # Dropout removed to reduce regularization
    audio_branch3 = LSTM(lstm_units_2, 
                         kernel_regularizer=l2(regularization_strength),
                         name='audio_branch3_lstm2')(audio_branch3)
    audio_branch3 = BatchNormalization(name='audio_branch3_bn2')(audio_branch3)

    # ---- Video modality fusion ----
    video_fusion = concatenate([video_branch1, video_branch2], name='video_fusion')
    video_fusion = Dense(lstm_units, activation='relu',
                         kernel_regularizer=l2(regularization_strength),
                         name='video_fusion_dense')(video_fusion)
    video_fusion = BatchNormalization(name='video_fusion_bn')(video_fusion)
    # Dropout removed to reduce regularization

    # ---- Audio modality fusion ----
    audio_fusion = concatenate([audio_branch1, audio_branch2, audio_branch3], name='audio_fusion')
    audio_fusion = Dense(lstm_units, activation='relu',
                         kernel_regularizer=l2(regularization_strength),
                         name='audio_fusion_dense')(audio_fusion)
    audio_fusion = BatchNormalization(name='audio_fusion_bn')(audio_fusion)
    # Dropout removed to reduce regularization

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

        # Final multimodal fusion with batch normalization
        cross_modal_fusion = concatenate([video_with_context, audio_with_context], name='cross_modal_fusion')
        cross_modal_fusion = BatchNormalization(name='cross_modal_bn')(cross_modal_fusion)
    except Exception as e:
        logging.warning(f"Could not use cross-modal attention: {str(e)}. Falling back to simple concatenation.")
        # Fallback to standard concatenation if attention fails
        cross_modal_fusion = concatenate([video_fusion, audio_fusion], name='cross_modal_fusion_fallback')
        cross_modal_fusion = BatchNormalization(name='cross_modal_bn_fallback')(cross_modal_fusion)

    # ---- Final classification layers with increased regularization ----
    dense_units = 128 if reduce_capacity else 256
    dense1 = Dense(dense_units, activation='relu', 
                   kernel_regularizer=l2(regularization_strength),
                   name='dense1')(cross_modal_fusion)
    dense1 = BatchNormalization(name='dense1_bn')(dense1)
    # Dropout removed to reduce regularization
    
    dense2 = Dense(dense_units//2, activation='relu', 
                   kernel_regularizer=l2(regularization_strength),
                   name='dense2')(dense1)
    dense2 = BatchNormalization(name='dense2_bn')(dense2)
    # Dropout removed to reduce regularization

    # Add activity regularization to final layer
    regularized = ActivityRegularization(l1=regularization_strength/2, 
                                       l2=regularization_strength)(dense2)
    
    # Output layer - NOTE: num_classes=6 for the 6-class model
    output_layer = Dense(num_classes, activation='softmax', 
                         kernel_regularizer=l2(regularization_strength),
                         name='output')(regularized)

    # Create the branched model
    model = Model(inputs=[video_input, audio_input], outputs=output_layer)

    return model

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # Using epsilon to avoid division by zero
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

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
            F1Score(name='f1_score')  # Added F1-score metric
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
    train_masks = [valid_frames_masks[i] for i in train_indices]

    X_val_video = [video_features[i] for i in val_indices]
    X_val_audio = [audio_features[i] for i in val_indices]
    y_val = labels_categorical[val_indices]
    val_filenames = [filenames[i] for i in val_indices]
    val_dataset_flags = [dataset_flags[i] for i in val_indices]
    val_masks = [valid_frames_masks[i] for i in val_indices]

    X_test_video = [video_features[i] for i in test_indices]
    X_test_audio = [audio_features[i] for i in test_indices]
    y_test = labels_categorical[test_indices]
    test_filenames = [filenames[i] for i in test_indices]
    test_dataset_flags = [dataset_flags[i] for i in test_indices]
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

    # Report dataset distribution in each split
    train_ravdess = sum(1 for d in train_dataset_flags if d == 0)
    train_cremad = sum(1 for d in train_dataset_flags if d == 1)
    val_ravdess = sum(1 for d in val_dataset_flags if d == 0)
    val_cremad = sum(1 for d in val_dataset_flags if d == 1)
    test_ravdess = sum(1 for d in test_dataset_flags if d == 0)
    test_cremad = sum(1 for d in test_dataset_flags if d == 1)

    logging.info(f"Train dataset distribution: RAVDESS={train_ravdess}, CREMA-D={train_cremad}")
    logging.info(f"Validation dataset distribution: RAVDESS={val_ravdess}, CREMA-D={val_cremad}")
    logging.info(f"Test dataset distribution: RAVDESS={test_ravdess}, CREMA-D={test_cremad}")

    return (X_train_video, X_train_audio, y_train, train_filenames, train_masks,
            X_val_video, X_val_audio, y_val, val_filenames, val_masks,
            X_test_video, X_test_audio, y_test, test_filenames, test_masks)

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

    # Pad video features
    padded_video = np.zeros((len(video_features), max_video_length, video_features[0].shape[1]))
    
    for i, video_feat in enumerate(video_features):
        # Get length of actual data
        length = min(video_feat.shape[0], max_video_length)
        # Apply mask if provided
        if valid_frames_masks is not None and i < len(valid_frames_masks):
            for j in range(length):
                if j < len(valid_frames_masks[i]) and valid_frames_masks[i][j]:
                    padded_video[i, j, :] = video_feat[j, :]
        else:
            # No mask, just copy the sequence
            padded_video[i, :length, :] = video_feat[:length, :]

    # Pad audio features
    padded_audio = np.zeros((len(audio_features), max_audio_length, audio_features[0].shape[1]))
    
    for i, audio_feat in enumerate(audio_features):
        length = min(audio_feat.shape[0], max_audio_length)
        padded_audio[i, :length, :] = audio_feat[:length, :]

    return padded_video, padded_audio

def compute_class_weights(y_train):
    """Compute class weights to handle class imbalance.

    Args:
        y_train: One-hot encoded training labels.

    Returns:
        Dictionary mapping class indices to weights.
    """
    # Convert one-hot to class indices
    y_indices = np.argmax(y_train, axis=1)

    # Compute class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_indices),
        y=y_indices
    )

    # Convert to dictionary
    weights_dict = {i: weights[i] for i in range(len(weights))}

    logging.info(f"Computed class weights: {weights_dict}")
    return weights_dict

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train a 6-class branched LSTM model on combined RAVDESS and CREMA-D datasets')
    parser.add_argument('--ravdess-dir', type=str, default='ravdess_features',
                       help='Directory containing processed RAVDESS data')
    parser.add_argument('--cremad-dir', type=str, default='crema_d_features',
                       help='Directory containing processed CREMA-D data')
    parser.add_argument('--model-dir', type=str, default='models/branched_6class',
                       help='Directory to save model checkpoints')
    parser.add_argument('--eval-dir', type=str, default='model_evaluation/branched_6class',
                       help='Directory to save evaluation results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--use-masking', action='store_true',
                       help='Use masking for invalid frames')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--cremad-only', action='store_true',
                       help='Use only CREMA-D dataset')
    parser.add_argument('--ravdess-only', action='store_true',
                       help='Use only RAVDESS dataset')
    parser.add_argument('--use-fixed-padding', action='store_true',
                       help='Use fixed-length padding instead of dynamic batch-wise padding')
    parser.add_argument('--max-sequence-length', type=int, default=None,
                       help='Maximum sequence length for padding (if None, use the maximum length in the data)')
    parser.add_argument('--max-audio-sequence-length', type=int, default=None,
                       help='Maximum audio sequence length for padding (if None, use the maximum length in the data)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Emotion class names for the 6-class model
    CLASS_NAMES = EMOTION_LABELS

    # Load RAVDESS data
    ravdess_data = None
    if not args.cremad_only:
        ravdess_data = load_ravdess_data(args.ravdess_dir)

    # Load CREMA-D data
    cremad_data = None
    if not args.ravdess_only:
        cremad_data = load_crema_d_data(args.cremad_dir)

    # Combine datasets or prepare single dataset
    if args.ravdess_only:
        # Extract individual components from ravdess_data
        video_features, audio_features, labels, original_labels, filenames, valid_frames_masks = ravdess_data
        # Create dataset flags as all zeros (indicating RAVDESS)
        dataset_flags = [0] * len(labels)
        # Package data in the same format as combine_datasets output
        combined_data = (video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks)
        logging.info("Using RAVDESS dataset only")
    elif args.cremad_only:
        # Extract individual components from cremad_data 
        video_features, audio_features, labels, original_labels, filenames, valid_frames_masks = cremad_data
        # Create dataset flags as all ones (indicating CREMA-D)
        dataset_flags = [1] * len(labels)
        # Package data in the same format as combine_datasets output
        combined_data = (video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks)
        logging.info("Using CREMA-D dataset only")
    else:
        combined_data = combine_datasets(ravdess_data, cremad_data)
        logging.info("Using combined RAVDESS and CREMA-D datasets")

    if combined_data is None:
        logging.error("No valid data found. Exiting.")
        sys.exit(1)

    # Split data into train, validation, and test sets
    video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks = combined_data

    dataset_split = create_dataset_split(
        video_features, audio_features, labels, dataset_flags, filenames, valid_frames_masks,
        test_size=0.2, val_size=0.1, random_state=args.random_seed
    )

    (X_train_video, X_train_audio, y_train, train_filenames, train_masks,
     X_val_video, X_val_audio, y_val, val_filenames, val_masks,
     X_test_video, X_test_audio, y_test, test_filenames, test_masks) = dataset_split

    # Log sequence length statistics
    train_video_lengths = [v.shape[0] for v in X_train_video]
    train_audio_lengths = [a.shape[0] for a in X_train_audio]
    
    logging.info("Training data sequence length statistics:")
    logging.info(f"Video frames: min={min(train_video_lengths)}, max={max(train_video_lengths)}, "
                f"avg={np.mean(train_video_lengths):.1f}, median={np.median(train_video_lengths)}")
    logging.info(f"Audio frames: min={min(train_audio_lengths)}, max={max(train_audio_lengths)}, "
                f"avg={np.mean(train_audio_lengths):.1f}, median={np.median(train_audio_lengths)}")

    # Use either fixed padding or dynamic batch-wise padding
    if args.use_fixed_padding:
        # Compute global maximum lengths for video and audio
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
        
        # If specific max lengths are specified, use those instead
        max_video_length = args.max_sequence_length if args.max_sequence_length is not None else global_max_video_length
        max_audio_length = args.max_audio_sequence_length if args.max_audio_sequence_length is not None else global_max_audio_length
        
        logging.info(f"Using fixed padding with max video length: {max_video_length}")
        logging.info(f"Using fixed padding with max audio length: {max_audio_length}")
        
        # Pad all splits with the same global maximum lengths
        X_train_video_padded, X_train_audio_padded = prepare_data_for_training(
            X_train_video, X_train_audio, train_masks,
            max_video_length=max_video_length,
            max_audio_length=max_audio_length
        )
        X_val_video_padded, X_val_audio_padded = prepare_data_for_training(
            X_val_video, X_val_audio, val_masks,
            max_video_length=max_video_length,
            max_audio_length=max_audio_length
        )
        X_test_video_padded, X_test_audio_padded = prepare_data_for_training(
            X_test_video, X_test_audio, test_masks,
            max_video_length=max_video_length,
            max_audio_length=max_audio_length
        )
        
        # Define model architecture
        video_input_shape = X_train_video_padded.shape[1:]
        audio_input_shape = X_train_audio_padded.shape[1:]
    else:
        # Using dynamic batch-wise padding
        logging.info("Using dynamic batch-wise padding")
        
        # Get a sample from each dataset to determine feature dimensions
        video_feature_dim = X_train_video[0].shape[1]
        audio_feature_dim = X_train_audio[0].shape[1]
        
        # Define input shapes with None for the time dimension (variable length)
        video_input_shape = (None, video_feature_dim)
        audio_input_shape = (None, audio_feature_dim)
        
        logging.info(f"Using variable-length input with video feature dim: {video_feature_dim}")
        logging.info(f"Using variable-length input with audio feature dim: {audio_feature_dim}")

    # Define and compile the model
    num_classes = 6  # 6 classes in our schema (no Surprised)
    logging.info(f"Video input shape: {video_input_shape}")
    logging.info(f"Audio input shape: {audio_input_shape}")
    logging.info(f"Number of classes: {num_classes}")
    
    # Anti-overfitting parameters
    logging.info("Using anti-overfitting measures for FaceNet features")
    model = define_branched_model(
        video_input_shape, audio_input_shape, num_classes,
        use_masking=args.use_masking,
        regularization_strength=0.001,  # L2 regularization
        add_noise=True,                 # Add Gaussian noise for better generalization
        reduce_capacity=False           # Keep model capacity but add regularization
    )

    compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    # Compute class weights to handle imbalance
    class_weights = compute_class_weights(y_train)

    # Train the model
    if args.use_fixed_padding:
        # Train with fixed padding
        print("Training model with fixed padding...")
        from tensorflow.keras.callbacks import TensorBoard
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(args.model_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(args.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Create model directory if it doesn't exist
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Train the model
        history = model.fit(
            [X_train_video_padded, X_train_audio_padded],
            y_train,
            validation_data=([X_val_video_padded, X_val_audio_padded], y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate the model
        metrics = model.evaluate(
            [X_test_video_padded, X_test_audio_padded],
            y_test,
            verbose=1
        )
    else:
        # Train with dynamic batch-wise padding
        print("Training model with dynamic batch-wise padding...")
        
        # Create data generators
        train_generator = SequenceDataGenerator(
            X_train_video, X_train_audio, y_train,
            batch_size=args.batch_size, shuffle=True, masks=train_masks
        )
        
        val_generator = ValidationDataGenerator(
            X_val_video, X_val_audio, y_val,
            batch_size=args.batch_size, masks=val_masks
        )
        
        test_generator = ValidationDataGenerator(
            X_test_video, X_test_audio, y_test,
            batch_size=args.batch_size, masks=test_masks
        )
        
        # Train the model using the custom generator functions
        model, history = train_model_with_generator(
            model,
            train_generator,
            val_generator,
            model_dir=args.model_dir,
            epochs=args.epochs,
            class_names=CLASS_NAMES,
            class_weights=class_weights
        )
        
        # Evaluate the model using the custom evaluation function
        metrics = evaluate_model_with_generator(
            model,
            test_generator,
            class_names=CLASS_NAMES,
            output_dir=args.eval_dir
        )

    print("Training complete!")

if __name__ == "__main__":
    main()
