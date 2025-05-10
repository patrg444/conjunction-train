#!/bin/bash
# Script to deploy the fixed TCN model to AWS and launch training

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# AWS instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    DEPLOYING & LAUNCHING FIXED TCN MODEL WITH IMPROVEMENTS    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Create the fixed TCN model script on the remote server directly
echo -e "${YELLOW}Creating fixed TCN model script on AWS instance...${NC}"

ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > ${REMOTE_DIR}/scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py" << 'EOL'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced hybrid model with comprehensive optimizations to address the accuracy plateau:
1. Enhanced regularization - balanced L2, weight constraints
2. Skip connections - improved gradient flow throughout network
3. Fixed optimizer - compatibility fix for AWS TensorFlow versions

This implementation fixes compatibility issues and adjusts hyperparameters
to push beyond the current 83.8% accuracy barrier.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Masking
from tensorflow.keras.layers import Activation, Add, SpatialDropout1D, LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam  # Using standard Adam for better compatibility
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from synchronized_data_generator import SynchronizedAugmentationDataGenerator
from sequence_data_generator import ValidationDataGenerator

# Global variables
BATCH_SIZE = 24
EPOCHS = 125  # Extended to 125 for better convergence with warm-up
NUM_CLASSES = 6  # 6 emotions
PATIENCE = 15    # Increased patience for learning rate cycles
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
AUGMENTATION_FACTOR = 2.5  # Increased from 2.0 for more diverse augmentation
L2_REGULARIZATION = 0.002  # Adjusted from 0.0025 to reduce overfitting
MAX_NORM_CONSTRAINT = 3.0  # Weight constraint value
LEARNING_RATE = 0.0006     # Reduced from 0.0008 for better convergence

# Parameters for optimized model capacity - more balanced scaling
AUDIO_CONV_FILTERS = [64, 128]  # Reduced from [80, 160] to avoid overparameterization
AUDIO_LSTM_UNITS = [128, 64]    # Reduced from [160, 80] for better efficiency
VIDEO_TCN_FILTERS = 128         # Reduced from 160 to avoid overfitting
MERGED_DENSE_UNITS = [256, 128] # Reduced from [320, 160] for balanced fusion

print("ADVANCED HYBRID MODEL WITH TCN ARCHITECTURE AND ENHANCED REGULARIZATION")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

class WarmUpCosineDecayScheduler(Callback):
    """Implements warm-up with cosine decay learning rate scheduling."""
    
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Warm-up phase
        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay after warm-up
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.3, l2_reg=L2_REGULARIZATION):
    """Creates a TCN block with residual connection."""
    
    # Save input for skip connection
    input_tensor = x

    # First dilated convolution
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    # Second dilated convolution
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    # Residual connection
    if input_tensor.shape[-1] != filters:
        # If dimensions don't match, use 1x1 conv to adapt dimensions
        input_tensor = Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=l2(l2_reg)
        )(input_tensor)

    # Add residual connection
    result = Add()([conv2, input_tensor])
    return Activation('relu')(result)

def create_enhanced_large_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim):
    """Create a hybrid model with TCN, skip connections, and balanced regularization."""
    
    print("Creating enhanced model with balanced regularization and TCN:")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Audio Conv Filters: {AUDIO_CONV_FILTERS}")
    print(f"- Audio LSTM Units: {AUDIO_LSTM_UNITS}")
    print(f"- Video TCN Filters: {VIDEO_TCN_FILTERS}")
    print(f"- Merged Dense Units: {MERGED_DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE}")

    # Audio branch
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    audio_masked = Masking(mask_value=0.0)(audio_input)

    # First conv layer
    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[0],
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    # Skip connection
    audio_skip = Conv1D(
        AUDIO_CONV_FILTERS[1],
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(audio_x)

    # Second conv layer
    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[1],
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)

    # Add skip connection
    audio_x = Add()([audio_x, audio_skip])
    audio_x = Activation('relu')(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    # Apply bidirectional LSTM
    audio_x = Bidirectional(LSTM(
        AUDIO_LSTM_UNITS[0],
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    ))(audio_x)
    audio_x = Dropout(0.3)(audio_x)

    audio_x = Bidirectional(LSTM(
        AUDIO_LSTM_UNITS[1],
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    ))(audio_x)

    audio_x = Dense(
        AUDIO_LSTM_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = LayerNormalization()(audio_x)
    audio_x = Dropout(0.4)(audio_x)

    # Video branch with TCN
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    video_masked = Masking(mask_value=0.0)(video_input)

    # Initial projection
    video_x = Conv1D(
        VIDEO_TCN_FILTERS,
        kernel_size=1,
        padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(video_masked)

    # Apply TCN blocks with increasing dilation rates
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=1)
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=2)
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=4)
    video_x = residual_tcn_block(video_x, filters=VIDEO_TCN_FILTERS, kernel_size=3, dilation_rate=8)

    # Global pooling
    video_avg_pool = GlobalAveragePooling1D()(video_x)
    video_max_pool = GlobalMaxPooling1D()(video_x)
    video_x = Concatenate()([video_avg_pool, video_max_pool])

    video_x = Dense(
        MERGED_DENSE_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(video_x)
    video_x = LayerNormalization()(video_x)
    video_x = Dropout(0.4)(video_x)

    # Merge branches
    audio_projection = Dense(
        MERGED_DENSE_UNITS[0]//2,
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(audio_x)

    video_projection = Dense(
        MERGED_DENSE_UNITS[0]//2,
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(video_x)

    merged = Concatenate()([audio_projection, video_projection])

    # Skip connection in merged layers
    merged_skip = Dense(
        MERGED_DENSE_UNITS[0],
        kernel_regularizer=l2(L2_REGULARIZATION)
    )(merged)

    merged = Dense(
        MERGED_DENSE_UNITS[0],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.5)(merged)

    merged = Add()([merged, merged_skip])
    merged = Activation('relu')(merged)

    merged = Dense(
        MERGED_DENSE_UNITS[1],
        activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.4)(merged)

    # Output layer
    output = Dense(
        NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION),
        kernel_constraint=MaxNorm(max_value=MAX_NORM_CONSTRAINT)
    )(merged)

    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)

    # Use standard Adam optimizer for compatibility
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def process_data(path_pattern, max_files=None):
    """Load and process NPZ files containing features."""
    files = glob.glob(path_pattern)
    if max_files:
        files = files[:max_files]

    if not files:
        print("Error: No files found matching pattern: %s" % path_pattern)
        return None, None, None

    print("Processing %d files from %s" % (len(files), path_pattern))

    # Lists to hold our data
    audio_data = []
    video_data = []
    labels = []

    # Maps emotion strings to integers
    emotion_map = {
        'ANG': 0,  # Anger
        'DIS': 1,  # Disgust
        'FEA': 2,  # Fear
        'HAP': 3,  # Happy
        'NEU': 4,  # Neutral
        'SAD': 5   # Sad
    }

    # RAVDESS emotion mapping (from file naming convention)
    ravdess_emotion_map = {
        '01': 'NEU',  # neutral
        '02': 'NEU',  # calm (mapped to neutral)
        '03': 'HAP',  # happy
        '04': 'SAD',  # sad
        '05': 'ANG',  # angry
        '06': 'FEA',  # fearful
        '07': 'DIS',  # disgust
        # '08' (surprised) is intentionally excluded to skip these samples
    }

    skipped = 0
    for file_path in files:
        try:
            # Extract emotion from filename
            filename = os.path.basename(file_path)

            # Check if this is a RAVDESS file (contains hyphens in filename)
            if '-' in filename:
                # RAVDESS format: 01-01-03-01-01-01-12.npz
                # where the 3rd segment (03) is the emotion code
                parts = filename.split('-')
                if len(parts) >= 3:
                    ravdess_code = parts[2]
                    emotion_code = ravdess_emotion_map.get(ravdess_code, None)
                else:
                    emotion_code = None
            else:
                # CREMA-D format: 1001_DFA_ANG_XX.npz
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None

            if emotion_code not in emotion_map:
                # Skip files with emotions not in our map
                skipped += 1
                continue

            # Load the npz file
            data = np.load(file_path)

            # Check that both features exist
            if 'audio_features' not in data or 'video_features' not in data:
                skipped += 1
                continue

            # Get features
            audio_features = data['audio_features']
            video_features = data['video_features']

            # Skip sequences that are too short - minimum viable length
            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue

            # Append to lists - keep original length
            audio_data.append(audio_features)
            video_data.append(video_features)

            # Create one-hot encoded label
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)

        except Exception as e:
            print("Error processing file %s: %s" % (file_path, str(e)))
            skipped += 1

    print("Processed %d files, skipped %d files" % (len(audio_data), skipped))

    return audio_data, video_data, np.array(labels)

def normalize_features(features_list, mean=None, std=None):
    """Normalize a list of variable-length feature arrays."""
    if mean is None or std is None:
        # Calculate statistics from this set
        all_features = np.vstack([feat for feat in features_list])
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    # Normalize each sequence individually using the provided stats
    normalized_list = []
    for features in features_list:
        normalized = (features - mean) / std
        normalized_list.append(normalized)

    return normalized_list, mean, std

def train_model():
    """Main function to train the model."""
    print("Starting optimized balanced model training with improved regularization...")

    # Process RAVDESS data
    ravdess_pattern = "ravdess_features_facenet/*/*.npz"
    ravdess_audio, ravdess_video, ravdess_labels = process_data(ravdess_pattern)

    # Process CREMA-D data
    cremad_pattern = "crema_d_features_facenet/*.npz"
    cremad_audio, cremad_video, cremad_labels = process_data(cremad_pattern)

    # Check if either dataset loaded successfully
    if ravdess_audio is None and cremad_audio is None:
        print("Error: Failed to load any data")
        return

    # Combine available datasets
    all_audio = []
    all_video = []
    all_labels = None

    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        all_labels = ravdess_labels
        print(f"Added RAVDESS: {len(ravdess_audio)} samples")

    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels is None:
            all_labels = cremad_labels
        else:
            all_labels = np.vstack([all_labels, cremad_labels])
        print(f"Added CREMA-D: {len(cremad_audio)} samples")

    print(f"Combined: {len(all_audio)} total samples")

    # Split into train/val sets with stratification
    class_indices = [np.where(all_labels[:, i] == 1)[0] for i in range(NUM_CLASSES)]

    train_idx = []
    val_idx = []

    # For each class, take TRAIN_RATIO for training and the rest for validation
    np.random.seed(RANDOM_SEED)
    for indices in class_indices:
        np.random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        train_idx.extend(indices[:split_idx])
        val_idx.extend(indices[split_idx:])

    # Shuffle the indices
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    # Split the data using the indices
    train_audio = [all_audio[i] for i in train_idx]
    train_video = [all_video[i] for i in train_idx]
    train_labels = all_labels[train_idx]

    val_audio = [all_audio[i] for i in val_idx]
    val_video = [all_video[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("Train/Val split with stratification:")
    print("- Train samples:", len(train_audio))
    print("- Validation samples:", len(val_audio))

    # Normalize features for better training
    print("Normalizing features (training statistics only)...")
    train_audio, audio_mean, audio_std = normalize_features(train_audio)
    val_audio, _, _ = normalize_features(val_audio, mean=audio_mean, std=audio_std)
    print("Normalization complete!")

    # Get the feature dimensionality
    audio_feature_dim = train_audio[0].shape[1]
    video_feature_dim = train_video[0].shape[1]

    # Create the custom data generators with synchronized augmentation
    print("Creating data generators with synchronized augmentation...")
    train_generator = SynchronizedAugmentationDataGenerator(
        train_video, train_audio, train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augmentation_factor=AUGMENTATION_FACTOR,
        augmentation_probability=0.8
    )

    val_generator = ValidationDataGenerator(
        val_video, val_audio, val_labels,
        batch_size=BATCH_SIZE
    )

    # Create and compile the model
    model = create_enhanced_large_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim)
    print('\nModel Summary:')
    model.summary()

    # Create output directories
    model_dir = 'models/branched_regularization_sync_aug_tcn_large_fixed'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define callbacks
    checkpoint_path = os.path.join(model_dir, 'model_best.keras')

    # Add warm-up cosine decay scheduler
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE,
        total_epochs=EPOCHS,
        warmup_epochs=10,
        min_learning_rate=5e-6
    )

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=4,
            min_lr=5e-6,
            verbose=1
        ),
        lr_scheduler
    ]

    # Calculate class weights to handle imbalance
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)

    print('Using class weights to handle imbalance:', class_weights)

    # Train the model
    print('Starting training with advanced hybrid architecture and balanced regularization...')
    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save the final model
    final_model_path = os.path.join(model_dir, 'final_model.keras')
    model.save(final_model_path)
    print('Final model saved to:', final_model_path)

    # Calculate training time
    train_time = time.time() - start_time
    print('Training completed in %.2f seconds (%.2f minutes)' % (train_time, train_time/60))

    # Print training history summary
    print('Training history summary:')
    print('- Final training accuracy:', history.history['accuracy'][-1])
    print('- Final validation accuracy:', history.history['val_accuracy'][-1])
    print('- Best validation accuracy:', max(history.history['val_accuracy']))
    print('- Best validation loss:', min(history.history['val_loss']))

    return model, history

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        import traceback
        print('ERROR:', str(e))
        print(traceback.format_exc())
EOL

echo -e "${GREEN}Fixed TCN model script created successfully!${NC}"

# Create a launch script on the remote server
echo -e "${YELLOW}Creating launch script on AWS instance...${NC}"

ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > ${REMOTE_DIR}/launch_fixed_tcn.sh" << 'EOL'
#!/bin/bash
cd ~/emotion_training

echo "Setting execute permissions on training script..."
chmod +x scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py

echo "Starting fixed TCN model training..."
nohup python3 scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py > training_branched_regularization_sync_aug_tcn_large_fixed.log 2>&1 &

# Save PID for monitoring
echo $! > fixed_tcn_large_pid.txt
echo "Training process started with PID: $(cat fixed_tcn_large_pid.txt)"
echo "Logs are being written to: training_branched_regularization_sync_aug_tcn_large_fixed.log"

# Display Python version info for debugging
echo "Python and TensorFlow versions:"
python3 --version
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
EOL

# Set execute permissions on the launch script
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${REMOTE_DIR}/launch_fixed_tcn.sh"

echo -e "${GREEN}Launch script created successfully!${NC}"

# Ask if the user wants to launch the training
echo ""
echo -e "${YELLOW}The fixed TCN model script and launch script have been created on the AWS instance.${NC}"
read -p "Would you like to launch the training now? (y/n): " launch_choice

if [[ "${launch_choice,,}" == "y" ]]; then
    echo -e "${YELLOW}Launching training process on AWS instance...${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "${REMOTE_DIR}/launch_fixed_tcn.sh"
    
    echo -e "${GREEN}Training launched successfully!${NC}"
    echo -e "${YELLOW}You can monitor the training using:${NC}"
    echo -e "  ./continuous_tcn_monitoring.sh"
    echo -e "  ./continuous_tcn_monitoring_crossplatform.sh"
else
    echo -e "${YELLOW}Training not launched. You can manually launch it later with:${NC}"
    echo -e "  ssh -i \"$KEY_FILE\" \"$USERNAME@$INSTANCE_IP\" \"${REMOTE_DIR}/launch_fixed_tcn.sh\""
fi

echo ""
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
