#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains an audio-only model using LSTM/BiLSTM on precomputed CNN audio features.
(Version 2 - created to bypass potential caching issues).
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Force CPU-only mode to avoid GPU memory conflict with video training
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Running in CPU-only mode to avoid GPU memory conflicts")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Masking, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from tqdm import tqdm
# from spectrogram_cnn_pooling_generator import SpectrogramCnnPoolingGenerator # OLD generator
from precomputed_cnn_audio_generator import PrecomputedCnnAudioGenerator # Import the audio-only generator

# --- Overfit Test Parameters ---
OVERFIT_TEST_ENABLED = False # Set to False to run normal training
OVERFIT_SUBSET_SIZE = 100
OVERFIT_EPOCHS = 30
OVERFIT_LR = 5e-4 # Fixed LR for overfit test
OVERFIT_L2 = 0.0 # No L2 for overfit test
OVERFIT_MAX_NORM = None # No MaxNorm for overfit test
OVERFIT_DROPOUT = 0.0 # No Dropout for overfit test
# --- Original Parameters ---
# Global variables (match train_audio_pooling_lstm.py where applicable)
BATCH_SIZE = 12 # Reduced to handle memory constraints
EPOCHS = 150 # Increased slightly as CNN features might take longer to converge
NUM_CLASSES = 6
PATIENCE = 20 # Increased patience slightly
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0005 # Slightly reduced LR might be beneficial

# Model specific parameters (match train_audio_pooling_lstm.py)
LSTM_UNITS = [96, 48] # Reduced units to fit in memory
DENSE_UNITS = [256, 128]

print("AUDIO ONLY CNN LSTM V2") # Updated identifier
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# --- Learning Rate Scheduler (copied from train_audio_pooling_lstm.py) ---
class WarmUpCosineDecayScheduler(Callback):
    """ Implements warm-up with cosine decay learning rate scheduling. """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        # Removed hasattr check - let subsequent line fail if 'lr' is missing
        # if not hasattr(self.model.optimizer, 'lr'): raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch < self.warmup_epochs: learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs; epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        # Use assign() method for TF 2.x compatibility
        self.model.optimizer.learning_rate.assign(learning_rate)
        # tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate) # Old method
        self.learning_rates.append(learning_rate); print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# --- Model Definition (Audio-Only) ---
def create_combined_lstm_model(audio_feature_dim): # Use a clear parameter name for audio dim
    """
    Creates an audio-only LSTM model using precomputed CNN features.
    """
    print("Creating audio-only LSTM model (Precomputed CNN Audio):") # Updated print
    print("- Audio feature dimension:", audio_feature_dim) # Use the new parameter name
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    # Adjust print based on mode
    if OVERFIT_TEST_ENABLED:
        print(f"- Learning rate: {OVERFIT_LR} (Fixed for overfit test)")
    else:
        print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    # Determine parameters based on overfit test flag
    l2_reg = OVERFIT_L2 if OVERFIT_TEST_ENABLED else L2_REGULARIZATION
    max_norm = OVERFIT_MAX_NORM if OVERFIT_TEST_ENABLED else MAX_NORM_CONSTRAINT
    dropout_rate = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.3 # Original LSTM dropout
    recurrent_dropout_rate = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.2 # Original LSTM recurrent dropout
    dense_dropout_1 = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.4 # Original first dense dropout
    dense_dropout_2 = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.5 # Original second dense dropout
    dense_dropout_3 = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.5 # Original third dense dropout
    dense_dropout_4 = OVERFIT_DROPOUT if OVERFIT_TEST_ENABLED else 0.4 # Original fourth dense dropout
    current_lr = OVERFIT_LR if OVERFIT_TEST_ENABLED else LEARNING_RATE

    if OVERFIT_TEST_ENABLED:
        print("\n--- RUNNING OVERFIT TEST CONFIGURATION ---")
        print(f"- L2 Reg: {l2_reg}")
        print(f"- Max Norm: {max_norm}")
        print(f"- Dropout Rates: {dropout_rate}, {recurrent_dropout_rate}, {dense_dropout_1}, {dense_dropout_2}, {dense_dropout_3}, {dense_dropout_4}")
        print("-------------------------------------------\n")

    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input') # Use function parameter
    masked_input = Masking(mask_value=0.0)(audio_input) # Use audio_input

    x = Bidirectional(LSTM(
        LSTM_UNITS[0], return_sequences=True, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
        kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg/2),
        kernel_constraint=MaxNorm(max_norm) if max_norm else None,
        recurrent_constraint=MaxNorm(max_norm) if max_norm else None
    ))(masked_input)
    x = Dropout(dense_dropout_1)(x) # Use adjusted dropout

    x = Bidirectional(LSTM(
        LSTM_UNITS[1], return_sequences=False, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
        kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg/2),
        kernel_constraint=MaxNorm(max_norm) if max_norm else None,
        recurrent_constraint=MaxNorm(max_norm) if max_norm else None
    ))(x)
    x = Dropout(dense_dropout_2)(x) # Use adjusted dropout

    # Classification Head
    x = Dense(
        DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm) if max_norm else None
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(dense_dropout_3)(x) # Use adjusted dropout

    x = Dense(
        DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm) if max_norm else None
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(dense_dropout_4)(x) # Use adjusted dropout

    output = Dense(
        NUM_CLASSES, activation='softmax',
        kernel_regularizer=l2(l2_reg), kernel_constraint=MaxNorm(max_norm) if max_norm else None
    )(x)

    model = Model(inputs=audio_input, outputs=output) # Use audio_input

    optimizer = Adam(learning_rate=current_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7) # Use potentially adjusted LR
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Data Loading (Audio-Only) ---
def load_data_paths_and_labels_audio_only(cnn_audio_dir_ravdess, cnn_audio_dir_cremad):
    """Finds precomputed CNN audio feature files and extracts labels."""
    # Find precomputed CNN audio feature files
    cnn_audio_files = glob.glob(os.path.join(cnn_audio_dir_ravdess, "Actor_*", "*.npy")) + \
                      glob.glob(os.path.join(cnn_audio_dir_cremad, "*.npy"))

    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None

        # Extract label
        try:
            if "Actor_" in cnn_audio_file: # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
            else: # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None # Ensure label is None on error

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found. Ensure CNN audio preprocessing ran and paths are correct.")

    return valid_cnn_audio_files, np.array(labels)


# --- Training Function (Audio-Only) ---
def train_model():
    """Main function to train the audio-only LSTM model using precomputed CNN audio features."""
    print("Starting Precomputed CNN Audio LSTM model training (Audio-Only)...")
    print(f"- Using learning rate: {LEARNING_RATE}")

    # Define feature directories (POINT TO FIXED CNN AUDIO FEATURES on the server)
    RAVDESS_CNN_AUDIO_DIR = "/home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed" # Use FIXED features
    CREMA_D_CNN_AUDIO_DIR = "/home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed" # Use FIXED features
    # Video directories removed

    # Load file paths and labels (Audio-Only)
    try:
        cnn_audio_files, all_labels = load_data_paths_and_labels_audio_only(
            RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Split data into train/validation sets (indices for file lists)
    class_indices = [np.where(all_labels[:, i] == 1)[0] for i in range(NUM_CLASSES)]
    train_idx, val_idx = [], []
    np.random.seed(RANDOM_SEED)
    for indices in class_indices:
        np.random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        train_idx.extend(indices[:split_idx])
        val_idx.extend(indices[split_idx:])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    # Create file lists for train/val (Audio-Only)
    train_cnn_audio_files = [cnn_audio_files[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_cnn_audio_files = [cnn_audio_files[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    # --- Overfit Test Data Subset ---
    if OVERFIT_TEST_ENABLED:
        print(f"\n--- Selecting Overfit Subset (Size: {OVERFIT_SUBSET_SIZE}) ---")
        # Ensure subset size is not larger than available training data
        actual_subset_size = min(OVERFIT_SUBSET_SIZE, len(train_cnn_audio_files))
        if actual_subset_size < OVERFIT_SUBSET_SIZE:
             print(f"Warning: Requested subset size {OVERFIT_SUBSET_SIZE} > available train samples {len(train_cnn_audio_files)}. Using {actual_subset_size}.")

        # Create a random subset of training indices
        subset_indices = np.random.choice(len(train_cnn_audio_files), actual_subset_size, replace=False)
        train_cnn_audio_files_subset = [train_cnn_audio_files[i] for i in subset_indices]
        train_labels_subset = train_labels[subset_indices]

        # Use the subset for training, keep validation the same for comparison (optional)
        train_cnn_audio_files_final = train_cnn_audio_files_subset
        train_labels_final = train_labels_subset
        # For overfit test, we often don't need a large validation set, or even one at all.
        # Let's use a small validation set just to monitor basic generalization.
        val_subset_size = min(50, len(val_cnn_audio_files)) # Small validation subset
        val_subset_indices = np.random.choice(len(val_cnn_audio_files), val_subset_size, replace=False)
        val_cnn_audio_files_final = [val_cnn_audio_files[i] for i in val_subset_indices]
        val_labels_final = val_labels[val_subset_indices]
        print(f"- Using {len(train_cnn_audio_files_final)} samples for training subset.")
        print(f"- Using {len(val_cnn_audio_files_final)} samples for validation subset.")
        print("--------------------------------------------------\n")
    else:
        # Use full data if not overfit test
        train_cnn_audio_files_final = train_cnn_audio_files
        train_labels_final = train_labels
        val_cnn_audio_files_final = val_cnn_audio_files
        val_labels_final = val_labels
        print("\nTrain/Val split (Full):")
        print(f"- Train samples: {len(train_cnn_audio_files_final)}")
        print(f"- Validation samples: {len(val_cnn_audio_files_final)}")

    # --- Calculate Max Sequence Length ---
    print("\nCalculating maximum sequence length from training features...")
    max_seq_len = 0
    for f_path in tqdm(train_cnn_audio_files_final, desc="Checking lengths"):
        try:
            shape = np.load(f_path).shape
            if len(shape) >= 1: # Ensure it has at least one dimension (time)
                max_seq_len = max(max_seq_len, shape[0])
        except Exception as e:
            print(f"Warning: Could not read shape from {f_path}: {e}")
            continue
    if max_seq_len == 0:
        print("Error: Could not determine max_seq_len. Exiting.")
        sys.exit(1)
    print(f"Determined max_seq_len: {max_seq_len}")
    # --- End Calculate Max Sequence Length ---


    # Create data generators (Audio-Only)
    print("\nCreating data generators (Precomputed CNN Audio - Audio Only)...")
    # Pass the calculated max_seq_len to the generators
    train_generator = PrecomputedCnnAudioGenerator(
        train_cnn_audio_files_final, train_labels_final, batch_size=BATCH_SIZE, shuffle=True, max_seq_len=max_seq_len # Use final lists and max_seq_len
    )
    # Use final validation lists for the generator and pass max_seq_len
    val_generator = PrecomputedCnnAudioGenerator(
        val_cnn_audio_files_final, val_labels_final, batch_size=BATCH_SIZE, shuffle=False, max_seq_len=max_seq_len
    )

    # Get audio dimension from generator
    cnn_audio_dim = train_generator.cnn_audio_dim # Use cnn_audio_dim from generator
    if cnn_audio_dim == 0:
        print("Error: Could not determine CNN audio feature dimension from generator.")
        sys.exit(1)

    # Create the audio-only model
    model = create_combined_lstm_model(cnn_audio_dim) # Pass audio dim
    model.summary()

    # Define callbacks based on mode
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if OVERFIT_TEST_ENABLED:
        current_epochs = OVERFIT_EPOCHS
        checkpoint_dir = f"models/overfit_test_cnn_lstm_audio_only_v2_{timestamp}"
        checkpoint_filename = os.path.join(checkpoint_dir, 'overfit_test_best.keras')
        print(f"Saving overfit test checkpoints to: {checkpoint_dir}")
        # Minimal callbacks for overfit test: just save the best based on train accuracy
        # No early stopping needed, we want to see if it *can* overfit
        checkpoint = ModelCheckpoint(checkpoint_filename, monitor='accuracy', save_best_only=True, mode='max', verbose=1)
        callbacks_list = [checkpoint]
        print("\nStarting Overfit Test Training...")
    else:
        current_epochs = EPOCHS
        checkpoint_dir = f"models/precomputed_cnn_lstm_audio_only_v2_{timestamp}" # Updated dir name for V2
        checkpoint_filename = os.path.join(checkpoint_dir, 'best_model_precomputed_cnn_audio_lstm_audio_only_v2.keras') # Updated checkpoint name for V2
        print(f"Saving checkpoints to: {checkpoint_dir}")
        checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        # Longer patience for early stopping in full run
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min')
        lr_scheduler = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE, total_epochs=current_epochs, warmup_epochs=10, min_learning_rate=5e-6)
        callbacks_list = [checkpoint, early_stopping, reduce_lr, lr_scheduler]
        print("\nStarting training (Audio-Only V2 - Full)...") # Updated print

    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    history = model.fit(
        train_generator, epochs=current_epochs, validation_data=val_generator, # Use current_epochs
        callbacks=callbacks_list, verbose=1 # Use adjusted callbacks
    )
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

    print("\nEvaluating the best model...")
    if os.path.exists(checkpoint_filename):
        print(f"Loading best weights from {checkpoint_filename}")
        model.load_weights(checkpoint_filename)
    else:
        print("Warning: Checkpoint file not found. Evaluating model with final weights.")

    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy ({checkpoint_filename}): {accuracy:.4f}")

if __name__ == '__main__':
    train_model()
