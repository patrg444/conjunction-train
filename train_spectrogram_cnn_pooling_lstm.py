import os
import sys
import numpy as np
import tensorflow as tf
import time
import glob
import json
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Masking, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.constraints import MaxNorm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
RAVDESS_CNN_AUDIO_DIR = "/home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed"  # Use FIXED features
CREMA_D_CNN_AUDIO_DIR = "/home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed"  # Use FIXED features

# Configuration
EPOCHS = 50
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LEARNING_RATE = 0.0005  # Small learning rate with our special scheduler
L2_REG = 0.002
MAX_NORM = 3.0
DROPOUT_RATE = 0.4
LSTM_UNITS = [128, 64]
DENSE_UNITS = [256, 128]
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 8
MODEL_CHECKPOINT_DIR = 'models'

# Global config
tf.random.set_seed(42)
np.random.seed(42)

# Print version info for debugging
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
        if not hasattr(self.model.optimizer, "learning_rate"): 
            raise ValueError("Optimizer must have a learning_rate attribute.")
        
        if epoch < self.warmup_epochs: 
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# --- Model Definition (Audio-Only) ---
def create_combined_lstm_model(audio_feature_dim): # Use a clear parameter name for audio dim
    """
    Creates an audio-only LSTM model using precomputed CNN features.
    """
    print("Creating audio-only LSTM model (Precomputed CNN Audio):") # Updated print
    print(f"- Audio feature dimension: {audio_feature_dim}")
    print(f"- L2 regularization strength: {L2_REG}")
    print(f"- Weight constraint: {MAX_NORM}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")
    
    # Define inputs with masking - use None for variable input sequence length 
    audio_input = Input(shape=(None, audio_feature_dim), name="audio_input")
    
    # Bidirectional LSTMs for audio
    not_equal = tf.not_equal(audio_input, 0)
    # any_not_equal = tf.reduce_any(not_equal, axis=-1)
    mask = Masking()(audio_input)
    
    # First Bidirectional LSTM with width (seq_len, features)
    lstm1 = Bidirectional(LSTM(LSTM_UNITS[0], 
                              return_sequences=True,
                              dropout=DROPOUT_RATE,
                              kernel_regularizer=l2(L2_REG),
                              kernel_constraint=MaxNorm(MAX_NORM),
                              recurrent_constraint=MaxNorm(MAX_NORM)))
                              
    lstm1_out = lstm1(mask, mask=any_not_equal)
    lstm1_out = Dropout(DROPOUT_RATE)(lstm1_out)
    
    # Second Bidirectional LSTM to compress sequence to a single vector
    lstm2 = Bidirectional(LSTM(LSTM_UNITS[1], 
                              kernel_regularizer=l2(L2_REG), 
                              kernel_constraint=MaxNorm(MAX_NORM),
                              recurrent_constraint=MaxNorm(MAX_NORM)))
                              
    lstm2_out = lstm2(lstm1_out, mask=any_not_equal)
    lstm2_out = Dropout(DROPOUT_RATE)(lstm2_out)
    
    # Dense layers for final classification with LN
    dense1 = Dense(DENSE_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REG))(lstm2_out)
    dense1 = LayerNormalization()(dense1)  # LN for better training stability
    dense1 = Dropout(DROPOUT_RATE)(dense1)
    
    dense2 = Dense(DENSE_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REG))(dense1)
    dense2 = LayerNormalization()(dense2)  # LN for better training stability
    dense2 = Dropout(DROPOUT_RATE)(dense2)
    
    # Final output layer with 6 emotions (we're using the filtered set as in our other models)
    output = Dense(6, activation='softmax', kernel_regularizer=l2(L2_REG))(dense2)
    
    model = Model(inputs=audio_input, outputs=output)
    
    return model

# --- Audio Data Generator (Precomputed CNN Features) ---
class PrecomputedCnnAudioGenerator:
    def __init__(self, cnn_audio_files, labels, batch_size, shuffle=True):
        self.cnn_audio_files = cnn_audio_files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(cnn_audio_files))
        self.label_encoder = LabelEncoder().fit(labels)
        self.n_classes = len(self.label_encoder.classes_)
        
        # Detect audio feature dimension from first sample
        # This will help us ensure we set up the right model input shape
        first_file = self.cnn_audio_files[0]
        audio_features = np.load(first_file)
        self.audio_feature_dim = audio_features.shape[1]
        
        if len(audio_features.shape) != 2:
            raise ValueError(f"Expected features to have shape (sequence_length, features), got {audio_features.shape}")
        
        # Get actual CNN feature rate (not critical but useful for reporting)
        cnn_file_name = os.path.basename(first_file)
        if '_mel_' in cnn_file_name and '_step_' in cnn_file_name:
            # Extract step size from filename if available
            step_parts = cnn_file_name.split('_step_')[1].split('_')[0]
            self.cnn_feature_step = float(step_parts) / 1000  # Convert to seconds
        else:
            # Default common CNN feature step size
            self.cnn_feature_step = 0.256  # 256ms 
        
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.cnn_audio_files) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find max sequence length in this batch to pad to
        max_seq_len = 0
        for i in indexes:
            audio_features = np.load(self.cnn_audio_files[i])
            max_seq_len = max(max_seq_len, audio_features.shape[0])
            
        # Prepare batch arrays
        batch_size = len(indexes)
        audio_batch = np.zeros((batch_size, max_seq_len, self.audio_feature_dim))
        labels_batch = np.zeros((batch_size, self.n_classes), dtype=np.float32)
        
        # Load and preprocess data
        for i, idx in enumerate(indexes):
            # Load CNN audio features
            audio_features = np.load(self.cnn_audio_files[idx])
            seq_len = audio_features.shape[0]
            
            # Set the actual data (the rest remains zeros for padding)
            audio_batch[i, :seq_len, :] = audio_features
            
            # One-hot encode the label
            label_idx = self.label_encoder.transform([self.labels[idx]])[0]
            labels_batch[i, label_idx] = 1.0
            
        return audio_batch, labels_batch
    
    def get_class_weights(self):
        # Get class weights for imbalanced training data
        integer_labels = self.label_encoder.transform(self.labels)
        class_weights = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(integer_labels),
            y=integer_labels
        )
        return dict(enumerate(class_weights))

# --- Main Training Function ---
def train_model():
    print("EARLY FUSION (SPECTROGRAM CNN POOLING) HYBRID MODEL WITH LSTM")
    
    print("Starting Precomputed CNN Audio LSTM model training (Audio-Only)...")
    print(f"- Using learning rate: {LEARNING_RATE}")
    
    # ==========================================================
    # Step 1: Load CNN audio features and extract labels
    # ==========================================================
    cnn_audio_files = []
    
    # Load CNN audio feature files
    for dir_path in [RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR]:
        if os.path.exists(dir_path):
            cnn_audio_files.extend(sorted(glob.glob(os.path.join(dir_path, '*.npy'))))
    
    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    
    # Extract labels from filenames
    labels = []
    valid_cnn_audio_files = []
    
    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        filename = os.path.basename(cnn_audio_file)
        
        # Extract emotion label from filename
        try:
            if 'ravdess' in cnn_audio_file.lower():
                # RAVDESS format: e.g., '03-01-07-02-01-01-24_mel_256_hop_32_step_256.npy'
                # Label is the 3rd field (07 = happy, 06 = fear, 05 = disgust, etc.)
                parts = filename.split('_')[0].split('-')
                label_code = int(parts[2])
                
                # Map RAVDESS emotion codes to standardized labels
                label_map = {
                    1: 'neutral',
                    2: 'calm',  # Skip or map
                    3: 'happy',
                    4: 'sad',
                    5: 'angry',
                    6: 'fear',
                    7: 'disgust',
                    8: 'surprise'
                }
                
                # Skip 'calm' as we don't use it in most models
                if label_code == 2:
                    continue
                    
                label = label_map.get(label_code)
                
            elif 'crema' in cnn_audio_file.lower():
                # CREMA-D format: e.g., '1076_MTI_SAD_XX_mel_256_hop_32_step_256.npy'
                # Label is the 3rd field (SAD, HAP, FEA, etc.)
                parts = filename.split('_')
                label_code = parts[2].upper()
                
                # Map CREMA-D emotion codes to standardized labels
                label_map = {
                    'NEU': 'neutral',
                    'HAP': 'happy',
                    'SAD': 'sad',
                    'ANG': 'angry',
                    'FEA': 'fear',
                    'DIS': 'disgust'
                }
                
                label = label_map.get(label_code)
            else:
                # Skip files that don't belong to our known datasets
                continue
                
            # Only proceed with valid and needed emotions
            if label in ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']:
                labels.append(label)
                valid_cnn_audio_files.append(cnn_audio_file)
        except (IndexError, ValueError):
            # Skip files that don't have the expected format
            continue
    
    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {len(cnn_audio_files) - len(valid_cnn_audio_files)} due to label parsing issues.")
    
    # ==========================================================
    # Step 2: Train/Validation split
    # ==========================================================
    train_files, val_files, train_labels, val_labels = train_test_split(
        valid_cnn_audio_files, labels, test_size=VALIDATION_SPLIT, random_state=42, 
        stratify=labels  # Keep class distribution consistent
    )
    
    print("\nTrain/Val split:")
    print(f"- Train samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    # ==========================================================
    # Step 3: Create data generators
    # ==========================================================
    print("\nCreating data generators (Precomputed CNN Audio - Audio Only)...")
    
    # Create train generator
    print("Determining CNN audio feature dimension from first valid sample...")
    train_gen = PrecomputedCnnAudioGenerator(
        train_files, 
        train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    print(f"  Detected CNN feature shape: ({train_gen.audio_feature_dim} -> Dim: {train_gen.audio_feature_dim}")
    print(f"\nCreated PrecomputedCnnAudioGenerator (Audio-Only):")
    print(f"- Samples: {len(train_files)}")
    print(f"- Precomputed CNN Audio Dim: {train_gen.audio_feature_dim}")
    print(f"- Max Sequence Len: Dynamic")
    print(f"- CNN Feature Step (Informational): {train_gen.cnn_feature_step*1000} ms")
    
    # Create validation generator
    val_gen = PrecomputedCnnAudioGenerator(
        val_files,
        val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False  # No need to shuffle validation data
    )
    print(f"Determining CNN audio feature dimension from first valid sample...")
    print(f"  Detected CNN feature shape: ({val_gen.audio_feature_dim} -> Dim: {val_gen.audio_feature_dim}")
    print(f"\nCreated PrecomputedCnnAudioGenerator (Audio-Only):")
    print(f"- Samples: {len(val_files)}")
    print(f"- Precomputed CNN Audio Dim: {val_gen.audio_feature_dim}")
    print(f"- Max Sequence Len: Dynamic")
    print(f"- CNN Feature Step (Informational): {val_gen.cnn_feature_step*1000} ms")
    
    # ==========================================================
    # Step 4: Create and compile model
    # ==========================================================
    model = create_combined_lstm_model(train_gen.audio_feature_dim)
    
    # Configure training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # ==========================================================
    # Step 5: Set up callbacks and train
    # ==========================================================
    # Create checkpoint directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{MODEL_CHECKPOINT_DIR}/precomputed_cnn_lstm_audio_only_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Saving checkpoints to: {checkpoint_dir}")
    
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model-ep{epoch:03d}-loss{val_loss:.4f}-acc{val_accuracy:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        
        # Early stopping on validation accuracy plateau
        EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate scheduler with warmup
        WarmUpCosineDecayScheduler(
            learning_rate_base=LEARNING_RATE,
            total_epochs=EPOCHS,
            warmup_epochs=5,
            min_learning_rate=1e-7
        )
    ]
    
    # Get class weights to handle imbalanced data
    class_weights = train_gen.get_class_weights()
    
    print("\nStarting training (Audio-Only)...")
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # ==========================================================
    # Step 6: Save final model and history
    # ==========================================================
    model.save(os.path.join(checkpoint_dir, 'final_model.h5'))
    
    # Save history as JSON for later analysis
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
        }, f, indent=4)
    
    print(f"\nTraining complete! Model saved to {checkpoint_dir}")
    
    return model, history

if __name__ == "__main__":
    train_model()
