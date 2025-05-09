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
# Change these to local directories for testing
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed"  # Local path for testing
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"  # Local path for testing

# Configuration
EPOCHS = 50
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LEARNING_RATE = 0.0005  # Small learning rate with our special scheduler
L2_REG = 0.002
MAX_NORM = 3.0
DROPOUT_RATE = 0.4
LSTM_UNITS = [128, 64]  # Fixed missing comma
DENSE_UNITS = [256, 128]  # Fixed missing comma
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 8
MODEL_CHECKPOINT_DIR = 'models'

# Global config
tf.random.set_seed(42)
np.random.seed(42)

# Print version info for debugging
print("TensorFlow version:", tf.__version__)  # Fixed missing comma
print("NumPy version:", np.__version__)  # Fixed missing comma
print("Python version:", sys.version)  # Fixed missing comma
print("EARLY FUSION (SPECTROGRAM CNN POOLING) HYBRID MODEL WITH LSTM")

# --- Learning Rate Scheduler (copied from train_audio_pooling_lstm.py) ---
class WarmUpCosineDecayScheduler(Callback):
    """ Implements warm-up with cosine decay learning rate scheduling. """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):  # Fixed missing commas
        super(WarmUpCosineDecayScheduler, self).__init__()  # Fixed missing comma
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):  # Fixed missing comma
        if not hasattr(self.model.optimizer, "learning_rate"):  # Fixed missing comma
            raise ValueError("Optimizer must have a learning_rate attribute.")

        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)  # Fixed missing comma
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# --- Model Definition (Audio-Only) ---
def create_combined_lstm_model(audio_feature_dim): # Use a clear parameter name for audio dim
    """
    Creates an audio-only LSTM model using precomputed CNN features.
    """
    print("Creating audio-only LSTM model (Precomputed CNN Audio):")  # Updated print
    print(f"- Audio feature dimension: {audio_feature_dim}")
    print(f"- L2 regularization strength: {L2_REG}")
    print(f"- Weight constraint: {MAX_NORM}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    # Define inputs with masking - use None for variable input sequence length
    audio_input = Input(shape=(None, audio_feature_dim), name="audio_input")  # Fixed missing comma

    # Using Masking layer instead of custom TF ops that cause issues
    mask = Masking()(audio_input)

    # First Bidirectional LSTM with width (seq_len features)
    lstm1 = Bidirectional(LSTM(LSTM_UNITS[0],
                              return_sequences=True,
                              dropout=DROPOUT_RATE,
                              kernel_regularizer=l2(L2_REG),
                              kernel_constraint=MaxNorm(MAX_NORM),
                              recurrent_constraint=MaxNorm(MAX_NORM)
                              ))(mask)

    # Normalize activations
    lstm1 = LayerNormalization()(lstm1)

    # Second Bidirectional LSTM stack
    lstm2 = Bidirectional(LSTM(LSTM_UNITS[1],
                              return_sequences=False,  # Get final state
                              dropout=DROPOUT_RATE,
                              kernel_regularizer=l2(L2_REG),
                              kernel_constraint=MaxNorm(MAX_NORM),
                              recurrent_constraint=MaxNorm(MAX_NORM)
                              ))(lstm1)
    
    # Normalize final LSTM output
    lstm2 = LayerNormalization()(lstm2)
    
    # Add dense layers for emotion classification
    x = Dense(DENSE_UNITS[0], activation='relu', 
              kernel_regularizer=l2(L2_REG),
              kernel_constraint=MaxNorm(MAX_NORM))(lstm2)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(DENSE_UNITS[1], activation='relu', 
              kernel_regularizer=l2(L2_REG),
              kernel_constraint=MaxNorm(MAX_NORM))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Final classification layer (softmax)
    predictions = Dense(8, activation='softmax')(x)  # 8 classes (Ravdess + CremD)
    
    # Create model
    model = Model(inputs=audio_input, outputs=predictions)
    
    # Return completed model
    return model

# --- Data Generator for precomputed CNN audio features (CNN pooling) ---
class PrecomputedCnnAudioGenerator:
    def __init__(self, files, labels, batch_size=32, shuffle=True, audio_feature_dim=None):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.audio_feature_dim = audio_feature_dim
        self.indexes = np.arange(len(self.files))
        self.on_epoch_end()
        print(f"\nCreated PrecomputedCnnAudioGenerator (Audio-Only):")
        print(f"- Samples: {len(self.files)}")

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, batch_idx):
        # Get batch indexes
        indexes = self.indexes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_files = [self.files[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]
        
        # Generate data
        X, y = self._generate_batch(batch_files, batch_labels)
        return X, y

    def _generate_batch(self, batch_files, batch_labels):
        # Load CNN audio features and convert to one-hot
        audio_features = []
        for file_path in batch_files:
            features = np.load(file_path)
            audio_features.append(features)
            
        # Create numpy arrays for batch
        max_len = max(f.shape[0] for f in audio_features)
        
        # Create batch arrays with padding
        batch_x = np.zeros((len(batch_files), max_len, self.audio_feature_dim))
        
        # Fill batch arrays
        for i, features in enumerate(audio_features):
            batch_x[i, :features.shape[0], :] = features
            
        # Convert labels to one-hot encoding
        batch_y = np.array(batch_labels)
        
        return batch_x, batch_y

# --- Emotion Label Extraction Functions ---
def extract_emotion_ravdess(filename):
    """Extract emotion label from RAVDESS audio filename."""
    filename = Path(filename).stem  # Get filename without extension
    parts = filename.split('-')
    
    # Third position is emotion (1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised)
    if len(parts) >= 3:
        emotion_code = int(parts[2])
        # Map to 0-7 range (standard in the project)
        emotion_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
        if emotion_code in emotion_map:
            return emotion_map[emotion_code]
    
    return None  # Can't determine emotion

def extract_emotion_crema_d(filename):
    """Extract emotion label from CREMA-D audio filename."""
    filename = Path(filename).stem  # Get filename without extension
    parts = filename.split('_')
    
    # Format: [number]_[actor][gender]_[emotion]_[level]
    if len(parts) >= 3:
        emotion_code = parts[2]
        # Map to 0-7 range (standard in the project)
        emotion_map = {
            'NEU': 0,  # Neutral
            'HAP': 2,  # Happy
            'SAD': 3,  # Sad
            'ANG': 4,  # Angry
            'FEA': 5,  # Fear
            'DIS': 6   # Disgust
        }
        if emotion_code in emotion_map:
            return emotion_map[emotion_code]
    
    return None  # Can't determine emotion

# --- Main Training Function ---
def train_model():
    """Main function to train the CNN-LSTM audio model."""
    print("Starting Precomputed CNN Audio LSTM model training (Audio-Only)...")
    print(f"- Using learning rate: {LEARNING_RATE}")
    
    # Create model directory
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    
    # Find all precomputed CNN audio features
    cnn_audio_files = []
    
    # Process RAVDESS CNN audio features
    ravdess_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "**/*.npy"), recursive=True)
    cnn_audio_files.extend(ravdess_files)
    
    # Process CREMA-D CNN audio features
    crema_d_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    cnn_audio_files.extend(crema_d_files)
    
    # Extract labels
    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    labels = []
    valid_files = []
    skipped = 0
    
    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):  # Fixed missing comma
        # Determine dataset based on path
        if RAVDESS_CNN_AUDIO_DIR in cnn_audio_file:
            emotion = extract_emotion_ravdess(cnn_audio_file)
        else:
            emotion = extract_emotion_crema_d(cnn_audio_file)
            
        if emotion is not None:
            labels.append(emotion)
            valid_files.append(cnn_audio_file)
        else:
            skipped += 1
    
    print(f"Found {len(valid_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped} due to label parsing issues.")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        valid_files, labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=labels
    )
    
    print("\nTrain/Val split:")
    print(f"- Train samples: {len(X_train)}")
    print(f"- Validation samples: {len(X_val)}")
    
    # Create data generators
    print("\nCreating data generators (Precomputed CNN Audio - Audio Only)...")
    
    # Get CNN audio feature dimension from first valid sample
    sample_data = np.load(valid_files[0])
    audio_feature_dim = sample_data.shape[1]
    print(f"  Detected CNN feature shape: ({audio_feature_dim} -> Dim: {audio_feature_dim}")
    
    # Create train/val generators
    train_gen = PrecomputedCnnAudioGenerator(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        audio_feature_dim=audio_feature_dim
    )
    
    val_gen = PrecomputedCnnAudioGenerator(
        X_val, y_val, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        audio_feature_dim=audio_feature_dim
    )
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nClass weights:", class_weight_dict)
    
    # Create combined model
    model = create_combined_lstm_model(train_gen.audio_feature_dim)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE,
        total_epochs=EPOCHS,
        warmup_epochs=5
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        verbose=1,
        min_lr=1e-7
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_CHECKPOINT_DIR, 'cnn_audio_lstm_model_{epoch:02d}_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [lr_scheduler, early_stopping, model_checkpoint]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(MODEL_CHECKPOINT_DIR, 'cnn_audio_lstm_model_final.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(MODEL_CHECKPOINT_DIR, 'cnn_audio_lstm_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")

# --- Main Entry Point ---
if __name__ == "__main__":
    train_model()
