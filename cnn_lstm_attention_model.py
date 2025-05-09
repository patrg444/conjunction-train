#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Masking, LayerNormalization
from tensorflow.keras.layers import Permute, Multiply, Lambda, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import glob
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Directory paths for CNN audio features
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed"
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"

# Emotion maps
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
RAVDESS_EMOTION_MAP = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

# Training parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001  # Increased from original 0.0005
PATIENCE = 15
TRAIN_RATIO = 0.8
L2_REGULARIZATION = 0.001  # Reduced from 0.002 to prevent overfitting
MAX_NORM_CONSTRAINT = 2.0  # Reduced for stability
LSTM_UNITS = [128, 64]
DENSE_UNITS = [256, 128]
NUM_CLASSES = 6

# Helper function to extract label from filename
def extract_label_from_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    if "Actor_" in filename:  # RAVDESS
        parts = base_name.split('-')
        try:
            emotion_code = RAVDESS_EMOTION_MAP.get(parts[2], None)
            return emotion_code, EMOTION_MAP.get(emotion_code, None) if emotion_code else None
        except:
            return None, None
    else:  # CREMA-D
        parts = base_name.split('_')
        try:
            emotion_code = parts[2]
            return emotion_code, EMOTION_MAP.get(emotion_code, None)
        except:
            return None, None

# Load CNN audio features
def load_data():
    # Find all CNN audio files
    ravdess_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "Actor_*", "*.npy"))
    cremad_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    
    # Print dataset sizes
    print(f"RAVDESS files: {len(ravdess_files)}")
    print(f"CREMA-D files: {len(cremad_files)}")
    
    # Load features and labels
    features = []
    labels = []
    
    # Process all files
    all_files = ravdess_files + cremad_files
    skipped = 0
    
    for file in tqdm(all_files, desc="Loading features"):
        # Extract label
        emotion_code, emotion_idx = extract_label_from_filename(file)
        if emotion_idx is None:
            skipped += 1
            continue
        
        # Load feature
        try:
            feature = np.load(file)
            features.append(feature)
            
            # Create one-hot encoded label
            label = np.zeros(NUM_CLASSES)
            label[emotion_idx] = 1
            labels.append(label)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            skipped += 1
    
    print(f"Successfully loaded {len(features)} features, skipped {skipped} files")
    return features, np.array(labels)

# Custom sequence generator with proper padding and normalization
class SequenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, labels, batch_size=16, shuffle=True, normalize=True):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.indexes = np.arange(len(features))
        
        # Find feature dimension from first feature
        self.feature_dim = features[0].shape[1] if features else 0
        
        # Compute mean and std if normalizing
        if normalize:
            self.compute_stats()
        
        self.on_epoch_end()
        
        print(f"Created SequenceDataGenerator:")
        print(f"- Samples: {len(features)}")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Batch size: {batch_size}")
        print(f"- Normalization: {normalize}")
    
    def __len__(self):
        return int(np.ceil(len(self.features) / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indexes
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch features and labels
        batch_features = [self.features[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]
        
        # Find the longest sequence in this batch
        max_len = max([f.shape[0] for f in batch_features])
        
        # Create padded batch
        batch_x = np.zeros((len(batch_features), max_len, self.feature_dim))
        
        # Fill batch with padded sequences
        for i, feature in enumerate(batch_features):
            # Normalize if needed
            if self.normalize:
                feature = (feature - self.mean) / (self.std + 1e-8)
            
            # Add padded sequence
            batch_x[i, :feature.shape[0], :] = feature
        
        return batch_x, batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def compute_stats(self):
        # Compute global mean and std
        all_features = np.vstack([f for f in self.features])
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0)
        
        # Replace zero std with 1 to avoid division by zero
        self.std[self.std == 0] = 1.0
        
        print(f"Feature stats:")
        print(f"- Mean range: {np.min(self.mean):.4f} to {np.max(self.mean):.4f}")
        print(f"- Std range: {np.min(self.std):.4f} to {np.max(self.std):.4f}")


# Create a simple attention mechanism
def attention_layer(inputs):
    # inputs shape: (batch_size, time_steps, hidden_size)
    
    # Attention scores: dense layer with tanh activation to 1 unit
    score = Dense(1, activation='tanh')(inputs)  # (batch_size, time_steps, 1)
    
    # Remove the last dimension
    score = Lambda(lambda x: tf.squeeze(x, axis=-1))(score)  # (batch_size, time_steps)
    
    # Apply softmax to get normalized attention weights
    attention_weights = Softmax()(score)  # (batch_size, time_steps)
    
    # Apply attention weights to the input sequence
    context = Lambda(lambda x: tf.matmul(
        tf.expand_dims(x[0], axis=1),  # (batch_size, 1, time_steps)
        x[1]                           # (batch_size, time_steps, hidden_size)
    ))([attention_weights, inputs])    # (batch_size, 1, hidden_size)
    
    # Remove the middle dimension
    context = Lambda(lambda x: tf.squeeze(x, axis=1))(context)  # (batch_size, hidden_size)
    
    return context


# Create BiLSTM model with attention for variable length sequences
def create_model(input_dim):
    # Input layer with variable sequence length
    inputs = Input(shape=(None, input_dim), name='audio_input')
    
    # Masking layer to handle padding
    masked = Masking(mask_value=0.0)(inputs)
    
    # First Bidirectional LSTM layer
    lstm1 = Bidirectional(LSTM(
        LSTM_UNITS[0], 
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        recurrent_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT),
        recurrent_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    ))(masked)
    
    lstm1_dropout = Dropout(0.4)(lstm1)
    
    # Second Bidirectional LSTM layer
    lstm2 = Bidirectional(LSTM(
        LSTM_UNITS[1],
        return_sequences=True,  # Changed to True for attention
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        recurrent_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT),
        recurrent_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    ))(lstm1_dropout)
    
    lstm2_dropout = Dropout(0.4)(lstm2)
    
    # Attention mechanism
    context_vector = attention_layer(lstm2_dropout)
    
    # Dense layers
    x = Dense(
        DENSE_UNITS[0],
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    )(context_vector)
    
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(
        DENSE_UNITS[1],
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(
        NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use legacy Adam optimizer for M1/M2 Macs
    try:
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        optimizer = LegacyAdam(learning_rate=LEARNING_RATE)
        print("Using legacy Adam optimizer")
    except ImportError:
        optimizer = Adam(learning_rate=LEARNING_RATE)
        print("Using standard Adam optimizer")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Main training function
def train_model():
    print("IMPROVED CNN-LSTM MODEL WITH ATTENTION MECHANISM")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    
    # Load data
    print("\nLoading CNN audio features...")
    features, labels = load_data()
    
    if len(features) == 0:
        print("Error: No features loaded. Check feature directories and file patterns.")
        return
    
    # Split data into train/val sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=1-TRAIN_RATIO, random_state=42, stratify=labels
    )
    
    print(f"\nTrain/Val split:")
    print(f"- Train samples: {len(train_features)}")
    print(f"- Validation samples: {len(val_features)}")
    
    # Create data generators
    train_generator = SequenceDataGenerator(
        train_features, train_labels, batch_size=BATCH_SIZE, shuffle=True, normalize=True
    )
    
    val_generator = SequenceDataGenerator(
        val_features, val_labels, batch_size=BATCH_SIZE, shuffle=False, normalize=True
    )
    
    # Create model
    model = create_model(train_generator.feature_dim)
    model.summary()
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"models/cnn_lstm_attention_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_format='h5'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE // 2,
        min_lr=1e-6,
        verbose=1,
        mode='min'
    )
    
    # Train model
    print(f"\nStarting training (CNN-LSTM with Attention)...")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")
    print(f"- Max norm constraint: {MAX_NORM_CONSTRAINT}")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate best model
    print("\nEvaluating best model...")
    best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
    
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Final validation accuracy: {accuracy:.4f}")
    
    # Save training history
    history_file = os.path.join(checkpoint_dir, 'training_history.npy')
    np.save(history_file, history.history)
    print(f"Training history saved to {history_file}")
    
    # Plot accuracy and loss curves
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'training_curves.png'))
        print(f"Training curves saved to {checkpoint_dir}/training_curves.png")
    except:
        print("Could not generate training curves plot.")
    
    return model, history, checkpoint_dir

if __name__ == "__main__":
    train_model()
