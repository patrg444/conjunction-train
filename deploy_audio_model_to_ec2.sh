#!/usr/bin/env bash
# Deploy and train the audio-only model on EC2

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
PROJECT_DIR="/home/ubuntu/emotion_audio"

# Create and run the remote script
ssh -i "$KEY_PATH" "$EC2_INSTANCE" << 'REMOTE_SCRIPT'
set -e  # Exit on any command failure

# Create project directory if it doesn't exist
mkdir -p ~/emotion_audio
cd ~/emotion_audio

# Activate the PyTorch environment (it has TensorFlow too)
source /opt/pytorch/bin/activate

# Check if we have the dataset already
if [ ! -d "ravdess_features_facenet" ] || [ ! -d "crema_d_features_facenet" ]; then
  echo "Features directory not found. Linking from datasets directory..."
  ln -sf /home/ubuntu/datasets/ravdess_videos ravdess_features_facenet
  ln -sf /home/ubuntu/datasets/crema_d_videos crema_d_features_facenet
fi

# Clone or update the repo
if [ ! -d "conjunction-train" ]; then
  git clone https://github.com/yourusername/conjunction-train.git
else
  cd conjunction-train
  git pull
  cd ..
fi

# Copy necessary scripts
cp conjunction-train/scripts/train_audio_pooling_lstm.py .
cp conjunction-train/scripts/audio_pooling_generator.py .
cp conjunction-train/scripts/fixed_audio_pooling_generator.py .

# Enhanced train_audio_pooling_lstm.py to improve performance
cat > train_audio_pooling_lstm_aws.py << 'EOL'
#!/usr/bin/env python3
# Enhanced version of train_audio_pooling_lstm.py optimized for AWS training

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import Masking, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
import argparse
from fixed_audio_pooling_generator import AudioPoolingDataGenerator 

# Global variables
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 6
PATIENCE = 15
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0006

# Model specific parameters
LSTM_UNITS = [128, 64]  # LSTM units for the shared sequence processing
DENSE_UNITS = [256, 128]  # Dense units after pooling

print("AUDIO-ONLY EMOTION RECOGNITION MODEL WITH LSTM")
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

def create_audio_only_lstm_model(audio_feature_dim):
    """Create an audio-only LSTM model for emotion recognition."""
    print("Creating audio-only LSTM model:")
    print("- Audio feature dimension:", audio_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    # Audio Input
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    masked_input = Masking(mask_value=0.0)(audio_input)

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(
        LSTM_UNITS[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(masked_input)
    x = Dropout(0.3)(x)  

    x = Bidirectional(LSTM(
        LSTM_UNITS[1], return_sequences=False, dropout=0.3, recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(x)

    # Dense layers
    x = Dense(
        DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(
        DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(
        NUM_CLASSES, activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)

    # Create model
    model = Model(inputs=audio_input, outputs=output)

    # Optimizer
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
        return None, None

    print("Processing %d files from %s" % (len(files), path_pattern))

    audio_data = []
    labels = []

    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    skipped = 0
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            if '-' in filename:
                parts = filename.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None) if len(parts) >= 3 else None
            else:
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None

            if emotion_code not in emotion_map:
                skipped += 1
                continue

            data = np.load(file_path)
            if 'audio_features' not in data:
                skipped += 1
                continue

            audio_features = data['audio_features']

            # Basic check for minimal length
            if len(audio_features) < 5:
                skipped += 1
                continue

            audio_data.append(audio_features)
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)

        except Exception as e:
            print("Error processing file %s: %s" % (file_path, str(e)))
            skipped += 1

    print("Processed %d files, skipped %d files" % (len(audio_data), skipped))
    return audio_data, np.array(labels)

def normalize_features(features_list, mean=None, std=None):
    """Normalize a list of variable-length feature arrays."""
    if not features_list:  # Handle empty list case
        return [], None, None

    # Filter out potentially empty arrays before stacking
    valid_features = [feat for feat in features_list if feat.shape[0] > 0]
    if not valid_features:
         return features_list, None, None  # Return original list if all were empty

    if mean is None or std is None:
        all_features = np.vstack(valid_features)
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)

    std = np.where(std == 0, 1.0, std)  # Avoid division by zero

    # Normalize only non-empty arrays
    normalized_list = []
    for features in features_list:
        if features.shape[0] > 0:
            normalized_list.append((features - mean) / std)
        else:
            normalized_list.append(features)  # Keep empty arrays as they are

    return normalized_list, mean, std

def train_model(args):
    """Main function to train the audio-only LSTM model."""
    print("Starting Audio-Only LSTM model training...")
    print(f"- Using learning rate: {LEARNING_RATE}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Output directory: {args.output_dir}")

    # Set up the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    ravdess_pattern = os.path.join(args.data_dir, "ravdess_features_facenet/*/*.npz")
    cremad_pattern = os.path.join(args.data_dir, "crema_d_features_facenet/*.npz")
    
    print(f"Loading RAVDESS data from {ravdess_pattern}")
    ravdess_audio, ravdess_labels = process_data(ravdess_pattern)
    
    print(f"Loading CREMA-D data from {cremad_pattern}")
    cremad_audio, cremad_labels = process_data(cremad_pattern)

    if ravdess_audio is None and cremad_audio is None:
        print("Error: Failed to load any data")
        return

    all_audio = []
    all_labels = None
    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_labels = ravdess_labels
    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        if all_labels is None: 
            all_labels = cremad_labels
        else: 
            all_labels = np.vstack([all_labels, cremad_labels])

    print(f"Combined: {len(all_audio)} total samples")

    # Split data
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

    train_audio_raw = [all_audio[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_audio_raw = [all_audio[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("Train/Val split:")
    print("- Train samples:", len(train_audio_raw))
    print("- Validation samples:", len(val_audio_raw))

    # Normalize features
    print("Normalizing features (training statistics only)...")
    train_audio_norm, audio_mean, audio_std = normalize_features(train_audio_raw)
    val_audio_norm, _, _ = normalize_features(val_audio_raw, mean=audio_mean, std=audio_std)
    print("Normalization complete.")

    # Get feature dimension for model creation
    if not train_audio_norm:
         print("Error: No valid training data after normalization.")
         return
    audio_feature_dim = train_audio_norm[0].shape[1] if train_audio_norm[0].shape[0] > 0 else 0
    if audio_feature_dim == 0:
        print("Error: Could not determine audio feature dimension.")
        return

    # Create customized data generators
    print("Creating data generators...")
    
    class AudioOnlyGenerator(tf.keras.utils.Sequence):
        def __init__(self, audio_features, labels, batch_size=32, shuffle=True, max_seq_len=None):
            self.audio_features = audio_features
            self.labels = labels
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.max_seq_len = max_seq_len
            self.indices = np.arange(len(self.audio_features))
            if shuffle:
                np.random.shuffle(self.indices)
                
        def __len__(self):
            return int(np.ceil(len(self.audio_features) / self.batch_size))
        
        def __getitem__(self, idx):
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            audio_batch = []
            label_batch = []
            
            for i in batch_indices:
                audio = self.audio_features[i]
                if self.max_seq_len:
                    if len(audio) > self.max_seq_len:
                        audio = audio[:self.max_seq_len]
                    elif len(audio) < self.max_seq_len:
                        # Pad with zeros
                        padding = np.zeros((self.max_seq_len - len(audio), audio.shape[1]))
                        audio = np.vstack([audio, padding])
                
                audio_batch.append(audio)
                label_batch.append(self.labels[i])
                
            return np.array(audio_batch), np.array(label_batch)
        
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
    
    # Create generators
    train_generator = AudioOnlyGenerator(
        train_audio_norm, train_labels, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = AudioOnlyGenerator(
        val_audio_norm, val_labels, batch_size=BATCH_SIZE, shuffle=False
    )

    # Create the model
    model = create_audio_only_lstm_model(audio_feature_dim)
    model.summary()

    # Define callbacks
    checkpoint_filename = os.path.join(args.output_dir, 'best_audio_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE, mode='max', verbose=1, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.6, patience=PATIENCE//2, min_lr=5e-6, verbose=1, mode='min'
    )
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6
    )

    # Save normalization stats
    np.save(os.path.join(args.output_dir, 'audio_mean.npy'), audio_mean)
    np.save(os.path.join(args.output_dir, 'audio_std.npy'), audio_std)

    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler], verbose=1
    )
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

    print("Evaluating the best model...")
    if os.path.exists(checkpoint_filename):
        print(f"Loading best weights from {checkpoint_filename}")
        model.load_weights(checkpoint_filename)
    else:
        print("Warning: Checkpoint file not found. Evaluating model with final weights.")

    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy: {accuracy:.4f}")

    # Save TF model (for TensorFlow inference)
    tf_model_path = os.path.join(args.output_dir, f'audio_only_model_{accuracy:.4f}')
    model.save(tf_model_path)
    print(f"Model saved to {tf_model_path}")

    # Extract weights to PyTorch-compatible format for fusion with SlowFast
    model_state = {}
    for layer in model.layers:
        if len(layer.weights) > 0:
            model_state[layer.name] = {}
            for weight in layer.weights:
                name = weight.name.split(':')[0].split('/')[-1]
                model_state[layer.name][name] = weight.numpy()
    
    # Save as numpy
    np.save(os.path.join(args.output_dir, 'audio_model_weights.npy'), model_state)
    print(f"Model weights extracted to {os.path.join(args.output_dir, 'audio_model_weights.npy')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train audio-only emotion recognition model')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing feature data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    
    args = parser.parse_args()
    
    # Override global variables with args
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    train_model(args)
EOL

# Create a script for fixed_audio_pooling_generator.py
cat > fixed_audio_pooling_generator.py << 'EOL'
#!/usr/bin/env python3
"""
Improved AudioPoolingDataGenerator for stable training on AWS.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import glob
import os
import math

class AudioPoolingDataGenerator(Sequence):
    """
    Generates batches of audio feature sequences for emotion recognition.

    Args:
        audio_data: List of audio feature arrays
        labels: Numpy array of one-hot encoded labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle sample order each epoch
        max_seq_len: If set, all sequences will be padded/truncated to this length
    """
    def __init__(self,
                 audio_data,
                 labels,
                 batch_size=32,
                 shuffle=True,
                 max_seq_len=None):
        self.audio_data = audio_data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.indices = np.arange(len(self.audio_data))
        
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Process audio data
        batch_audio = []
        batch_labels = []
        
        for i in batch_indices:
            audio = self.audio_data[i]
            label = self.labels[i]
            
            # Handle max_seq_len if specified
            if self.max_seq_len is not None:
                if len(audio) > self.max_seq_len:
                    audio = audio[:self.max_seq_len]
                elif len(audio) < self.max_seq_len:
                    # Pad with zeros
                    padding = np.zeros((self.max_seq_len - len(audio), audio.shape[1]))
                    audio = np.vstack([audio, padding])
            
            batch_audio.append(audio)
            batch_labels.append(label)
        
        return np.array(batch_audio), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
EOL

# Create launch script
cat > run_audio_training.sh << 'EOL'
#!/usr/bin/env bash
# Train the audio-only emotion recognition model

# Set up environment
source /opt/pytorch/bin/activate

# Launch training
python train_audio_pooling_lstm_aws.py \
    --data_dir=/home/ubuntu/emotion_audio \
    --output_dir=/home/ubuntu/emotion_audio/model \
    --batch_size=32 \
    --epochs=100 \
    --lr=0.0005

# Print status
echo "Audio model training complete"
echo "Model saved to /home/ubuntu/emotion_audio/model"
EOL

# Make the run script executable
chmod +x run_audio_training.sh

# Run the training
./run_audio_training.sh
REMOTE_SCRIPT

# Check if the remote script executed successfully
if [ $? -eq 0 ]; then
  echo "Audio model training has been deployed to EC2."
  echo "The model will be trained on the EC2 instance."
  echo "You can monitor the progress by SSH-ing into the instance."
else
  echo "Failed to deploy audio model training to EC2."
fi
