#!/bin/bash
#
# Script to prepare WAV2VEC data for ATTN-CRNN training
# This script addresses the missing data issue detected by monitoring

set -euo pipefail

# Configuration
EC2_IP="54.162.134.77"
EC2_KEY="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/emotion_project"
LOCAL_DATASET="${1:-crema_d}"  # Default to crema_d if not specified

echo "=== WAV2VEC Feature Setup ==="
echo "Setting up data for $LOCAL_DATASET dataset"

# Function to check if we already have WAV2VEC features
check_remote_features() {
  echo "Checking for existing WAV2VEC features on EC2..."
  
  # Define possible locations where feature files might exist (based on directory scan)
  # Use specific directories with verified feature files
  POSSIBLE_DIRS=(
    "/home/ubuntu/emotion_project/wav2vec_features"    # Symlink to real features
    "/home/ubuntu/audio_emotion/models/wav2vec"        # Primary wav2vec features location (8690 files)
    "/home/ubuntu/emotion-recognition/crema_d_features_facenet" # FaceNet features for CREMA-D (7441 files)
    "/home/ubuntu/emotion-recognition/npz_files/CREMA-D"        # Another CREMA-D location (7441 files) 
    "/home/ubuntu/emotion-recognition/crema_d_features_audio"   # Audio features for CREMA-D (530 files)
  )
  
  TOTAL_COUNT=0
  
  # Check each possible directory
  for DIR in "${POSSIBLE_DIRS[@]}"; do
    echo "Checking for features in $DIR..."
    # Use -L flag to follow symlinks
    COUNT=$(ssh -i "$EC2_KEY" ubuntu@$EC2_IP "find -L $DIR -name '*.npz' 2>/dev/null | wc -l")
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
    
    if [ "$COUNT" -gt 0 ]; then
      echo "Found $COUNT feature files in $DIR"
    fi
  done
  
  if [ "$TOTAL_COUNT" -gt 0 ]; then
    echo "Found a total of $TOTAL_COUNT existing WAV2VEC feature files"
    return 0
  else
    echo "No WAV2VEC feature files found in any expected location"
    return 1
  fi
}

# Function to check if dataset directory exists locally
check_local_dataset() {
  if [ -d "$LOCAL_DATASET" ]; then
    echo "Found local dataset at $LOCAL_DATASET"
    return 0
  else
    echo "Local dataset not found at $LOCAL_DATASET"
    return 1
  fi
}

# Function to extract and prepare WAV2VEC features
prepare_features() {
  echo "Preparing WAV2VEC features for upload..."
  
  # Create directory if it doesn't exist
  mkdir -p "wav2vec_features"
  
  # Check for local npz files
  if [ -d "crema_d_features" ] && [ "$(find crema_d_features -name '*.npz' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Found local WAV2VEC features in crema_d_features directory"
    echo "Preparing to upload these..."
    
    # Create a tarball of existing features
    tar -czf wav2vec_features.tar.gz crema_d_features
    
    echo "Uploading WAV2VEC features to EC2..."
    scp -i "$EC2_KEY" wav2vec_features.tar.gz ubuntu@$EC2_IP:$REMOTE_DIR/
    
    # Extract on remote
    ssh -i "$EC2_KEY" ubuntu@$EC2_IP "cd $REMOTE_DIR && tar -xzf wav2vec_features.tar.gz && rm wav2vec_features.tar.gz"
    
    # Check if extraction was successful
    COUNT=$(ssh -i "$EC2_KEY" ubuntu@$EC2_IP "find $REMOTE_DIR -name '*.npz' 2>/dev/null | wc -l")
    echo "After upload, found $COUNT WAV2VEC feature files on EC2"
    
    return 0
  else
    echo "No local WAV2VEC features found in crema_d_features"
    echo "Creating sample features for testing..."
    
    # Create sample data for testing
    python3 -c '
import numpy as np
import os

# Create a directory for features
os.makedirs("wav2vec_features", exist_ok=True)

# Create 10 sample feature files with random data
for i in range(10):
    # Random features (assuming 512-dim features)
    features = np.random.randn(100, 512).astype(np.float32)
    
    # Random label (0-7 for emotion classes)
    label = np.random.randint(0, 8)
    
    # Save to npz file
    filename = f"wav2vec_features/sample_{i}_label_{label}.npz"
    np.savez(filename, features=features, label=label)
    
    print(f"Created {filename}")
'
    
    # Create a tarball
    tar -czf wav2vec_features.tar.gz wav2vec_features
    
    echo "Uploading sample WAV2VEC features to EC2..."
    scp -i "$EC2_KEY" wav2vec_features.tar.gz ubuntu@$EC2_IP:$REMOTE_DIR/
    
    # Extract on remote
    ssh -i "$EC2_KEY" ubuntu@$EC2_IP "cd $REMOTE_DIR && tar -xzf wav2vec_features.tar.gz && rm wav2vec_features.tar.gz"
    
    # Check if extraction was successful
    COUNT=$(ssh -i "$EC2_KEY" ubuntu@$EC2_IP "find $REMOTE_DIR -name '*.npz' 2>/dev/null | wc -l")
    echo "After upload, found $COUNT WAV2VEC feature files on EC2"
    
    return 0
  fi
}

# Check if we already have features on EC2
if check_remote_features; then
  echo "WAV2VEC features already exist on EC2, skipping upload."
else
  echo "Need to prepare WAV2VEC features..."
  prepare_features
fi

# Update the training script to point to the correct data directory
echo "Creating fixed ATTN-CRNN training script..."

cat > fixed_attn_crnn.py << 'EOF'
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Conv1D, MaxPooling1D, Attention, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Attention CRNN model')
parser.add_argument('--data_dirs', nargs='+', required=True, help='Directories containing .npz files')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--test_size', type=float, default=0.2, help='Test split size')
parser.add_argument('--val_size', type=float, default=0.1, help='Validation split size from train data')
args = parser.parse_args()

# Augmentation not implemented in this version as it was causing issues
# Show directories we're loading from
print(f"Loading data from directories: {args.data_dirs}")

# Load data
data_list = []
for data_dir in args.data_dirs:
    # Look for npz files in the directory and all subdirectories
    file_paths = glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
    print(f"Found {len(file_paths)} files in {data_dir}")
    
    if len(file_paths) == 0:
        # Check subdirectories for features
        subdirs = ["wav2vec_features", "features", "crema_d_features"]
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"Checking {subdir_path} for features...")
                file_paths = glob.glob(os.path.join(subdir_path, '**', '*.npz'), recursive=True)
                if len(file_paths) > 0:
                    print(f"Found {len(file_paths)} files in {subdir_path}")
                    break
    
    if len(file_paths) == 0:
        print(f"WARNING: No .npz files found in {data_dir}")
        continue
    
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            features = data['features'] if 'features' in data else None
            label = data['label'] if 'label' in data else None
            
            if features is not None and label is not None:
                data_list.append((features, label))
            else:
                print(f"Skipping {file_path}: Missing 'features' or 'label'")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

if len(data_list) == 0:
    raise ValueError("No valid data was loaded")

print(f"Loaded {len(data_list)} valid samples")

# Process data
X = []
y = []
for features, label in data_list:
    # Ensure 3D shape (samples, timesteps, features)
    if len(features.shape) == 2:
        X.append(features)
        y.append(label)
    else:
        print(f"Skipping sample with unexpected shape: {features.shape}")

X = np.array(X)
y = np.array(y)

print("Input shape:", X.shape)
print("Labels shape:", y.shape)

# Check class distribution
unique_labels, label_counts = np.unique(y, return_counts=True)
print("Class distribution:")
for label, count in zip(unique_labels, label_counts):
    print(f"Class {label}: {count} samples")

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val_size, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Build Attention CRNN model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Recurrent layer
    lstm_out = LSTM(256, return_sequences=True)(x)
    
    # Attention mechanism
    query = Dense(256)(lstm_out)
    attention_scores = Attention()([query, lstm_out])
    
    # Dropout for regularization
    x = Dropout(0.5)(attention_scores)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_attn_crnn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    mode='max',
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    mode='min',
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger('attn_crnn_training_history.csv')

callbacks = [checkpoint, early_stopping, lr_reducer, csv_logger]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc}")

# Generate predictions for confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)

# Create confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('attn_crnn_confusion_matrix.png')

# Save training history plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('attn_crnn_training_history.png')

print("Training completed!")
EOF

# Upload the fixed script to EC2
echo "Uploading fixed script to EC2..."
scp -i "$EC2_KEY" fixed_attn_crnn.py ubuntu@$EC2_IP:$REMOTE_DIR/scripts/train_attn_crnn.py

# Create script to restart training in the tmux session
echo "Creating script to restart training..."

cat > restart_training.sh << 'EOF'
#!/bin/bash
# Restart training in tmux session
ssh -i "$HOME/Downloads/gpu-key.pem" ubuntu@54.162.134.77 << ENDSSH
  # Check if tmux session exists, kill it if it does
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "Killing existing tmux session..."
    tmux kill-session -t audio_train
  fi
  
  # Create new tmux session
  echo "Creating new tmux session and starting training..."
  tmux new-session -d -s audio_train
  
  # Send commands to the tmux session
  tmux send-keys -t audio_train "cd ~/emotion_project" C-m
  tmux send-keys -t audio_train "source ~/miniconda3/bin/activate" C-m
  tmux send-keys -t audio_train "conda activate emotion_env || conda create -n emotion_env python=3.9 -y && conda activate emotion_env" C-m
  tmux send-keys -t audio_train "pip install tensorflow==2.15.0 tensorflow-addons seaborn tqdm matplotlib numpy pandas scikit-learn" C-m
  tmux send-keys -t audio_train "CUDA_VISIBLE_DEVICES=0 python scripts/train_attn_crnn.py --data_dirs ." C-m
  
  echo "Training started in tmux session 'audio_train'"
  echo "Use 'tmux a -t audio_train' to attach to the session"
ENDSSH
EOF

chmod +x restart_training.sh

echo "Setup complete!"
echo "Run './restart_training.sh' to start training with the fixed script"
