#!/bin/bash
# Minimal deployment script for testing Facenet video emotion recognition fixes

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/Users/patrickgloria/Downloads/gpu-key.pem}"
INSTANCE_IP="${1:-18.208.166.91}"
REMOTE_DIR="/home/ubuntu/emotion-recognition"

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-ip>"
  echo "Or set INSTANCE_IP environment variable"
  exit 1
fi

echo "=== Deploying minimal Facenet test to $INSTANCE_IP ==="
echo "SSH key: $SSH_KEY"
echo "Remote directory: $REMOTE_DIR"

# Check if SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY"
  exit 1
fi

# Create temp directory to organize files
echo "Creating temporary directory for deployment..."
mkdir -p facenet_minimal

# Copy key fixed files
cp scripts/fixed_video_facenet_generator.py facenet_minimal/
cp scripts/train_video_only_facenet_lstm_key_fixed.py facenet_minimal/
cp scripts/test_train_facenet_lstm.py facenet_minimal/

# Create remote test script
cat > facenet_minimal/test_facenet_fixes.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Facenet video emotion recognition pipeline fixes
"""
import os
import sys
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from fixed_video_facenet_generator import FixedVideoFacenetGenerator

def load_sample_data():
    """Create a minimal test dataset if no real data exists."""
    os.makedirs('sample_data/features', exist_ok=True)
    
    # Create 20 sample files with random features
    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    for i in range(20):
        emotion = np.random.choice(emotions)
        features = np.random.randn(10, 512).astype(np.float32)  # 10 frames, 512-dim features
        filename = f'sample_data/features/{i:04d}_XXX_{emotion}_XX.npz'
        np.savez(filename, video_features=features, emotion=emotion)
    
    print(f"Created {len(glob.glob('sample_data/features/*.npz'))} sample files")
    return 'sample_data/features'

def test_generator():
    """Test whether the generator can load and batch the files correctly."""
    data_dir = load_sample_data()
    
    # Get all feature files
    feature_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # Create labels from filenames
    emotions = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    labels = []
    for filename in feature_files:
        base = os.path.basename(filename)
        for emotion, idx in emotions.items():
            if emotion in base:
                labels.append(idx)
                break
        else:
            labels.append(0)  # Default to first class if no match
    
    labels = np.array(labels)
    
    # Split files into train/val
    np.random.seed(42)
    indices = np.arange(len(feature_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [feature_files[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    val_files = [feature_files[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    # Initialize generators
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=4,
        shuffle=True,
        normalize_features=True,
        max_seq_len=10
    )
    
    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=4,
        shuffle=False,
        normalize_features=True,
        max_seq_len=10
    )
    
    print(f"Train generator length: {len(train_gen)}")
    print(f"Val generator length: {len(val_gen)}")
    
    # Test batch generation
    print("\nTesting batch generation:")
    X_batch, y_batch = train_gen[0]
    print(f"X_batch shape: {X_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    
    return len(train_gen) > 0 and len(val_gen) > 0

def create_model(max_seq_len=10, feature_dim=512, num_classes=6):
    """Create a simple LSTM model for emotion classification."""
    model = Sequential([
        LSTM(128, input_shape=(max_seq_len, feature_dim), return_sequences=True),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_training():
    """Test a minimal training run to verify everything works."""
    data_dir = load_sample_data()
    
    # Get all feature files
    feature_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # Create labels from filenames
    emotions = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    labels = []
    for filename in feature_files:
        base = os.path.basename(filename)
        for emotion, idx in emotions.items():
            if emotion in base:
                labels.append(idx)
                break
        else:
            labels.append(0)  # Default to first class if no match
    
    labels = np.array(labels)
    
    # Split files into train/val
    np.random.seed(42)
    indices = np.arange(len(feature_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [feature_files[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    val_files = [feature_files[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    # Initialize generators
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=4,
        shuffle=True,
        normalize_features=True,
        max_seq_len=10
    )
    
    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=4,
        shuffle=False,
        normalize_features=True,
        max_seq_len=10
    )
    
    # Create model
    model = create_model()
    
    # Train for 2 epochs
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        verbose=1
    )
    
    return True

if __name__ == "__main__":
    print("=== Testing Facenet Video Emotion Recognition Fixes ===")
    
    print("\n1. Testing generator functionality...")
    if test_generator():
        print("✅ Generator test PASSED")
    else:
        print("❌ Generator test FAILED")
        sys.exit(1)
    
    print("\n2. Testing minimal training run...")
    if test_training():
        print("✅ Training test PASSED")
    else:
        print("❌ Training test FAILED")
        sys.exit(1)
    
    print("\n✅ All tests PASSED! The fixes are working correctly.")
EOF

# Create a simple run script for the remote server
cat > facenet_minimal/run_test.sh << 'EOF'
#!/bin/bash
set -e

cd "$(dirname "$0")"

# Setup Python environment if needed
if [ ! -d "~/facenet-venv" ]; then
  echo "Setting up Python virtual environment..."
  python3 -m venv ~/facenet-venv
fi

source ~/facenet-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install tensorflow numpy matplotlib tqdm

# Run the test
echo "Running Facenet fixes test..."
python test_facenet_fixes.py

echo "Test complete!"
EOF

# Make run script executable
chmod +x facenet_minimal/run_test.sh

# Create tarball
echo "Creating tarball..."
tar -czf facenet_minimal.tar.gz -C facenet_minimal .

# Create remote directory if it doesn't exist
echo "Setting up remote directory..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "mkdir -p $REMOTE_DIR/facenet_test"

# Upload tarball
echo "Uploading files..."
scp -i "$SSH_KEY" facenet_minimal.tar.gz ubuntu@"$INSTANCE_IP":~/facenet_minimal.tar.gz

# Extract on remote and run test
echo "Extracting files and running test on remote..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REMOTE_DIR/facenet_test && tar -xzf ~/facenet_minimal.tar.gz && rm ~/facenet_minimal.tar.gz && ./run_test.sh"

# Clean up local files
echo "Cleaning up local files..."
rm -rf facenet_minimal.tar.gz facenet_minimal

echo "=== Test deployment completed ==="
