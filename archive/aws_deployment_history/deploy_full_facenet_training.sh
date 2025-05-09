#!/bin/bash
# Deploy and run full-scale Facenet video emotion recognition training on AWS GPU

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/Users/patrickgloria/Downloads/gpu-key.pem}"
INSTANCE_IP="${1:-18.208.166.91}"
REMOTE_DIR="/home/ubuntu/emotion-recognition"
MODEL_NAME="facenet_lstm_fixed_gpu"
BATCH_SIZE=32
EPOCHS=100
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-ip>"
  echo "Or set INSTANCE_IP environment variable"
  exit 1
fi

echo "=== Deploying full Facenet training to $INSTANCE_IP ==="
echo "SSH key: $SSH_KEY"
echo "Remote directory: $REMOTE_DIR"
echo "Model name: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"

# Check if SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY"
  exit 1
fi

# Create temp directory to organize files
echo "Creating temporary directory for deployment..."
mkdir -p facenet_deployment

# Copy key fixed files
cp scripts/fixed_video_facenet_generator.py facenet_deployment/
cp scripts/train_video_only_facenet_lstm_key_fixed.py facenet_deployment/
cp scripts/test_train_facenet_lstm.py facenet_deployment/

# Create full training script
cat > facenet_deployment/train_facenet_full.py << 'EOF'
#!/usr/bin/env python3
"""
Full-scale training script for Facenet video-only emotion recognition
"""
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from fixed_video_facenet_generator import FixedVideoFacenetGenerator

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facenet_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("facenet_training")

def create_model(max_seq_len, feature_dim=512, num_classes=6):
    """Create LSTM model for video-only emotion recognition."""
    model = Sequential([
        LSTM(128, input_shape=(max_seq_len, feature_dim), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def find_feature_files(data_dir):
    """Find all valid feature files in the specified directory."""
    logger.info(f"Searching for feature files in {data_dir}")
    feature_files = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
    logger.info(f"Found {len(feature_files)} feature files")
    return feature_files

def get_emotion_labels(feature_files):
    """Extract emotion labels from filenames."""
    # Map emotion codes to indices
    emotions = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    
    labels = []
    valid_files = []
    
    for filename in feature_files:
        base = os.path.basename(filename)
        for emotion, idx in emotions.items():
            if emotion in base:
                valid_files.append(filename)
                labels.append(idx)
                break
    
    logger.info(f"Found {len(valid_files)} files with valid emotion labels")
    return valid_files, np.array(labels)

def train_model(data_dir="crema_d_features_facenet", batch_size=32, epochs=100):
    """Run full training on feature files."""
    # Find all feature files
    feature_files = find_feature_files(data_dir)
    
    # Get emotion labels
    valid_files, labels = get_emotion_labels(feature_files)
    
    if len(valid_files) == 0:
        logger.error("No valid files with emotion labels found!")
        sys.exit(1)
    
    # Split into train/val
    np.random.seed(42)
    indices = np.arange(len(valid_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [valid_files[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    val_files = [valid_files[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    logger.info(f"Train samples: {len(train_files)}")
    logger.info(f"Validation samples: {len(val_files)}")
    
    # Calculate max sequence length from training set
    max_seq_len = 0
    sample_size = min(100, len(train_files))
    sample_indices = np.random.choice(len(train_files), sample_size, replace=False)
    
    logger.info("Calculating maximum sequence length...")
    for idx in sample_indices:
        try:
            file_path = train_files[idx]
            with np.load(file_path, allow_pickle=True) as data:
                if 'video_features' in data:
                    features = data['video_features']
                    max_seq_len = max(max_seq_len, features.shape[0])
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    logger.info(f"Maximum sequence length: {max_seq_len}")
    
    # Initialize data generators
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        normalize_features=True,
        max_seq_len=max_seq_len
    )
    
    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        normalize_features=True,
        max_seq_len=max_seq_len
    )
    
    # Setup model
    model = create_model(max_seq_len=max_seq_len)
    model.summary()
    
    # Create model save directory
    model_dir = f"models/facenet_lstm_{batch_size}_{epochs}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"logs/facenet_lstm_{batch_size}_{epochs}",
            histogram_freq=1
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, "final_model.h5"))
    logger.info(f"Training complete. Final model saved to {os.path.join(model_dir, 'final_model.h5')}")
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Facenet video-only emotion recognition model")
    parser.add_argument("--data-dir", type=str, default="crema_d_features_facenet", 
                        help="Directory containing feature files")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
EOF

# Create run script
cat > facenet_deployment/run_full_training.sh << EOF
#!/bin/bash
# Script to run full-scale Facenet training

set -e

cd "\$(dirname "\$0")"

# Set up Python environment if needed
if [ ! -d "~/facenet-venv" ]; then
  echo "Setting up Python environment..."
  python3 -m venv ~/facenet-venv
fi

source ~/facenet-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install tensorflow numpy matplotlib tqdm tensorboard pandas h5py

# Create directories
mkdir -p models/
mkdir -p logs/

# Run training
echo "Starting full-scale training..."
python train_facenet_full.py --batch-size ${BATCH_SIZE} --epochs ${EPOCHS}

echo "Training complete!"
EOF

# Create monitoring script
cat > facenet_deployment/monitor_training.sh << 'EOF'
#!/bin/bash
# Script to monitor training progress

cd "$(dirname "$0")"

echo "=== Facenet Training Status ==="

# Check GPU utilization
echo -e "\nGPU Utilization:"
nvidia-smi

# Check training process
echo -e "\nTraining Process:"
ps aux | grep train_facenet_full.py | grep -v grep

# Check latest log entries
echo -e "\nLatest Training Logs:"
tail -n 20 facenet_training.log

# Check validation accuracy
echo -e "\nLatest Validation Metrics:"
grep "val_accuracy" facenet_training.log | tail -n 5

echo -e "\nMonitoring Complete"
EOF

# Make scripts executable
chmod +x facenet_deployment/run_full_training.sh
chmod +x facenet_deployment/monitor_training.sh

# Create tarball
echo "Creating tarball..."
tar -czf facenet_deployment.tar.gz -C facenet_deployment .

# Create remote directory if it doesn't exist
echo "Setting up remote directory..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "mkdir -p $REMOTE_DIR/facenet_full_training"

# Upload tarball
echo "Uploading files..."
scp -i "$SSH_KEY" facenet_deployment.tar.gz ubuntu@"$INSTANCE_IP":~/facenet_deployment.tar.gz

# Extract on remote and start training in tmux
echo "Extracting files and starting training..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REMOTE_DIR/facenet_full_training && tar -xzf ~/facenet_deployment.tar.gz && rm ~/facenet_deployment.tar.gz && tmux new-session -d -s facenet_training './run_full_training.sh > training_output.log 2>&1'"

# Clean up local files
echo "Cleaning up local files..."
rm -rf facenet_deployment.tar.gz facenet_deployment

echo "=== Full training started in tmux session 'facenet_training' ==="
echo ""
echo "To monitor training:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  cd $REMOTE_DIR/facenet_full_training && ./monitor_training.sh"
echo ""
echo "To view live training progress:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  tmux attach -t facenet_training"
echo ""
echo "To detach from tmux, press Ctrl+B then D"
echo ""
echo "To set up TensorBoard (opens on port 6006):"
echo "  ssh -i $SSH_KEY -L 6006:localhost:6006 ubuntu@$INSTANCE_IP"
echo "  source ~/facenet-venv/bin/activate && cd $REMOTE_DIR/facenet_full_training && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"
