#!/usr/bin/env bash
# Script to train an emotion recognition model using the extracted wav2vec features

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_TYPE="lstm"  # Options: lstm, transformer

echo "=== Training Wav2Vec-based Emotion Recognition Model ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo "Model Type: $MODEL_TYPE"
echo

# Create a training script
cat > train_wav2vec_emotion.sh << 'EOL'
#!/bin/bash
# Train a model using wav2vec features on EC2

# Parse command line arguments
MODEL_TYPE=${1:-lstm}
BATCH_SIZE=${2:-32}
EPOCHS=${3:-50}
LEARNING_RATE=${4:-0.001}

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Set the directory
cd /home/ubuntu/audio_emotion

# Ensure directories exist
mkdir -p models/audio_emotion

# Check if we have extracted features
WAV2VEC_DIR="models/wav2vec"
if [ ! -d "$WAV2VEC_DIR" ] || [ -z "$(ls -A $WAV2VEC_DIR)" ]; then
    echo "No wav2vec features found in $WAV2VEC_DIR"
    echo "Please run feature extraction first!"
    exit 1
fi

# Count available features
FEATURE_COUNT=$(find $WAV2VEC_DIR -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT feature files for training"

echo "=== Starting training with $MODEL_TYPE model at $(date) ==="
echo "Parameters: batch_size=$BATCH_SIZE, epochs=$EPOCHS, learning_rate=$LEARNING_RATE"

# Run appropriate training script based on model type
if [ "$MODEL_TYPE" = "lstm" ]; then
    echo "Training LSTM model..."
    python scripts/train_wav2vec_lstm.py \
        --features_dir $WAV2VEC_DIR \
        --output_dir models/audio_emotion \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE
elif [ "$MODEL_TYPE" = "transformer" ]; then
    echo "Training Transformer model..."
    python scripts/train_wav2vec_transformer.py \
        --features_dir $WAV2VEC_DIR \
        --output_dir models/audio_emotion \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE
else
    echo "Unsupported model type: $MODEL_TYPE"
    echo "Supported types: lstm, transformer"
    exit 1
fi

echo
echo "=== Training completed at $(date) ==="
echo "Model saved to: models/audio_emotion/"

# Create a tarball of the trained model for easier download
echo "Creating tarball of trained model..."
tar -czvf wav2vec_emotion_model.tar.gz models/audio_emotion/
echo "Tarball created at: /home/ubuntu/audio_emotion/wav2vec_emotion_model.tar.gz"
EOL
chmod +x train_wav2vec_emotion.sh

# Check if scripts directory exists on EC2
echo "Checking if required training scripts exist on EC2..."
SCRIPT_CHECK=$(ssh -i "$KEY_PATH" "$EC2_INSTANCE" "test -f $REMOTE_DIR/scripts/train_wav2vec_lstm.py && echo 'Found' || echo 'Missing'")
if [ "$SCRIPT_CHECK" = "Missing" ]; then
    echo "WARNING: train_wav2vec_lstm.py script was not found on the EC2 instance."
    echo "Creating minimal training scripts..."
    
    # Create minimal LSTM training script
    cat > train_wav2vec_lstm.py << 'EOF'
#!/usr/bin/env python3
"""
Train an LSTM model using the wav2vec features extracted from audio files.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import glob

def load_features(features_dir):
    """Load all wav2vec features from NPZ files."""
    feature_files = glob.glob(os.path.join(features_dir, "*.npz"))
    if not feature_files:
        raise ValueError(f"No feature files found in {features_dir}")
    
    features = []
    labels = []
    
    for file in feature_files:
        data = np.load(file)
        feature = data["wav2vec_features"]
        label = data["label"]
        
        # Max sequence length handling - truncate or pad
        max_len = 500  # You can adjust this based on your data
        if feature.shape[0] > max_len:
            feature = feature[:max_len]
        else:
            pad_len = max_len - feature.shape[0]
            feature = np.pad(feature, ((0, pad_len), (0, 0)), 'constant')
        
        features.append(feature)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def build_lstm_model(input_shape, num_classes=6):
    """Build an LSTM model for emotion classification."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train LSTM model on wav2vec features")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory with wav2vec features")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Configure GPU memory growth for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features_dir}...")
    features, labels = load_features(args.features_dir)
    
    # Verify loaded data
    print(f"Loaded {len(features)} samples with shape {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Build model
    input_shape = (features.shape[1], features.shape[2])
    num_classes = labels.shape[1]
    
    model = build_lstm_model(input_shape, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_wav2vec_lstm_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        verbose=1
    )
    
    # Train model
    print(f"Training model with batch size {args.batch_size} for {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'final_wav2vec_lstm_model.h5'))
    
    # Save training history
    np.save(os.path.join(args.output_dir, 'training_history.npy'), history.history)
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

    # Create minimal Transformer training script
    cat > train_wav2vec_transformer.py << 'EOF'
#!/usr/bin/env python3
"""
Train a Transformer model using the wav2vec features extracted from audio files.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import glob

def load_features(features_dir):
    """Load all wav2vec features from NPZ files."""
    feature_files = glob.glob(os.path.join(features_dir, "*.npz"))
    if not feature_files:
        raise ValueError(f"No feature files found in {features_dir}")
    
    features = []
    labels = []
    
    for file in feature_files:
        data = np.load(file)
        feature = data["wav2vec_features"]
        label = data["label"]
        
        # Max sequence length handling - truncate or pad
        max_len = 500  # You can adjust this based on your data
        if feature.shape[0] > max_len:
            feature = feature[:max_len]
        else:
            pad_len = max_len - feature.shape[0]
            feature = np.pad(feature, ((0, pad_len), (0, 0)), 'constant')
        
        features.append(feature)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer encoder block."""
    # Multi-head attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    res = x + inputs
    
    # Feed-forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=4, mlp_units=[128], dropout=0.3, mlp_dropout=0.3, num_classes=6):
    """Build a Transformer model for emotion classification."""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # MLP layers
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)

def main():
    parser = argparse.ArgumentParser(description="Train Transformer model on wav2vec features")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory with wav2vec features")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Configure GPU memory growth for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features_dir}...")
    features, labels = load_features(args.features_dir)
    
    # Verify loaded data
    print(f"Loaded {len(features)} samples with shape {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Build model
    input_shape = (features.shape[1], features.shape[2])
    num_classes = labels.shape[1]
    
    model = build_transformer_model(input_shape, num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_wav2vec_transformer_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        verbose=1
    )
    
    # Train model
    print(f"Training model with batch size {args.batch_size} for {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'final_wav2vec_transformer_model.h5'))
    
    # Save training history
    np.save(os.path.join(args.output_dir, 'training_history.npy'), history.history)
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
EOF

    # Create scripts directory if not exists
    ssh -i "$KEY_PATH" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR/scripts"
    
    # Upload scripts
    echo "Uploading training scripts to EC2..."
    scp -i "$KEY_PATH" train_wav2vec_lstm.py "$EC2_INSTANCE:$REMOTE_DIR/scripts/"
    scp -i "$KEY_PATH" train_wav2vec_transformer.py "$EC2_INSTANCE:$REMOTE_DIR/scripts/"
    
    # Cleanup local files
    rm train_wav2vec_lstm.py train_wav2vec_transformer.py
fi

# Upload the training script
echo "Uploading training script to EC2..."
scp -i "$KEY_PATH" train_wav2vec_emotion.sh "$EC2_INSTANCE:$REMOTE_DIR/"

echo
echo "Training script uploaded to EC2."
echo
echo "To start training on EC2 (when feature extraction is complete):"
echo "  ssh -i $KEY_PATH $EC2_INSTANCE"
echo "  cd $REMOTE_DIR"
echo "  ./train_wav2vec_emotion.sh $MODEL_TYPE 32 50 0.001"
echo
echo "Parameters:"
echo "  1. Model type: lstm or transformer (default: lstm)"
echo "  2. Batch size (default: 32)"
echo "  3. Epochs (default: 50)"
echo "  4. Learning rate (default: 0.001)"
echo
echo "After training completes, download the model:"
echo "  mkdir -p wav2vec_emotion_model"
echo "  scp -i $KEY_PATH $EC2_INSTANCE:$REMOTE_DIR/wav2vec_emotion_model.tar.gz wav2vec_emotion_model/"
echo "  tar -xzvf wav2vec_emotion_model/wav2vec_emotion_model.tar.gz -C wav2vec_emotion_model/"
