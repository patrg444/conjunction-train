#!/bin/bash
set -e

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== Launching ATTN-CRNN Wav2Vec Training ==="
echo "1. Connecting to EC2 instance..."
echo "2. Creating synthetic wav2vec data..."
echo "3. Starting training in tmux session..."

ssh -i "$PEM" ubuntu@$IP "
# Create directories if they don't exist
mkdir -p ~/emotion_project/crema_d_features
mkdir -p ~/emotion_project/logs
mkdir -p ~/emotion_project/models

# Generate synthetic data
echo 'Generating synthetic wav2vec features...'
python3 -c \"
import numpy as np
import os
import random

# Emotion classes and their indices
emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

# Generate 100 samples
for i in range(100):
    # Random emotion index
    emotion_idx = random.randint(0, len(emotions)-1)
    emotion = emotions[emotion_idx]
    
    # Create feature sequence of random length (50-150 frames)
    seq_length = random.randint(50, 150)
    
    # Wav2vec features are 768-dimensional
    features = np.random.normal(0, 1, (seq_length, 768)).astype(np.float32)
    
    # Save file with standard Wav2Vec format
    filename = f'/home/ubuntu/emotion_project/crema_d_features/sample_{i:03d}_{emotion}.npz'
    np.savez(filename, 
             wav2vec_features=features,
             emotion_class=emotion_idx)
    
    if i % 20 == 0:
        print(f'Created {i+1} samples...')

print('Generated 100 sample files in ~/emotion_project/crema_d_features/')
\"

# Make script executable
chmod +x ~/emotion_project/scripts/train_attn_crnn.py

# Check if tmux session exists and kill it if it does
tmux ls 2>/dev/null | grep -q audio_train && tmux kill-session -t audio_train || true

# Launch training in tmux
echo 'Starting training in tmux session...'
cd ~/emotion_project && tmux new-session -d -s audio_train 'python3 scripts/train_attn_crnn.py --data_dirs crema_d_features --epochs 20 --batch_size 16'

echo 'Training started successfully in tmux session.'
echo 'You can monitor it using the monitoring script.'
"

echo ""
echo "=== Training Launched Successfully ==="
echo "To monitor training progress:"
echo "  ./monitor_attn_crnn_training.sh      # Show both GPU and logs"
echo "  ./monitor_attn_crnn_training.sh -s   # Show GPU status only"
echo "  ./monitor_attn_crnn_training.sh -l   # Show logs only" 
echo ""
