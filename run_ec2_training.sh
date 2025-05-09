#!/bin/bash
set -e

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

# Create a Python script to generate the data and run the Attn-CRNN training
ssh -i "$PEM" ubuntu@$IP "
# Create directories
mkdir -p ~/emotion_project/crema_d_features
mkdir -p ~/emotion_project/models

# Create a Python script to generate sample data
cat > ~/emotion_project/gen_samples.py << 'PYEOF'
import numpy as np
import os
import random

# Emotion classes
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
    
    # Save file
    filename = f'/home/ubuntu/emotion_project/crema_d_features/sample_{i:03d}_{emotion}.npz'
    np.savez(filename, 
            wav2vec_features=features,
            emotion=emotion,
            emotion_class=emotion_idx)
    
    print(f'Created {filename}, emotion: {emotion}, length: {seq_length}')

print(f'Generated 100 sample files')
PYEOF

# Run the data generation script
python3 ~/emotion_project/gen_samples.py

# Start the training in a tmux session
tmux new-session -d -s audio_train 'cd ~/emotion_project && python3 scripts/train_attn_crnn.py --data_dirs crema_d_features'

echo 'Data generation complete and training started in tmux session'
echo 'You can monitor the training using the monitoring script'
"

echo "========================================="
echo "Training launched on EC2 instance. Use the monitoring script to check progress:"
echo "./monitor_attn_crnn_training.sh -s  # to check GPU status"
echo "./monitor_attn_crnn_training.sh -l  # to view training logs"
echo "========================================="
