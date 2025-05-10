#!/bin/bash
#
# Fixed script for starting ATTN-CRNN training with correct WAV2VEC directory
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== STARTING FIXED ATTN-CRNN TRAINING ==="
echo "Ensuring model looks in the correct directory for WAV2VEC features..."

# Connect to remote server and fix the training
ssh -i "$PEM" ubuntu@$IP << 'EOF'
  # Make sure the symlink exists
  if [ ! -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    echo "Creating symbolic link to WAV2VEC features..."
    ln -sf /home/ubuntu/audio_emotion/models/wav2vec /home/ubuntu/emotion_project/wav2vec_features
  fi

  # Check if any tmux sessions are running and kill them
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "Stopping existing training session..."
    tmux kill-session -t audio_train
  fi

  # Create training script that uses the correct path
  echo "Setting up training script with correct paths..."
  cat > /home/ubuntu/start_attn_crnn.sh << 'INNERSCRIPT'
#!/bin/bash
source /opt/pytorch/bin/activate

cd /home/ubuntu/emotion_project

# Configure Python path
export PYTHONPATH=/home/ubuntu/emotion_project:$PYTHONPATH

# Start training with the correct paths
python /home/ubuntu/emotion_project/scripts/train_attn_crnn.py \
  --data_dirs=/home/ubuntu/emotion_project/wav2vec_features \
  --epochs=50 \
  --batch_size=32 \
  --lr=0.001 \
  --model_save_path=/home/ubuntu/emotion_project/best_attn_crnn_model.h5
INNERSCRIPT

  chmod +x /home/ubuntu/start_attn_crnn.sh

  # Start training in a tmux session
  echo "Starting training in tmux session..."
  tmux new-session -d -s audio_train '/home/ubuntu/start_attn_crnn.sh'

  echo "Training started in tmux session 'audio_train'"
  echo "Checking tmux session status..."
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "✓ tmux session 'audio_train' is running"
  else
    echo "✗ Failed to start tmux session"
  fi
EOF

echo "=== ATTN-CRNN TRAINING SETUP COMPLETE ==="
echo "You can monitor training progress with: ./continuous_attn_crnn_monitor.sh"
