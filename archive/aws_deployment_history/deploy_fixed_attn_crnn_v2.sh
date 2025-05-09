#!/bin/bash
#
# Deploy the TensorFlow-compatible memory-efficient version of the ATTN-CRNN training script
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== DEPLOYING TENSORFLOW-COMPATIBLE ATTN-CRNN TRAINING ==="

# First, upload the fixed script to the server
echo "Uploading fixed script to server..."
scp -i "$PEM" train_attn_crnn_fixed_v2.py ubuntu@$IP:/home/ubuntu/emotion_project/scripts/train_attn_crnn_fixed_v2.py

# Now connect to the server and set up the training
ssh -i "$PEM" ubuntu@$IP << 'EOF'
  set -euo pipefail

  # Ensure directories exist
  mkdir -p /home/ubuntu/emotion_project/logs
  mkdir -p /home/ubuntu/emotion_project/models
  
  # Ensure symlink to wav2vec features exists
  if [ ! -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    ln -sf "/home/ubuntu/audio_emotion/models/wav2vec" /home/ubuntu/emotion_project/wav2vec_features
  fi
  
  # Make script executable
  chmod +x /home/ubuntu/emotion_project/scripts/train_attn_crnn_fixed_v2.py
  
  # Kill any existing audio_train session if it exists
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "Stopping existing audio_train session..."
    tmux kill-session -t audio_train
  fi
  
  # Create a script to launch the training
  cat > /home/ubuntu/run_fixed_attn_crnn_v2.sh << 'SHELL'
#!/bin/bash
cd /home/ubuntu
source /opt/pytorch/bin/activate

# Set up environment
export PYTHONPATH=/home/ubuntu/emotion_project:${PYTHONPATH:-}
LOG_FILE="/home/ubuntu/emotion_project/logs/attn_crnn_training_v2_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting TensorFlow-compatible ATTN-CRNN training at $(date) ===" | tee -a $LOG_FILE
echo "Using memory-efficient data generator approach with fixed TensorFlow code" | tee -a $LOG_FILE

# Verify data access
echo "Checking WAV2VEC feature files..." | tee -a $LOG_FILE
FEATURE_COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT feature files" | tee -a $LOG_FILE

# Run with full error capture
python -u /home/ubuntu/emotion_project/scripts/train_attn_crnn_fixed_v2.py \
  --data_dirs /home/ubuntu/emotion_project/wav2vec_features \
  --epochs 50 \
  --batch_size 16 \
  --patience 10 \
  --max_seq_length 1000 2>&1 | tee -a $LOG_FILE
SHELL

  chmod +x /home/ubuntu/run_fixed_attn_crnn_v2.sh
  
  # Start a new tmux session
  echo "Starting new tmux session 'audio_train' with TensorFlow-compatible training..."
  tmux new-session -d -s audio_train '/home/ubuntu/run_fixed_attn_crnn_v2.sh; exec bash'
  
  # Check if the session is running
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "✓ tmux session 'audio_train' is running"
    TMUX_PID=$(tmux list-panes -t audio_train -F "#{pane_pid}" 2>/dev/null || echo "")
    if [ -n "$TMUX_PID" ]; then
      echo "✓ tmux session has active process (PID: $TMUX_PID)"
    fi
    echo "✓ TensorFlow-compatible training started successfully"
  else
    echo "✗ Failed to start tmux session"
    exit 1
  fi
EOF

echo "=== DEPLOYMENT COMPLETE ==="
echo "Use these commands to monitor training:"
echo "General monitoring:    ssh -i $PEM ubuntu@$IP \"tail -n 50 /home/ubuntu/emotion_project/logs/attn_crnn_training_v2_*.log\""
echo "Live session view:     ssh -i $PEM ubuntu@$IP \"tmux attach -t audio_train\""
echo "Note: To detach from tmux without killing the session, press Ctrl+b, then d"
