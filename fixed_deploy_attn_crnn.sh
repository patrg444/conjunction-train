#!/bin/bash
#
# Fixed deployment script that uses only the supported parameters
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== DEPLOYING FIXED ATTN-CRNN TRAINING WITH CORRECT PARAMETERS ==="

ssh -i "$PEM" ubuntu@$IP << 'EOF'
  set -euo pipefail

  # Ensure we have the symlink set up correctly
  mkdir -p /home/ubuntu/emotion_project/scripts
  
  # Ensure symlink to wav2vec features exists
  if [ ! -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    ln -sf "/home/ubuntu/audio_emotion/models/wav2vec" /home/ubuntu/emotion_project/wav2vec_features
  fi
  
  # Create working directories
  mkdir -p /home/ubuntu/emotion_project/models
  mkdir -p /home/ubuntu/emotion_project/logs
  
  # Check what parameters the script actually supports
  echo "Checking accepted parameters for train_attn_crnn.py..."
  python /home/ubuntu/emotion_project/scripts/train_attn_crnn.py --help || true
  
  # Create a new tmux session with fixed parameters
  echo "Creating tmux session with correct parameters..."
  cat > /home/ubuntu/run_fixed_attn_crnn.sh << 'SHELL'
#!/bin/bash
cd /home/ubuntu
source /opt/pytorch/bin/activate

# Set up environment
export PYTHONPATH=/home/ubuntu/emotion_project:${PYTHONPATH:-}
LOG_FILE="/home/ubuntu/emotion_project/logs/attn_crnn_training_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting ATTN-CRNN training at $(date) ===" | tee -a $LOG_FILE
echo "Using supported parameters only..." | tee -a $LOG_FILE

# Verify data access
echo "Checking feature files..." | tee -a $LOG_FILE
FEATURE_COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT feature files" | tee -a $LOG_FILE

# Run with full error capture and only using supported parameters
python -u /home/ubuntu/emotion_project/scripts/train_attn_crnn.py \
  --data_dirs /home/ubuntu/emotion_project/wav2vec_features \
  --epochs 50 \
  --batch_size 32 \
  --patience 10 2>&1 | tee -a $LOG_FILE
SHELL

  chmod +x /home/ubuntu/run_fixed_attn_crnn.sh
  
  # Kill any existing audio_train session if it exists
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "Stopping existing audio_train session..."
    tmux kill-session -t audio_train
  fi
  
  # Start a new tmux session
  echo "Starting new tmux session 'audio_train'..."
  tmux new-session -d -s audio_train '/home/ubuntu/run_fixed_attn_crnn.sh; exec bash'
  
  # Check if the session is running
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "✓ tmux session 'audio_train' is running"
    TMUX_PID=$(tmux list-panes -t audio_train -F "#{pane_pid}" 2>/dev/null || echo "")
    if [ -n "$TMUX_PID" ]; then
      echo "✓ tmux session has active process (PID: $TMUX_PID)"
    fi
    echo "✓ Training started successfully"
  else
    echo "✗ Failed to start tmux session"
    exit 1
  fi
EOF

echo "=== FIXED DEPLOYMENT COMPLETE ==="
echo "Use this command to monitor training logs:"
echo "ssh -i $PEM ubuntu@$IP \"tail -n 30 /home/ubuntu/emotion_project/logs/attn_crnn_training_*.log\""
