#!/bin/bash
#
# Robust script to deploy ATTN-CRNN training with fullest debugging and error handling
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== DEPLOYING ROBUST ATTN-CRNN TRAINING WITH ERROR CAPTURE ==="

# Clean up the deployment
ssh -i "$PEM" ubuntu@$IP << 'EOF'
  set -euo pipefail
  
  echo "Step 1: Checking system status..."
  # More explicitly check the feature directory 
  FEATURE_DIR="/home/ubuntu/audio_emotion/models/wav2vec"
  FEATURE_COUNT=$(find "$FEATURE_DIR" -name "*.npz" | wc -l)
  echo "Found $FEATURE_COUNT WAV2VEC features in main source directory"
  
  # Kill any existing sessions if they exist
  echo "Step 2: Cleaning up existing sessions..."
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "Stopping existing audio_train session..."
    tmux kill-session -t audio_train
  fi
  
  # Create a clean directory structure
  echo "Step 3: Setting up working environment..."
  mkdir -p /home/ubuntu/emotion_project
  
  # Create reliable symlink 
  if [ -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    rm -f /home/ubuntu/emotion_project/wav2vec_features
  fi
  ln -sf "$FEATURE_DIR" /home/ubuntu/emotion_project/wav2vec_features
  
  # Verify symlink
  if [ -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    echo "✓ Symlink created: $(readlink /home/ubuntu/emotion_project/wav2vec_features)"
    echo "✓ Feature count: $(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)"
  else
    echo "✗ Failed to create symlink"
    exit 1
  fi
  
  # Create a more robust training script with error logging
  echo "Step 4: Creating training script with error logging..."
  cat > /home/ubuntu/run_attn_crnn_training.sh << 'INNERSCRIPT'
#!/bin/bash
set -eo pipefail

# Activate PyTorch environment
source /opt/pytorch/bin/activate

# Create output log directory
mkdir -p /home/ubuntu/emotion_project/logs
LOG_FILE="/home/ubuntu/emotion_project/logs/attn_crnn_training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training at $(date)" > $LOG_FILE

# Set up environment
cd /home/ubuntu/emotion_project
export PYTHONPATH=/home/ubuntu/emotion_project:$PYTHONPATH

# Verify WAV2VEC features
echo "Verifying WAV2VEC features access..." | tee -a $LOG_FILE
FEATURE_COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT WAV2VEC feature files" | tee -a $LOG_FILE

# Get list of first 5 feature files to confirm content
echo "Sample files:" | tee -a $LOG_FILE
find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | head -n 5 | tee -a $LOG_FILE

if [ $FEATURE_COUNT -eq 0 ]; then
  echo "ERROR: No WAV2VEC feature files found. Cannot proceed." | tee -a $LOG_FILE
  exit 1
else
  echo "✓ Features verified successfully!" | tee -a $LOG_FILE
fi

# Check that script file is accessible
if [ ! -f "/home/ubuntu/emotion_project/scripts/train_attn_crnn.py" ]; then
  mkdir -p /home/ubuntu/emotion_project/scripts
  echo "Script not found - copying from main script location" | tee -a $LOG_FILE
  cp /home/ubuntu/scripts/train_attn_crnn.py /home/ubuntu/emotion_project/scripts/ 2>/dev/null || \
  cp /home/ubuntu/*/train_attn_crnn.py /home/ubuntu/emotion_project/scripts/ 2>/dev/null || \
  find /home/ubuntu -name "train_attn_crnn.py" -exec cp {} /home/ubuntu/emotion_project/scripts/ \; 2>/dev/null || \
  echo "ERROR: Could not find train_attn_crnn.py script!" | tee -a $LOG_FILE
fi

# Start training with full output capture
echo "Starting ATTN-CRNN training with WAV2VEC features..." | tee -a $LOG_FILE
echo "Command: python /home/ubuntu/emotion_project/scripts/train_attn_crnn.py --data_dirs=/home/ubuntu/emotion_project/wav2vec_features --epochs=50 --batch_size=32 --lr=0.001 --model_save_path=/home/ubuntu/emotion_project/best_attn_crnn_model.h5" | tee -a $LOG_FILE

# Run with error output captured
python /home/ubuntu/emotion_project/scripts/train_attn_crnn.py \
  --data_dirs=/home/ubuntu/emotion_project/wav2vec_features \
  --epochs=50 \
  --batch_size=32 \
  --lr=0.001 \
  --model_save_path=/home/ubuntu/emotion_project/best_attn_crnn_model.h5 2>&1 | tee -a $LOG_FILE
INNERSCRIPT

  chmod +x /home/ubuntu/run_attn_crnn_training.sh
  
  # Start a fresh tmux session for training
  echo "Step 5: Starting new tmux session for training..."
  tmux new-session -d -s audio_train '/home/ubuntu/run_attn_crnn_training.sh; exec bash'
  sleep 3
  
  # Check for the session
  if tmux has-session -t audio_train 2>/dev/null; then
    echo "✓ tmux session 'audio_train' is running"
    echo "✓ You can attach to it with: tmux attach -t audio_train"
  else
    echo "✗ Failed to start tmux session 'audio_train'"
    exit 1
  fi
  
  # Verify the tmux session is actually running something
  echo "Step 6: Checking for running process in tmux session..."
  TMUX_PROC=$(tmux list-panes -t audio_train -F "#{pane_pid}" 2>/dev/null || echo "")
  if [ -n "$TMUX_PROC" ]; then
    echo "✓ tmux session has active process (PID: $TMUX_PROC)"
    ps -p "$TMUX_PROC" -o pid,cmd || echo "Process not found"
  else
    echo "✗ No process found in tmux session"
  fi
  
  echo "✓ ATTN-CRNN training deployment complete"
EOF

echo "=== DEPLOYMENT COMPLETE ==="
echo "Use the following command to check training status:"
echo "  ssh -i $PEM ubuntu@$IP \"tail -n 30 /home/ubuntu/emotion_project/logs/attn_crnn_training_*.log 2>/dev/null || echo 'No log file found yet'\""
