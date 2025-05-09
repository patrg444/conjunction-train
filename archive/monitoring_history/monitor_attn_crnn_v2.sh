#!/bin/bash
# Script to monitor ATTN-CRNN v2 training progress on EC2

# Default flags
CONTINUOUS=false
LOGS_ONLY=false

# Parse command line arguments
while getopts "cl" opt; do
  case ${opt} in
    c )
      CONTINUOUS=true
      ;;
    l )
      LOGS_ONLY=true
      ;;
    \? )
      echo "Usage: $0 [-c] [-l]"
      echo "  -c  Continuous monitoring (stream output)"
      echo "  -l  Show only the recent logs without checking status"
      exit 1
      ;;
  esac
done

# Get EC2 IP
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # Adjust if your key path is different

# Function to check training status
check_status() {
  echo "=== ATTN-CRNN V2 TRAINING HEALTH CHECK ==="
  echo "Checking training status..."
  
  ssh -i $SSH_KEY ubuntu@$EC2_IP << EOF
    cd ~/emotion_recognition
    
    # Check if tmux session exists
    if tmux has-session -t attn_crnn_v2 2>/dev/null; then
      echo "✓ tmux session 'attn_crnn_v2' is active"
    else
      echo "✗ WARNING: tmux session 'attn_crnn_v2' not found"
    fi
    
    # Check if the training process is running
    if pgrep -f "python scripts/train_attn_crnn_v2.py" > /dev/null; then
      echo "✓ Training process is running"
    else
      echo "✗ WARNING: training process not running"
    fi
    
    # Check for WAV2VEC feature files
    feature_count=\$(find /data/wav2vec_features /data/wav2vec_crema_d -type f -name "*.npz" 2>/dev/null | wc -l)
    echo "WAV2VEC feature files found: \$feature_count"
    
    # Check for model checkpoint
    if [ -f "checkpoints/attn_crnn_v2/best_attn_crnn_model.keras" ]; then
      echo "✓ Model checkpoint found"
    else
      echo "Model checkpoint not found yet"
    fi
EOF
}

# Function to display recent logs
show_logs() {
  echo "RECENT TRAINING LOGS ==="
  
  ssh -i $SSH_KEY ubuntu@$EC2_IP << EOF
    cd ~/emotion_recognition
    
    # Show the last 50 lines of the log
    if [ -f "training_attn_crnn_v2.log" ]; then
      tail -n 50 training_attn_crnn_v2.log
    else
      echo "No training log file found"
    fi
EOF
}

# Function to continuously monitor the logs
stream_logs() {
  echo "=== STARTING CONTINUOUS STREAM ==="
  echo "Press Ctrl+C to exit"
  echo
  echo "Streaming filtered training output (epoch updates)..."
  
  # Use SSH to tail the log file and filter for important lines about epochs, validation, etc.
  ssh -i $SSH_KEY ubuntu@$EC2_IP "cd ~/emotion_recognition && tail -f training_attn_crnn_v2.log | grep -E 'Epoch|val_|accuracy|loss|MODEL|Training complete'"
}

# Main execution
if [ "$LOGS_ONLY" = true ]; then
  show_logs
else
  check_status
  echo
  show_logs
fi

# If continuous mode is enabled, stream the logs
if [ "$CONTINUOUS" = true ]; then
  echo
  stream_logs
fi
