#!/bin/bash
# Video-Only Emotion Recognition with Facenet Features (Key-Fixed Version)
LOG_FILE="video_only_facenet_lstm_key_fixed.log"
SSH_KEY="~/Downloads/gpu-key.pem"
REMOTE_HOST="ubuntu@18.208.166.91"
CHECKPOINT_PATTERN="/home/ubuntu/emotion-recognition/models/video_only_facenet_lstm_*"

echo "=== Monitoring Video-Only Facenet + LSTM Training (Fixed Version) ==="
echo "Log file: /home/ubuntu/emotion-recognition/$LOG_FILE"
echo "Checkpoint pattern: $CHECKPOINT_PATTERN"
echo "Connecting to 18.208.166.91..."

function monitor_checkpoints() {
  echo
  echo "--- Current Checkpoints ---"
  ssh -i $SSH_KEY $REMOTE_HOST "ls -dt $CHECKPOINT_PATTERN 2>/dev/null | head -1 || echo 'No checkpoint directory found yet.'"
  echo
}

function monitor_log() {
  echo "--- Recent Training Log ---"
  ssh -i $SSH_KEY $REMOTE_HOST "tail -30 /home/ubuntu/emotion-recognition/$LOG_FILE"
  echo
}

# Main monitoring loop
while true; do
  clear
  date
  monitor_checkpoints
  monitor_log
  
  # Check if best model exists
  LATEST_DIR=$(ssh -i $SSH_KEY $REMOTE_HOST "ls -dt $CHECKPOINT_PATTERN 2>/dev/null | head -1 || echo ''")
  if [ ! -z "$LATEST_DIR" ]; then
    echo "--- Best Model Check ---"
    ssh -i $SSH_KEY $REMOTE_HOST "ls -lt $LATEST_DIR | grep 'best_model_video_only_facenet_lstm.keras' | head -n 1" 2>/dev/null || echo "No best model file found yet."
  else
    echo "No checkpoint directory found yet."
  fi
  
  echo
  echo "Press Ctrl+C to exit monitoring..."
  sleep 10
done
