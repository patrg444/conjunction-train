#!/bin/bash

# Monitor the training process for the AUDIO-ONLY precomputed CNN + LSTM model (V2)

LOG_FILE="precomputed_cnn_lstm_audio_only_v2.log" # Updated log file name for V2
CHECKPOINT_DIR_PATTERN="models/precomputed_cnn_lstm_audio_only_v2_*" # Updated checkpoint pattern for V2
KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"
REMOTE_USER="ubuntu"
REMOTE_HOST="18.208.166.91"
REMOTE_DIR="/home/ubuntu/emotion-recognition"

echo "=== Monitoring Precomputed CNN + LSTM Training (Audio-Only V2) ===" # Updated title
echo "Log file: $REMOTE_DIR/$LOG_FILE"
echo "Checkpoint pattern: $REMOTE_DIR/$CHECKPOINT_DIR_PATTERN"
echo "Connecting to $REMOTE_HOST..."
echo "Press Ctrl+C to stop monitoring."
echo ""

# Command to execute remotely
# 1. Find the latest checkpoint directory based on timestamp in the name
# 2. List the contents of that directory, sorted by time, showing the latest checkpoint file
# 3. Tail the log file
REMOTE_COMMAND="
echo '--- Latest Checkpoint Directory ---';
LATEST_CHKPT_DIR=\$(ls -td $REMOTE_DIR/$CHECKPOINT_DIR_PATTERN 2>/dev/null | head -n 1); # Added error redirection
if [ -d \"\$LATEST_CHKPT_DIR\" ]; then
  echo \"Latest Dir: \$LATEST_CHKPT_DIR\";
  echo '--- Latest Checkpoint File ---';
  ls -lt \"\$LATEST_CHKPT_DIR\" | head -n 5;
else
  echo 'No checkpoint directory found yet.';
fi;
echo;
echo '--- Tailing Log File ($LOG_FILE) ---';
tail -f $REMOTE_DIR/$LOG_FILE"

# Execute the command via SSH
ssh -i "$KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" "$REMOTE_COMMAND"

echo "Monitoring stopped."
