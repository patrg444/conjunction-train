#!/bin/bash

# Monitor the Video-Only Facenet + LSTM Training

LOG_FILE="/home/ubuntu/emotion-recognition/video_only_facenet_lstm_full.log"
CHECKPOINT_PATTERN="/home/ubuntu/emotion-recognition/models/video_only_facenet_lstm_*"
REMOTE_HOST="ubuntu@18.208.166.91"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "=== Monitoring Video-Only Facenet + LSTM Training ==="
echo "Log file: $LOG_FILE"
echo "Checkpoint pattern: $CHECKPOINT_PATTERN"
echo "Connecting to ${REMOTE_HOST##*@}..."
echo "Press Ctrl+C to stop monitoring."
echo ""

# Function to fetch and display latest checkpoint directory and file
monitor_checkpoints() {
    echo "--- Latest Checkpoint Directory ---"
    LATEST_DIR=$(ssh -i "$SSH_KEY" "$REMOTE_HOST" "ls -td $CHECKPOINT_PATTERN | head -n 1" 2>/dev/null)
    if [ -n "$LATEST_DIR" ]; then
        echo "Latest Dir: $LATEST_DIR"
        echo "--- Latest Checkpoint File ---"
        # Look for the specific best model file name
        ssh -i "$SSH_KEY" "$REMOTE_HOST" "ls -lt $LATEST_DIR | grep 'best_model_video_only_facenet_lstm.keras' | head -n 1" 2>/dev/null || echo "No best model file found yet."
    else
        echo "No checkpoint directory found yet."
    fi
    echo ""
}

# Initial checkpoint check
monitor_checkpoints

# Tail the log file and periodically check checkpoints
ssh -i "$SSH_KEY" "$REMOTE_HOST" "tail -f $LOG_FILE" &
TAIL_PID=$!

# Function to kill the tail process on exit
cleanup() {
    echo "Stopping log tail..."
    kill $TAIL_PID
    wait $TAIL_PID 2>/dev/null
    echo "Monitoring stopped."
}

# Trap Ctrl+C and exit signals to run cleanup
trap cleanup SIGINT SIGTERM EXIT

# Periodically check checkpoints while tail is running
while kill -0 $TAIL_PID 2> /dev/null; do
    sleep 60 # Check every 60 seconds
    # Uncomment the line below if you want periodic checkpoint updates in the terminal
    # monitor_checkpoints
done

# Final cleanup in case the loop exits unexpectedly
cleanup
