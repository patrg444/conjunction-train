#!/bin/bash

# Monitor the Facenet Feature Extraction Process

LOG_FILE="/home/ubuntu/emotion-recognition/process_facenet_features.log"
REMOTE_HOST="ubuntu@18.208.166.91"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "=== Monitoring Facenet Feature Extraction ==="
echo "Log file: $LOG_FILE"
echo "Connecting to ${REMOTE_HOST##*@}..."
echo "Press Ctrl+C to stop monitoring."
echo ""

# Tail the log file
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

# Wait for the tail process to finish (or be interrupted)
wait $TAIL_PID

# Final cleanup in case the loop exits unexpectedly
cleanup
