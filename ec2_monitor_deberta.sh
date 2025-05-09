#!/bin/bash

# Set log directory and log file
LOG_DIR=~/conjunction-train/training_logs_humor
LOG_FILE=/tmp/ec2_deberta_training_log.txt

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

echo "Starting DeBERTa v3 training monitor on EC2 instance..."
echo "Saving checkpoints to: $LOG_DIR/microsoft_deberta-v3-base_single/checkpoints"
echo "Log file: $LOG_FILE"

# Run in the background and redirect to a log file
touch $LOG_FILE
echo "Real-time monitoring activated. Streaming logs and checking for checkpoints..."

# Function to check for checkpoints
check_checkpoints() {
  CHECKPOINT_DIR=$LOG_DIR/microsoft_deberta-v3-base_single/checkpoints
  if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINTS=$(ls -la $CHECKPOINT_DIR 2>/dev/null | grep -c 'epoch=')
    if [ $CHECKPOINTS -gt 0 ]; then
      echo "Found $CHECKPOINTS checkpoints."
      ls -la $CHECKPOINT_DIR | grep 'epoch='
    else
      echo "No checkpoints found yet."
    fi
  else
    echo "No checkpoints found yet."
  fi
  echo "Next checkpoint check in 60 seconds..."
}

# Initial checkpoint check
check_checkpoints

# Start continuous monitoring
while true; do
  # Display most recent logs
  tail -n 20 $(find $LOG_DIR -name "events.out.tfevents.*" -type f -print | sort -r | head -n 1) 2>/dev/null
  
  # Check for checkpoints
  check_checkpoints
  
  # Wait before checking again
  sleep 60
done
