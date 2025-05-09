#!/bin/bash

# Script to continuously monitor the progress of the three models
# being trained with 100 epochs

echo "=========================================================="
echo "  CONTINUOUS TRAINING MONITORING"
echo "=========================================================="
echo "Press Ctrl+C to exit"
echo

# Settings
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
AWS_INSTANCE="ec2-user@3.235.76.0"
CHECK_INTERVAL=300  # Check every 5 minutes (300 seconds)

# Models to monitor
MODELS=(
  "branched_optimizer"
  "hybrid_attention_training"
  "branched_regularization"
)

# Function to check if a model is still training
check_training_status() {
  local model=$1
  # Check if training process is running on the remote server
  RUNNING=$(ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "ps aux | grep train_${model//_attention_training/} | grep -v grep | wc -l")

  if [ "$RUNNING" -gt "0" ]; then
    echo "✓ Training is ACTIVE for $model"
    return 0
  else
    echo "✗ No active training process found for $model"
    return 1
  fi
}

# Function to show recent progress and extract current epoch
show_recent_progress() {
  local model=$1
  local log_file="training_${model}.log"

  echo "Recent training progress for $model:"
  echo "-----------------------------------------------------------"
  # Get the last 3 validation accuracy lines
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "grep -a 'val_accuracy:' ~/emotion_training/$log_file | tail -3" 2>/dev/null
  
  # Try to extract current epoch
  EPOCH_INFO=$(ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "grep -a 'Epoch 0' ~/emotion_training/$log_file | tail -1" 2>/dev/null)
  if [[ ! -z "$EPOCH_INFO" ]]; then
    echo "Current epoch: $EPOCH_INFO"
  fi
  
  echo
}

# Function to get training duration
get_training_duration() {
  local model=$1
  # Get process start time
  PROCESS_TIME=$(ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "ps -eo pid,etime,cmd | grep train_${model//_attention_training/} | grep -v grep | awk '{print \$2}'" 2>/dev/null)
  
  if [[ ! -z "$PROCESS_TIME" ]]; then
    echo "Training duration: $PROCESS_TIME"
  else
    echo "Training duration: Unknown"
  fi
}

# Main monitoring loop
while true; do
  clear
  echo "=========================================================="
  echo "  CONTINUOUS TRAINING MONITORING"
  echo "=========================================================="
  echo "Last check: $(date)"
  echo "Next check in $CHECK_INTERVAL seconds"
  echo "Press Ctrl+C to exit"
  echo

  all_completed=true

  for model in "${MODELS[@]}"; do
    echo "=========================================================="
    echo "  MODEL: $model"
    echo "=========================================================="

    check_training_status "$model"
    status=$?
    
    if [ $status -eq 0 ]; then
      all_completed=false
      get_training_duration "$model"
      show_recent_progress "$model"
    else
      echo "Training may have completed. Checking final results:"
      ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "tail -n 20 ~/emotion_training/training_${model}.log | grep -a 'val_accuracy'" 2>/dev/null
      echo
    fi
  done

  if $all_completed; then
    echo "=========================================================="
    echo "  ALL TRAINING PROCESSES COMPLETED"
    echo "=========================================================="
    echo "You can analyze the results using:"
    echo "./analyze_last_10_epochs.py"
    break
  fi

  # Wait for the specified interval
  echo "Waiting $CHECK_INTERVAL seconds before next check..."
  sleep $CHECK_INTERVAL
done
