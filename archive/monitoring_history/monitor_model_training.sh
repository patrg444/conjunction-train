#!/bin/bash

# Script to monitor the progress of the three models being trained with extended epochs

echo "=========================================================="
echo "  MONITORING TRAINING PROGRESS"
echo "=========================================================="
echo

# Settings
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
AWS_INSTANCE="ec2-user@3.235.76.0"

# Models to monitor
MODELS=(
  "branched_optimizer"
  "hybrid_attention_training"
  "branched_regularization"
)

# Function to check if a model is still training
check_training_status() {
  local model=$1
  echo "Checking training status for $model..."
  
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

# Function to show recent progress
show_recent_progress() {
  local model=$1
  local log_file="training_${model}.log"
  
  echo "Recent training progress for $model:"
  echo "-----------------------------------------------------------"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "grep -a 'val_accuracy:' ~/emotion_training/$log_file | tail -3" 2>/dev/null
  echo
}

# Main monitoring loop
echo "Checking training status for all models..."
echo

for model in "${MODELS[@]}"; do
  echo "=========================================================="
  echo "  MODEL: $model"
  echo "=========================================================="
  
  check_training_status "$model"
  status=$?
  
  show_recent_progress "$model"
  
  if [ $status -eq 0 ]; then
    echo "To monitor training logs in real-time:"
    echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"tail -f ~/emotion_training/training_${model}.log\""
  else
    echo "To check if training completed successfully:"
    echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"tail -n 20 ~/emotion_training/training_${model}.log\""
  fi
  echo
done

echo "To re-run this monitor script:"
echo "./monitor_model_training.sh"
echo
echo "After training completes, run the analyzer to see the new trends:"
echo "./analyze_last_10_epochs.py"
