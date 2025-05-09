#!/bin/bash

# Script to continue training for the three models showing strong positive trends
# Based on 10-epoch trend analysis

echo "=========================================================="
echo "Training Models with Strong Positive Trends"
echo "=========================================================="
echo

# Settings for continued training
ADDITIONAL_EPOCHS=30
AWS_INSTANCE="ec2-user@3.235.76.0"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

# Models with strong positive trends
MODELS=(
  "branched_optimizer"     # 10-epoch trend: 0.0083
  "hybrid_attention_training" # 10-epoch trend: 0.0072
  "branched_regularization"   # 10-epoch trend: 0.0032
)

echo "The following models will be trained for $ADDITIONAL_EPOCHS more epochs:"
for model in "${MODELS[@]}"; do
  echo "- $model"
done
echo

# Function to continue training for a specific model
continue_training() {
  local model=$1
  echo "=========================================================="
  echo "Continuing training for model: $model"
  echo "=========================================================="
  
  # Construct the SSH command to continue training
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "cd ~/emotion_training && \
    echo 'Continuing training for $model for $ADDITIONAL_EPOCHS more epochs' && \
    python continue_training.py --model $model --epochs $ADDITIONAL_EPOCHS \
    >> training_${model}.log 2>&1"
  
  echo "Requested continuation of training for $model"
  echo "Training logs will be appended to ~/emotion_training/training_${model}.log"
  echo
}

# Execute training continuation for each model
for model in "${MODELS[@]}"; do
  continue_training $model
done

echo "=========================================================="
echo "Monitoring training progress"
echo "=========================================================="
echo "To monitor training progress, use:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"tail -f ~/emotion_training/training_MODEL_NAME.log\""
echo
echo "To check validation accuracy at the end of training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"grep -a 'val_accuracy:' ~/emotion_training/training_MODEL_NAME.log | tail -10\""
echo
echo "To analyze trends after additional training:"
echo "./analyze_last_10_epochs.py"
echo
echo "Training initiated for all models with strong positive trends."
