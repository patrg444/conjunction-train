#!/bin/bash

# Script to continue training for models showing positive trends
# Created based on analysis of training logs

echo "=========================================================="
echo "Training Continuation Script for Improving Models"
echo "=========================================================="
echo

# Define models that showed positive trends in validation accuracy
IMPROVING_MODELS=(
  "branched_optimizer"       # Slope: 0.0169, Final Acc: 0.7610
  "hybrid_attention_training" # Slope: 0.0115, Final Acc: 0.6552
  "rl_model"                 # Slope: 0.0068, Final Acc: 0.1704
  "branched_sync_aug"        # Slope: 0.0014, Final Acc: 0.8453
)

# Settings for continued training
ADDITIONAL_EPOCHS=20
AWS_INSTANCE="ec2-user@3.235.76.0"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

echo "The following models will be trained for $ADDITIONAL_EPOCHS more epochs:"
for model in "${IMPROVING_MODELS[@]}"; do
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
for model in "${IMPROVING_MODELS[@]}"; do
  continue_training $model
done

echo "=========================================================="
echo "Monitoring training progress"
echo "=========================================================="
echo "To monitor training progress, use:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"tail -f ~/emotion_training/training_MODEL_NAME.log\""
echo
echo "To check validation accuracy at the end of training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"grep -a 'val_accuracy:' ~/emotion_training/training_MODEL_NAME.log | tail -5\""
echo
echo "Script completed."
