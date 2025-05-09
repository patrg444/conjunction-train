#!/bin/bash

# Script to continue training for models showing positive trends
# Based on extended trend analysis

echo "=========================================================="
echo "Extended Training Continuation Script"
echo "=========================================================="
echo

# Settings for continued training
ADDITIONAL_EPOCHS=30
AWS_INSTANCE="ec2-user@3.235.76.0"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

# Definite continue models
DEFINITE_CONTINUE_MODELS=(
  "branched_optimizer"
  "hybrid_attention_training"
)

# Consider continue models
CONSIDER_CONTINUE_MODELS=(
  "rl_model"
  "branched_tcn"
  "branched_sync_aug"
)

# Function to continue training for a specific model
continue_training() {
  local model=$1
  local epochs=$2
  echo "=========================================================="
  echo "Continuing training for model: $model ($epochs epochs)"
  echo "=========================================================="
  
  # Construct the SSH command to continue training
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "cd ~/emotion_training && \
    echo 'Continuing training for $model for $epochs more epochs' && \
    python continue_training.py --model $model --epochs $epochs \
    >> training_${model}.log 2>&1"
  
  echo "Requested continuation of training for $model"
  echo "Training logs will be appended to ~/emotion_training/training_${model}.log"
  echo
}

echo "Models that will DEFINITELY be trained for $ADDITIONAL_EPOCHS more epochs:"
for model in "${DEFINITE_CONTINUE_MODELS[@]}"; do
  echo "- $model"
done
echo

echo "Models that will be CONSIDERED for further training:"
for model in "${CONSIDER_CONTINUE_MODELS[@]}"; do
  echo "- $model"
done
echo

# Ask user which models to train
read -p "Train definitely recommended models? (y/n): " train_definite
read -p "Train models to consider as well? (y/n): " train_consider
echo

# Execute training continuation for definite models
if [[ "$train_definite" == "y" ]]; then
  for model in "${DEFINITE_CONTINUE_MODELS[@]}"; do
    continue_training "$model" $ADDITIONAL_EPOCHS
  done
fi

# Execute training continuation for considered models
if [[ "$train_consider" == "y" ]]; then
  for model in "${CONSIDER_CONTINUE_MODELS[@]}"; do
    continue_training "$model" $ADDITIONAL_EPOCHS
  done
fi

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
