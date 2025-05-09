#!/bin/bash

# Script to continue training for models showing positive trends
# Based on last 10 epochs trend analysis

echo "=========================================================="
echo "10-Epoch Training Continuation Script"
echo "=========================================================="
echo

# Settings for continued training
ADDITIONAL_EPOCHS=30
AWS_INSTANCE="ec2-user@3.235.76.0"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

# Models with strong positive trends
CONTINUE_MODELS=(
  "branched_optimizer"
  "hybrid_attention_training"
  "branched_regularization"
)

# Models with slight positive trends
CONSIDER_MODELS=(
  "branched_cross_attention"
  "branched_tcn"
  "branched_focal_loss"
  "branched_self_attention"
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

echo "Models that showed strong positive trends over the last 10 epochs:"
for model in "${CONTINUE_MODELS[@]}"; do
  echo "- $model"
done
echo

echo "Models that showed slight positive trends over the last 10 epochs:"
for model in "${CONSIDER_MODELS[@]}"; do
  echo "- $model"
done
echo

# Execute training continuation for recommended models
for model in "${CONTINUE_MODELS[@]}"; do
  continue_training "$model" $ADDITIONAL_EPOCHS
done

# Ask if user wants to train the "consider" models too
read -p "Would you like to continue training for the models with slight positive trends? (y/n): " train_consider
if [[ "$train_consider" == "y" ]]; then
  for model in "${CONSIDER_MODELS[@]}"; do
    continue_training "$model" $ADDITIONAL_EPOCHS
  done
fi

echo "=========================================================="
echo "Monitoring training progress"
echo "=========================================================="
echo "To monitor training progress, use:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"tail -f ~/emotion_training/training_MODEL_NAME.log\""
echo
echo "To check validation accuracy after training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \"grep -a 'val_accuracy:' ~/emotion_training/training_MODEL_NAME.log | tail -10\""
echo
echo "Script completed."
