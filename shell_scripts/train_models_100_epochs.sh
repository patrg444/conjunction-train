#!/bin/bash

# Script to deploy and train all three models with 100 epochs
# This will train:
# 1. branched_optimizer
# 2. hybrid_attention
# 3. branched_regularization

echo "=========================================================="
echo "  DEPLOYING ALL MODELS FOR 100 EPOCH TRAINING"
echo "=========================================================="

# Settings for training
AWS_INSTANCE="3.235.76.0"
USERNAME="ec2-user"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"

# First, check if we can connect to the server
echo "Checking connection to AWS instance..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE "echo 'Connection successful'" || {
  echo "Error: Cannot connect to AWS instance. Check your SSH key and connection details."
  exit 1
}

echo "=========================================================="
echo "  STEP 1: Creating scripts with 100 epochs"
echo "=========================================================="

# First create and deploy branched_optimizer model
echo "Creating 100-epoch version of branched_optimizer model..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE "cat > ${REMOTE_DIR}/run_extended_training.sh << 'EOL'
#!/bin/bash

echo \"===================================================\"
echo \"  STARTING TRAINING WITH 100 EPOCHS\"
echo \"===================================================\"

# Stop any currently running training processes
pkill -f 'train_branched_optimizer.py' || true
pkill -f 'train_hybrid_attention.py' || true
pkill -f 'train_branched_regularization.py' || true

# Start branched_optimizer training
echo \"Starting branched_optimizer training...\"
cd ~/emotion_training
rm -f training_branched_optimizer.log
sed -i 's/EPOCHS = 50/EPOCHS = 100/' scripts/train_branched_optimizer.py
nohup python3 scripts/train_branched_optimizer.py > training_branched_optimizer.log 2>&1 &
echo \"Training process started with PID \$!\"
echo \"Logs are being written to training_branched_optimizer.log\"

# Start hybrid_attention_training
echo \"Starting hybrid_attention_training...\"
cd ~/emotion_training
rm -f training_hybrid_attention_training.log
sed -i 's/EPOCHS = 50/EPOCHS = 100/' scripts/train_hybrid_attention.py
nohup python3 scripts/train_hybrid_attention.py > training_hybrid_attention_training.log 2>&1 &
echo \"Training process started with PID \$!\"
echo \"Logs are being written to training_hybrid_attention_training.log\"

# Start branched_regularization training
echo \"Starting branched_regularization training...\"
cd ~/emotion_training
rm -f training_branched_regularization.log
sed -i 's/EPOCHS = 50/EPOCHS = 100/' scripts/train_branched_regularization.py
nohup python3 scripts/train_branched_regularization.py > training_branched_regularization.log 2>&1 &
echo \"Training process started with PID \$!\"
echo \"Logs are being written to training_branched_regularization.log\"

echo \"All training processes started\"
echo \"===================================================\"
EOL"

# Make the script executable
echo "Making the script executable..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE "chmod +x ${REMOTE_DIR}/run_extended_training.sh"

# Run the script
echo "Running extended training script on AWS instance..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE "${REMOTE_DIR}/run_extended_training.sh"

echo "=========================================================="
echo "  MONITORING COMMANDS"
echo "=========================================================="
echo "To monitor branched_optimizer training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE \"tail -f ~/emotion_training/training_branched_optimizer.log\""
echo
echo "To monitor hybrid_attention training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE \"tail -f ~/emotion_training/training_hybrid_attention_training.log\""
echo
echo "To monitor branched_regularization training:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE \"tail -f ~/emotion_training/training_branched_regularization.log\""
echo
echo "To check if training processes are running:"
echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $USERNAME@$AWS_INSTANCE \"ps aux | grep train_\""
echo
echo "=========================================================="
echo "  DEPLOYMENT COMPLETE"
echo "=========================================================="
echo "All three models are now being trained with 100 epochs each."
echo "This will take several hours to complete."
