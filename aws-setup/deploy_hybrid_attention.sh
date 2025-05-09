#!/bin/bash
# Deploy the new hybrid Conv1D-TCN model with cross-modal attention to EC2
# This script:
# 1. Terminates the poorly performing RL model
# 2. Uploads the new hybrid attention model script
# 3. Sets up and launches the training process

set -e  # Exit on any error

# EC2 instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
SCRIPT_PATH="scripts/train_hybrid_attention.py"
DEST_PATH="/home/ec2-user/emotion_training/scripts/"
REMOTE_SCRIPT_NAME="train_hybrid_attention.py"

echo "======================================================"
echo "  DEPLOYING HYBRID CONV1D-TCN MODEL WITH CROSS-MODAL ATTENTION"
echo "======================================================"
echo "Target: $USERNAME@$INSTANCE_IP"
echo "Using key: $KEY_FILE"
echo "Local script: $SCRIPT_PATH"
echo "Remote path: $DEST_PATH"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script file not found: $SCRIPT_PATH"
    exit 1
fi

# Step 1: Terminate the RL model that's performing poorly
echo "Step 1: Terminating the poorly performing RL model (PID 51304)..."
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "if ps -p 51304 > /dev/null; then kill -9 51304; echo 'RL model process terminated'; else echo 'RL model process not found'; fi"

# Step 2: Upload the script
echo "Step 2: Uploading the hybrid model script..."
scp -i "$KEY_FILE" "$SCRIPT_PATH" "$USERNAME@$INSTANCE_IP:$DEST_PATH$REMOTE_SCRIPT_NAME"

# Step 3: Make the script executable
echo "Step 3: Making the script executable..."
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x $DEST_PATH$REMOTE_SCRIPT_NAME"

# Step 4: Launch the training
echo "Step 4: Launching the hybrid model training..."
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd /home/ec2-user/emotion_training && nohup python3 $DEST_PATH$REMOTE_SCRIPT_NAME > hybrid_attention_training.log 2>&1 &"

# Step 5: Verify the process is running
echo "Step 5: Verifying the process is running..."
sleep 2
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "ps aux | grep python | grep -v grep"

echo "======================================================"
echo "  DEPLOYMENT COMPLETE"
echo "======================================================"
echo "The hybrid model is now running on the EC2 instance."
echo "Monitor the training with:"
echo "  ./aws-setup/monitor_hybrid_attention.sh"
echo "Download results with:"
echo "  ./aws-setup/download_hybrid_attention_results.sh"
echo "======================================================"
