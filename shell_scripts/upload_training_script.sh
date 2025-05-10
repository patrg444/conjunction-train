#!/bin/bash

# Upload continuation training script to the AWS instance
# This script deploys the continue_training.py script to the EC2 instance

echo "=========================================================="
echo "Uploading Training Continuation Scripts to AWS Instance"
echo "=========================================================="
echo

# Configuration
AWS_INSTANCE="ec2-user@3.235.76.0"
SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOCAL_SCRIPT="continue_training.py"
REMOTE_DIR="~/emotion_training"

# Verify the script exists
if [ ! -f "$LOCAL_SCRIPT" ]; then
  echo "Error: Script $LOCAL_SCRIPT not found!"
  exit 1
fi

# Upload script to the instance
echo "Uploading $LOCAL_SCRIPT to $AWS_INSTANCE:$REMOTE_DIR/"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$LOCAL_SCRIPT" "$AWS_INSTANCE:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
  echo "Successfully uploaded $LOCAL_SCRIPT to the AWS instance"
  
  # Make the script executable on the remote server
  echo "Setting execution permissions..."
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$AWS_INSTANCE" "chmod +x $REMOTE_DIR/$(basename $LOCAL_SCRIPT)"
  
  echo "Done. You can now run the training continuation script."
  echo
  echo "To execute the continuation script, run:"
  echo "./training_continuation_script.sh"
else
  echo "Error: Failed to upload the script to the AWS instance"
  exit 1
fi
