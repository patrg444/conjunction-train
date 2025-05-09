#!/bin/bash
# Deploy and run the alternate test script

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"

echo "===== Deploying Alternate Learning Rate Debug Script ====="

# Upload the debug script to the server
scp -i $KEY_PATH debug_learning_rate_alternate.py $SERVER:/home/ubuntu/audio_emotion/

# Run the debug script on the server
echo "Running debug script..."
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && python debug_learning_rate_alternate.py"

echo "===== Debug Complete ====="
