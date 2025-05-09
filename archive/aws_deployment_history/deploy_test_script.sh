#!/bin/bash
# Deploy and run the test script to debug the learning rate issue

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"

echo "===== Deploying Learning Rate Debug Script ====="

# Upload the debug script to the server
scp -i $KEY_PATH debug_learning_rate.py $SERVER:/home/ubuntu/audio_emotion/

# Run the debug script on the server
echo "Running debug script..."
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && python debug_learning_rate.py"

echo "===== Debug Complete ====="
