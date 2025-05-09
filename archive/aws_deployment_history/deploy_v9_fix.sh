#!/bin/bash
# Script to deploy the improved WAV2VEC emotion recognition model with attention mechanism

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"

echo "===== Deploying WAV2VEC Attention Model (v9) ====="

# Upload the improved training script to the server
echo "Uploading fixed_v9_attention.py to server..."
scp -i $KEY_PATH fixed_v9_attention.py $SERVER:/home/ubuntu/audio_emotion/

# Make the script executable
ssh -i $KEY_PATH $SERVER "chmod +x /home/ubuntu/audio_emotion/fixed_v9_attention.py"

# Stop any existing processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*wav2vec.*\.py' || true"

# Start training with the improved script
echo "Starting training with attention-based model..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_v9_attention_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python fixed_v9_attention.py > $LOG_FILE 2>&1 &"

echo "Training launched with log file: $LOG_FILE"
echo ""
echo "To monitor the progress, use the monitoring script: ./monitor_v9_attention.sh"
echo "===== Deployment Complete ====="
