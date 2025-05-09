#!/bin/bash
# Deploy and launch the fixed wav2vec training script (v3) on EC2
# This version supports the actual data directory structure found on the server

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_SCRIPT="./scripts/train_wav2vec_audio_only_fixed_v3.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="wav2vec_fixed_training_v3_${TIMESTAMP}.log"

echo "Deploying improved wav2vec training script with file structure fixes"

# Copy the training script
echo "Copying training script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

# Make the script executable
ssh -i $KEY_PATH $EC2_HOST "chmod +x $REMOTE_DIR/$(basename $LOCAL_SCRIPT)"

# Launch training in the background with nohup
echo "Launching training on EC2..."
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 $(basename $LOCAL_SCRIPT) \
  --data_dir=$REMOTE_DIR/models/wav2vec \
  --batch_size=64 \
  --epochs=100 \
  --lr=0.001 \
  --dropout=0.5 \
  --model_name=wav2vec_audio_only_fixed_v3_${TIMESTAMP} \
  --debug \
  > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo "To monitor training progress:"
echo "  ./monitor_wav2vec_fixed_training_v3.sh"
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i $KEY_PATH -L 6006:localhost:6006 $EC2_HOST"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"
