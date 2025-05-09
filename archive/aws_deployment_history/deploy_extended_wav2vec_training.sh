#!/bin/bash
# Deploy and launch the extended wav2vec training script on EC2
# This script continues training from the best checkpoint for additional epochs

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_SCRIPT="./train_wav2vec_extended_epochs.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="wav2vec_extended_training_${TIMESTAMP}.log"

# Path to the best weights from the previous training
BEST_WEIGHTS="${REMOTE_DIR}/checkpoints/wav2vec_audio_only_fixed_v4_restarted_20250422_132858_best.weights.h5"

echo "Deploying extended wav2vec training script"

# Make the script executable locally
chmod +x ${LOCAL_SCRIPT}

# Copy the required scripts
echo "Copying required scripts to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/
scp -i $KEY_PATH fixed_v4_script.py $EC2_HOST:$REMOTE_DIR/

# Launch training in the background with nohup
echo "Launching extended training on EC2..."
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 $(basename $LOCAL_SCRIPT) \
  --data_dir=$REMOTE_DIR/models/wav2vec \
  --weights_path=$BEST_WEIGHTS \
  --batch_size=64 \
  --epochs=100 \
  --lr=0.0005 \
  --dropout=0.5 \
  --model_name=wav2vec_continued \
  > $LOG_FILE 2>&1 &"

echo "Extended training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo ""
echo "To monitor training progress, create a monitoring script with:"
echo '#!/bin/bash'
echo "ssh -i $KEY_PATH $EC2_HOST \"tail -n 50 $REMOTE_DIR/$LOG_FILE\""
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i $KEY_PATH -L 6006:localhost:6006 $EC2_HOST"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"
