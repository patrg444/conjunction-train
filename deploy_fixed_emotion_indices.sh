#!/bin/bash
# Deploy and launch the wav2vec training script with fixed continuous emotion indices
# This ensures the model has exactly 6 emotion classes with proper index mapping

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_SCRIPT="./fixed_v5_script_continuous_indices.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="wav2vec_six_classes_${TIMESTAMP}.log"

echo "Deploying wav2vec training script with continuous emotion indices (6 classes)"

# Make the script executable locally
chmod +x ${LOCAL_SCRIPT}

# Copy the required scripts
echo "Copying required scripts to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/
scp -i $KEY_PATH fixed_v4_script.py $EC2_HOST:$REMOTE_DIR/

# Delete any existing cache to ensure fresh data processing
echo "Removing old cache files to ensure fresh data processing..."
ssh -i $KEY_PATH $EC2_HOST "rm -f $REMOTE_DIR/cache/wav2vec_data_cache_neutral_calm_v2.npz"

# Launch training in the background with nohup
echo "Launching training on EC2..."
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 $(basename $LOCAL_SCRIPT) \
  --data_dir=$REMOTE_DIR/models/wav2vec \
  --batch_size=64 \
  --epochs=100 \
  --lr=0.001 \
  --dropout=0.5 \
  --model_name=wav2vec_six_classes \
  > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo ""
echo "To monitor training progress create a monitoring script with:"
echo '#!/bin/bash'
echo "ssh -i $KEY_PATH $EC2_HOST \"tail -n 50 $REMOTE_DIR/$LOG_FILE\""
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i $KEY_PATH -L 6006:localhost:6006 $EC2_HOST"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"
