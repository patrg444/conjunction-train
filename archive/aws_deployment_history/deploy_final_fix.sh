#!/bin/bash
# Deploy the final fixed Wav2Vec training script that handles dataset-specific emotion coding
# and uses the correct wav2vec_features key

# Check if key exists
if [ ! -f ~/Downloads/gpu-key.pem ]; then
  echo "Error: SSH key not found at ~/Downloads/gpu-key.pem"
  exit 1
fi

# Set permissions if needed
chmod 400 ~/Downloads/gpu-key.pem

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOCAL_SCRIPT="fixed_v6_script_final.py"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Deploying final fixed wav2vec training script..."

# Copy required script to EC2
echo "Copying script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

# Clear any old cache files to ensure fresh processing
echo "Removing old cache files to ensure fresh data processing..."
ssh -i $KEY_PATH $EC2_HOST "rm -f $REMOTE_DIR/checkpoints/wav2vec_six_classes_best.weights.h5"

# Launch the training
echo "Launching training on EC2..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_final_fix_$TIMESTAMP.log"
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 fixed_v6_script_final.py > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo ""
echo "To monitor training progress create a monitoring script with:"
echo "#!/bin/bash"
echo "ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \"tail -n 50 $REMOTE_DIR/$LOG_FILE\""
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"

# Create monitor script
cat > monitor_final_fix.sh << EOL
#!/bin/bash
# Monitor training for the final fixed wav2vec model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="$LOG_FILE"
REMOTE_DIR="$REMOTE_DIR"

echo "Finding the most recent final fix training log file..."
LOG_FILE=\$(ssh -i \$KEY_PATH \$EC2_HOST "find \$REMOTE_DIR -name 'wav2vec_final_fix_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
echo "Using log file: \$LOG_FILE"
echo "==============================================================="

# Check if the process is still running
echo "Checking if training process is running..."
PID=\$(ssh -i \$KEY_PATH \$EC2_HOST "pgrep -f fixed_v6_script_final.py")
if [ -z "\$PID" ]; then
    echo "PROCESS NOT RUNNING!"
else
    echo "Process is running with PID \$PID"
fi

echo ""
echo "Latest log entries:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "tail -n 50 \$LOG_FILE"

# Check for emotion distribution information
echo ""
echo "Check for emotion distribution information:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep -A20 'Emotion distribution in dataset' \$LOG_FILE | tail -20"

# Check for proper class encoding
echo ""
echo "Check for proper class encoding:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep 'Number of classes after encoding' \$LOG_FILE"
ssh -i \$KEY_PATH \$EC2_HOST "grep 'Original unique label values' \$LOG_FILE"

# Check for training progress
echo ""
echo "Check for latest training epoch:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep -E 'Epoch [0-9]+/100' \$LOG_FILE | tail -5"
ssh -i \$KEY_PATH \$EC2_HOST "grep -E 'val_accuracy: [0-9.]+' \$LOG_FILE | tail -5"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
EOL

chmod +x monitor_final_fix.sh
echo "Created monitoring script: monitor_final_fix.sh"

# Create download script for the final model
cat > download_final_model.sh << EOL
#!/bin/bash
# Download the final fixed wav2vec model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_FILE="checkpoints/wav2vec_six_classes_best.weights.h5"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOCAL_MODEL="wav2vec_final_fixed_model_\$TIMESTAMP.h5"

echo "Downloading final fixed wav2vec model..."
scp -i \$KEY_PATH \$EC2_HOST:\$REMOTE_DIR/\$MODEL_FILE \$LOCAL_MODEL

if [ -f "\$LOCAL_MODEL" ]; then
    echo "Model successfully downloaded to \$LOCAL_MODEL"
else
    echo "Error: Model download failed"
fi
EOL

chmod +x download_final_model.sh
echo "Created download script: download_final_model.sh"
