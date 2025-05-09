#!/bin/bash
# Deploy Wav2Vec training script with dataset-specific emotion coding support

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
LOCAL_SCRIPT="fixed_v5_script_dataset_specific.py"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Deploying wav2vec training script with dataset-specific emotion coding..."

# Copy required script to EC2
echo "Copying required script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

# Clear any old cache files to ensure fresh processing
echo "Removing old cache files to ensure fresh data processing..."
ssh -i $KEY_PATH $EC2_HOST "rm -f $REMOTE_DIR/checkpoints/wav2vec_six_classes_best.weights.h5"

# Launch the training
echo "Launching training on EC2..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_dataset_specific_$TIMESTAMP.log"
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 fixed_v5_script_dataset_specific.py > $LOG_FILE 2>&1 &"

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
cat > monitor_dataset_specific.sh << EOL
#!/bin/bash
# Monitor training for the dataset-specific emotion model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="$LOG_FILE"
REMOTE_DIR="$REMOTE_DIR"

echo "Finding the most recent dataset-specific training log file..."
LOG_FILE=\$(ssh -i \$KEY_PATH \$EC2_HOST "find \$REMOTE_DIR -name 'wav2vec_dataset_specific_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
echo "Using log file: \$LOG_FILE"
echo "==============================================================="

# Check if the process is still running
echo "Checking if training process is running..."
PID=\$(ssh -i \$KEY_PATH \$EC2_HOST "pgrep -f fixed_v5_script_dataset_specific.py")
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

echo ""
echo "Monitor complete. Run this script again to see updated progress."
EOL

chmod +x monitor_dataset_specific.sh
echo "Created monitoring script: monitor_dataset_specific.sh"
