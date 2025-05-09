#!/bin/bash
# This script directly edits the file on the server using sed
# to make the exact comma fix needed

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
REMOTE_FILE="/home/ubuntu/audio_emotion/fixed_v6_script_final.py"

echo "Directly fixing the comma issue on the server..."

# Use SSH to run sed command directly on the server to fix the missing comma
ssh -i $KEY_PATH $EC2_HOST "sed -i 's/tf.keras.backend.set_value(self.model.optimizer.learning_rate warmup_lr)/tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)/g' $REMOTE_FILE"

# Verify the fix worked by printing the line
echo "Verifying the fix worked (should see a comma before 'warmup_lr'):"
ssh -i $KEY_PATH $EC2_HOST "grep -n 'set_value.*learning_rate' $REMOTE_FILE"

# Stop any existing training
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $EC2_HOST "pkill -f fixed_v6_script_final.py || true"

# Launch the training
echo "Launching training with directly fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_direct_fix_$TIMESTAMP.log"
ssh -i $KEY_PATH $EC2_HOST "cd /home/ubuntu/audio_emotion && nohup python3 fixed_v6_script_final.py > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: /home/ubuntu/audio_emotion/$LOG_FILE"
echo ""
echo "To monitor training progress use the included monitoring script:"
echo "  ./monitor_direct_fix.sh"

# Create monitor script
cat > monitor_direct_fix.sh << EOL
#!/bin/bash
# Monitor training for the directly fixed wav2vec model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="$LOG_FILE"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Finding the most recent direct fix training log file..."
LOG_FILE=\$(ssh -i \$KEY_PATH \$EC2_HOST "find \$REMOTE_DIR -name 'wav2vec_direct_fix_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
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

chmod +x monitor_direct_fix.sh
echo "Created monitoring script: monitor_direct_fix.sh"
