#!/bin/bash
# Script to deploy the fixed script with both comma fix and correct key names to the AWS server

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_SCRIPT_PATH="/home/ubuntu/audio_emotion/fixed_v6_script_final.py"
LOCAL_FIXED_SCRIPT="fixed_script_with_keys.py"

echo "===== Deploying Wav2Vec Data Format & Learning Rate Fix ====="

# 1. Back up the original script on the server
echo "Creating backup of the original script on the server..."
ssh -i $KEY_PATH $SERVER "cp $REMOTE_SCRIPT_PATH ${REMOTE_SCRIPT_PATH}.backup_full"

# 2. Upload the fixed script to the server
echo "Uploading the fixed script to the server..."
scp -i $KEY_PATH $LOCAL_FIXED_SCRIPT $SERVER:$REMOTE_SCRIPT_PATH

if [ $? -ne 0 ]; then
    echo "Error: Failed to upload the fixed script to the server."
    exit 1
fi

# 3. Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*fixed_v6_script_final.py' || true"

# 4. Start the new training process
echo "Starting the new training process with the fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_keys_fixed_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $REMOTE_SCRIPT_PATH > $LOG_FILE 2>&1 &"

# 5. Create monitoring script
echo "Creating monitoring script..."
cat > monitor_fixed_keys.sh << EOF
#!/bin/bash
# Script to monitor the progress of the fixed Wav2Vec training with correct keys

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_keys_fixed_*.log | head -1"
LATEST_LOG=\$(ssh -i \$KEY_PATH \$SERVER "\$SSH_CMD")
echo "Using log file: \$LATEST_LOG"
echo "==============================================================="

echo "Checking if training process is running..."
PROCESS_COUNT=\$(ssh -i \$KEY_PATH \$SERVER "ps aux | grep 'python.*fixed_v6_script_final.py' | grep -v grep | wc -l")
if [ "\$PROCESS_COUNT" -gt 0 ]; then
    echo "PROCESS RUNNING (count: \$PROCESS_COUNT)"
else
    echo "PROCESS NOT RUNNING!"
fi
echo ""

echo "Latest log entries:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "tail -n 30 \$LATEST_LOG"
echo ""

echo "Check for emotion distribution information:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -A10 'emotion:' \$LATEST_LOG | head -10"
echo ""

echo "Check for proper class encoding:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep 'Number of classes after encoding' \$LATEST_LOG"
ssh -i \$KEY_PATH \$SERVER "grep 'Original unique label values' \$LATEST_LOG"
echo ""

echo "Check for training progress (epochs):"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -A1 'Epoch [0-9]' \$LATEST_LOG | tail -10"
echo ""

echo "Monitor complete. Run this script again to see updated progress."
EOF

chmod +x monitor_fixed_keys.sh

echo "===== Deployment Complete ====="
echo "The fixed script has been deployed and training has been restarted."
echo "This version fixes BOTH the comma syntax error AND the NPZ key name mismatch."
echo "To monitor the training progress, run: ./monitor_fixed_keys.sh"
