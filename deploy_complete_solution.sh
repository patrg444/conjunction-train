#!/bin/bash
# Script to deploy the complete solution that fixes all issues:
# 1. Missing commas in set_value calls
# 2. Correct key names for NPZ files (wav2vec_features)
# 3. Variable sequence length padding

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_SCRIPT_PATH="/home/ubuntu/audio_emotion/complete_wav2vec_solution.py"
LOCAL_FIXED_SCRIPT="complete_solution.py"

echo "===== Deploying Complete Wav2Vec Solution ====="

# Upload the complete solution to the server
echo "Uploading the complete solution to the server..."
scp -i $KEY_PATH $LOCAL_FIXED_SCRIPT $SERVER:$REMOTE_SCRIPT_PATH

if [ $? -ne 0 ]; then
    echo "Error: Failed to upload the complete solution to the server."
    exit 1
fi

# Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*wav2vec.*\.py' || true"

# Start the new training process with the complete solution
echo "Starting the new training process with the complete solution..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_complete_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $REMOTE_SCRIPT_PATH > $LOG_FILE 2>&1 &"

# Create monitoring script
echo "Creating monitoring script..."
cat > monitor_complete_solution.sh << EOF
#!/bin/bash
# Script to monitor the progress of the complete Wav2Vec solution

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_complete_*.log | head -1"
LATEST_LOG=\$(ssh -i \$KEY_PATH \$SERVER "\$SSH_CMD")
echo "Using log file: \$LATEST_LOG"
echo "==============================================================="

echo "Checking if training process is running..."
PROCESS_COUNT=\$(ssh -i \$KEY_PATH \$SERVER "ps aux | grep 'python.*complete_wav2vec_solution.py' | grep -v grep | wc -l")
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
ssh -i \$KEY_PATH \$SERVER "grep -A10 'emotion:' \$LATEST_LOG | head -15"
echo ""

echo "Check for sequence length statistics:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -A5 'Sequence length statistics' \$LATEST_LOG"
echo ""

echo "Check for class encoding:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep 'Number of classes after encoding' \$LATEST_LOG"
ssh -i \$KEY_PATH \$SERVER "grep 'Original unique label values' \$LATEST_LOG"
echo ""

echo "Check for padding information:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep 'Padding sequences to length' \$LATEST_LOG"
ssh -i \$KEY_PATH \$SERVER "grep 'Padded train shape' \$LATEST_LOG"
echo ""

echo "Check for training progress (epochs):"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -A1 'Epoch [0-9]' \$LATEST_LOG | tail -10"
ssh -i \$KEY_PATH \$SERVER "grep 'val_accuracy' \$LATEST_LOG | tail -5"
echo ""

echo "Check for any errors:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -i error \$LATEST_LOG | tail -10"
echo ""

echo "Monitor complete. Run this script again to see updated progress."
EOF

chmod +x monitor_complete_solution.sh

# Create download script for the trained model
echo "Creating download script..."
cat > download_complete_model.sh << EOF
#!/bin/bash
# Script to download the model trained with the complete solution

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_CHECKPOINT_DIR="/home/ubuntu/audio_emotion/checkpoints"
LOCAL_CHECKPOINT_DIR="./checkpoints_wav2vec_complete"

echo "===== Downloading Complete Wav2Vec Model ====="

# Create local directory
mkdir -p \$LOCAL_CHECKPOINT_DIR

# Download the best model
echo "Downloading best model..."
scp -i \$KEY_PATH \$SERVER:\$REMOTE_CHECKPOINT_DIR/best_model.h5 \$LOCAL_CHECKPOINT_DIR/

# Download the final model
echo "Downloading final model..."
scp -i \$KEY_PATH \$SERVER:\$REMOTE_CHECKPOINT_DIR/final_model.h5 \$LOCAL_CHECKPOINT_DIR/

# Download label classes
echo "Downloading label encoder classes..."
scp -i \$KEY_PATH \$SERVER:\$REMOTE_CHECKPOINT_DIR/label_classes.npy \$LOCAL_CHECKPOINT_DIR/

# Download normalization parameters
echo "Downloading normalization parameters..."
scp -i \$KEY_PATH \$SERVER:/home/ubuntu/audio_emotion/audio_mean.npy \$LOCAL_CHECKPOINT_DIR/
scp -i \$KEY_PATH \$SERVER:/home/ubuntu/audio_emotion/audio_std.npy \$LOCAL_CHECKPOINT_DIR/

echo "===== Download Complete ====="
echo "Models saved to \$LOCAL_CHECKPOINT_DIR"
EOF

chmod +x download_complete_model.sh

echo "===== Deployment Complete ====="
echo "The complete solution has been deployed and training has been restarted."
echo "This solution fixes ALL THREE issues:"
echo "  1. The comma syntax error in set_value calls"
echo "  2. The NPZ key name mismatch for wav2vec_features"
echo "  3. The variable sequence length issue with padding"
echo ""
echo "To monitor the training progress, run: ./monitor_complete_solution.sh"
echo "To download the trained model when done, run: ./download_complete_model.sh"
