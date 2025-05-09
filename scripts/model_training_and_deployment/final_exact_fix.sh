#!/bin/bash
# This script directly fixes the exact lines with missing commas on the server

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
SCRIPT_PATH="/home/ubuntu/audio_emotion/complete_wav2vec_solution.py"

echo "===== Directly Fixing the Exact Lines on Server ====="

# Create a direct fix script with exact line replacements
cat > exact_fix.sh << 'EOF'
#!/bin/bash
# Fix the specific lines with missing commas

# Path to the script we want to fix
SCRIPT_PATH="/home/ubuntu/audio_emotion/complete_wav2vec_solution.py"

# First, let's back up the original file
cp "$SCRIPT_PATH" "${SCRIPT_PATH}.bak_exact_fix"

# Use exact line replacements for the comma issues
# Line 63
sed -i '63s/learning_rate warmup_lr/learning_rate, warmup_lr/g' "$SCRIPT_PATH"

# Line 87
sed -i '87s/learning_rate new_lr/learning_rate, new_lr/g' "$SCRIPT_PATH"

# Verify the changes
echo "Verifying changes..."
grep -n "set_value.*learning_rate" "$SCRIPT_PATH"

echo "Fix completed"
EOF

# Upload the fix script
chmod +x exact_fix.sh
scp -i $KEY_PATH exact_fix.sh $SERVER:/tmp/

# Execute the fix script on the server
echo "Applying exact line fix on server..."
ssh -i $KEY_PATH $SERVER "chmod +x /tmp/exact_fix.sh && /tmp/exact_fix.sh"

# Stop any existing processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*wav2vec.*\.py' || true"

# Update our complete solution script and deploy it (in case the padding fixes were missed)
echo "Uploading complete solution with proper commas..."
sed -i 's/\(self\.model\.optimizer\.learning_rate\) \(warmup_lr\)/\1, \2/g' complete_solution.py
sed -i 's/\(self\.model\.optimizer\.learning_rate\) \(new_lr\)/\1, \2/g' complete_solution.py
scp -i $KEY_PATH complete_solution.py $SERVER:$SCRIPT_PATH

# Start training with the fixed script
echo "Starting training with fully fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_exact_fixed_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &"

# Create monitoring script
cat > monitor_exact_fix.sh << EOF
#!/bin/bash
# Script to monitor the accurately fixed version

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_exact_fixed_*.log | head -1"
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

echo "Check for sequence length statistics:"
echo "==============================================================="
ssh -i \$KEY_PATH \$SERVER "grep -A5 'Sequence length statistics' \$LATEST_LOG"
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
ssh -i \$KEY_PATH \$SERVER "grep -i error \$LATEST_LOG | tail -5"
echo ""

echo "Monitor complete. Run this script again to see updated progress."
EOF

chmod +x monitor_exact_fix.sh

# Clean up
rm exact_fix.sh

# Create a script for downloading the model
cat > download_fixed_model.sh << EOF
#!/bin/bash
# Script to download the model trained with the fixed solution

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_CHECKPOINT_DIR="/home/ubuntu/audio_emotion/checkpoints"
LOCAL_CHECKPOINT_DIR="./checkpoints_wav2vec_final"

echo "===== Downloading Final Wav2Vec Model ====="

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

chmod +x download_fixed_model.sh

echo "===== Exact Fix Applied and Training Restarted ====="
echo "The comma issues have been addressed with targeted line replacements."
echo "Training has been restarted with the fully fixed script."
echo "To monitor the training progress, run: ./monitor_exact_fix.sh"
echo "To download the model when training is complete, run: ./download_fixed_model.sh"
