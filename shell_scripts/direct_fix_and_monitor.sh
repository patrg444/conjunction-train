#!/bin/bash
# This script will directly fix the issues on the server and monitor the results

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
SCRIPT_PATH="/home/ubuntu/audio_emotion/complete_wav2vec_solution.py"

echo "===== Directly Fixing the Wav2Vec Script on Server ====="

# Create a direct fix script to run on server
cat > direct_fix.sh << 'EOF'
#!/bin/bash
# Fix both the comma issues in the set_value calls and make sure padding is properly implemented

# Path to the script we want to fix
SCRIPT_PATH="/home/ubuntu/audio_emotion/complete_wav2vec_solution.py"

# First, let's back up the original file
cp "$SCRIPT_PATH" "${SCRIPT_PATH}.bak_$(date +%Y%m%d_%H%M%S)"

# Now fix the comma issues directly using sed
echo "Fixing set_value comma issues..."
sed -i 's/\(self\.model\.optimizer\.learning_rate\) \(warmup_lr\)/\1, \2/g' "$SCRIPT_PATH"
sed -i 's/\(self\.model\.optimizer\.learning_rate\) \(new_lr\)/\1, \2/g' "$SCRIPT_PATH"

# Verify the fix worked
echo "Verifying fix..."
grep -n "set_value" "$SCRIPT_PATH"

echo "Fix completed"
EOF

# Upload the fix script
chmod +x direct_fix.sh
scp -i $KEY_PATH direct_fix.sh $SERVER:/tmp/

# Execute the fix script on the server
echo "Applying fix on server..."
ssh -i $KEY_PATH $SERVER "chmod +x /tmp/direct_fix.sh && /tmp/direct_fix.sh"

# Stop any existing processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*wav2vec.*\.py' || true"

# Start training with the fixed script
echo "Starting training with fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_fixed_direct_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &"

# Create monitoring script
cat > monitor_direct_fixed.sh << EOF
#!/bin/bash
# Script to monitor the directly fixed version

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_fixed_direct_*.log | head -1"
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

chmod +x monitor_direct_fixed.sh

# Clean up
rm direct_fix.sh

echo "===== Fix Applied and Training Restarted ====="
echo "The comma issues have been directly fixed on the server."
echo "Training has been restarted with the fixed script."
echo "To monitor the training progress, run: ./monitor_direct_fixed.sh"
