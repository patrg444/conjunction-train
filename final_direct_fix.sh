#!/bin/bash
# Script to directly edit the file on the server to fix the comma issue

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_SCRIPT_PATH="/home/ubuntu/audio_emotion/fixed_v6_script_final.py"

echo "===== Applying Direct Fix to Wav2Vec Script on Server ====="

# Create a temporary SSH script to perform the edits on the server
cat > temp_ssh_fix.sh << 'EOF'
#!/bin/bash
# Fix the comma issue in the set_value calls (line 63 and line 104)
SCRIPT_PATH="/home/ubuntu/audio_emotion/fixed_v6_script_final.py"

# Create a backup first
cp "$SCRIPT_PATH" "${SCRIPT_PATH}.backup_direct_fix"

# Check if there are missing commas in the script by looking for "set_value" lines
echo "Checking for missing commas in set_value calls..."
grep -n "set_value.*learning_rate " "$SCRIPT_PATH"

# Fix both lines directly using sed
echo "Fixing missing commas in set_value calls..."
sed -i '
/set_value.*learning_rate [^,]/s/learning_rate /learning_rate, /g
' "$SCRIPT_PATH"

# Verify the fixes
echo "Verifying fixes..."
grep -n "set_value.*learning_rate" "$SCRIPT_PATH"

# Show diff between original and fixed file
echo "Showing diff between original and fixed file..."
diff "$SCRIPT_PATH" "${SCRIPT_PATH}.backup_direct_fix" || echo "Files are different, patches applied"

echo "Direct fix applied to server script"
EOF

# Upload the fix script to the server
echo "Uploading fix script to server..."
scp -i $KEY_PATH temp_ssh_fix.sh $SERVER:/tmp/

# Make the script executable and run it
echo "Running fix script on server..."
ssh -i $KEY_PATH $SERVER "chmod +x /tmp/temp_ssh_fix.sh && /tmp/temp_ssh_fix.sh"

# Clean up
rm temp_ssh_fix.sh

# Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*fixed_v6_script_final.py' || true"

# Start the new training process with the directly fixed script
echo "Starting the new training process with the directly fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_direct_fixed_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $REMOTE_SCRIPT_PATH > $LOG_FILE 2>&1 &"

# Create monitoring script
echo "Creating monitoring script..."
cat > monitor_direct_fix.sh << EOF
#!/bin/bash
# Script to monitor the progress of the directly fixed Wav2Vec training

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_direct_fixed_*.log | head -1"
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

echo "Monitor complete. Run this script again to see updated progress."
EOF

chmod +x monitor_direct_fix.sh

echo "===== Direct Fix Applied and Training Restarted ====="
echo "The commas have been directly added to the set_value calls on the server."
echo "Training has been restarted with the fixed script."
echo "To monitor the training progress, run: ./monitor_direct_fix.sh"
