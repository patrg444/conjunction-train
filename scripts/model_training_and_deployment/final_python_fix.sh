#!/bin/bash
# Final approach using Python to directly edit the file on the server

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
SCRIPT_PATH="/home/ubuntu/audio_emotion/final_wav2vec_fix.py"

echo "===== Creating Final Python Fix Script ====="

# Create a Python script to edit the file on the server
cat > python_file_editor.py << 'EOF'
#!/usr/bin/env python3
"""
Script to directly fix the Wav2Vec learning rate issue on the server.
This Python script will create a completely new file with the fix applied.
"""

import os
import shutil
import re

# Path to the original script
SOURCE_SCRIPT = "/home/ubuntu/audio_emotion/fixed_wav2vec_final.py"
# Path to the new fixed script
TARGET_SCRIPT = "/home/ubuntu/audio_emotion/final_wav2vec_fix.py"

def fix_script():
    """Fix the set_value calls in the script."""
    with open(SOURCE_SCRIPT, 'r') as f:
        content = f.read()
    
    # Fix the learning_rate warmup_lr line
    content = re.sub(
        r'(tf\.keras\.backend\.set_value\(self\.model\.optimizer\.learning_rate) (warmup_lr\))',
        r'\1, \2',
        content
    )
    
    # Fix the learning_rate new_lr line
    content = re.sub(
        r'(tf\.keras\.backend\.set_value\(self\.model\.optimizer\.learning_rate) (new_lr\))',
        r'\1, \2',
        content
    )
    
    # Write the fixed content to the new file
    with open(TARGET_SCRIPT, 'w') as f:
        f.write(content)
    
    # Make the new script executable
    os.chmod(TARGET_SCRIPT, 0o755)
    
    print(f"Fixed script created at {TARGET_SCRIPT}")
    print("Verification:")
    
    # Read the fixed file to verify the changes
    with open(TARGET_SCRIPT, 'r') as f:
        for i, line in enumerate(f, 1):
            if "set_value" in line and "learning_rate" in line:
                print(f"Line {i}: {line.strip()}")

if __name__ == "__main__":
    # Create a backup of the source script first
    backup_path = f"{SOURCE_SCRIPT}.bak"
    shutil.copy2(SOURCE_SCRIPT, backup_path)
    print(f"Backup created at {backup_path}")
    
    # Fix the script
    fix_script()
EOF

# Upload the Python script to the server
chmod +x python_file_editor.py
scp -i $KEY_PATH python_file_editor.py $SERVER:/tmp/

# Execute the Python script on the server
echo "Running Python fix script on server..."
ssh -i $KEY_PATH $SERVER "chmod +x /tmp/python_file_editor.py && python3 /tmp/python_file_editor.py"

# Stop any existing processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*wav2vec.*\.py' || true"

# Start training with the new fixed script
echo "Starting training with the Python-fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_final_fixed_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &"

# Create monitoring script
cat > monitor_final_fix.sh << EOF
#!/bin/bash
# Script to monitor the final fixed version

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
LOG_FILE="$LOG_FILE"

echo "Finding the most recent training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_final_fixed_*.log | head -1"
LATEST_LOG=\$(ssh -i \$KEY_PATH \$SERVER "\$SSH_CMD")
echo "Using log file: \$LATEST_LOG"
echo "==============================================================="

echo "Checking if training process is running..."
PROCESS_COUNT=\$(ssh -i \$KEY_PATH \$SERVER "ps aux | grep 'python.*final_wav2vec_fix.py' | grep -v grep | wc -l")
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

chmod +x monitor_final_fix.sh

# Create a script for downloading the model
cat > download_final_model.sh << EOF
#!/bin/bash
# Script to download the model trained with the final fixed solution

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

chmod +x download_final_model.sh

# Clean up
rm python_file_editor.py

echo "===== Final Python Fix Applied and Training Restarted ====="
echo "A Python script has directly fixed the comma issues in the Wav2Vec script."
echo "Training has been restarted with the completely fixed script."
echo "To monitor the training progress, run: ./monitor_final_fix.sh"
echo "To download the model when training is complete, run: ./download_final_model.sh"
