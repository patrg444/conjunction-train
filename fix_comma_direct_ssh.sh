#!/bin/bash
# This script directly edits the file on the server using SSH and sed
# It's a targeted approach focusing on the exact lines that need fixes

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
SCRIPT_PATH="/home/ubuntu/audio_emotion/fixed_v6_script_final.py"

echo "Connecting to server to find and fix the missing commas..."

# First create a backup of the file just in case
ssh -i $KEY_PATH $SERVER "cp $SCRIPT_PATH ${SCRIPT_PATH}.bak"

# Use grep to find the exact line numbers where the commas are missing
echo "Searching for problematic lines..."
LINE_NUMBERS=$(ssh -i $KEY_PATH $SERVER "grep -n 'set_value.*learning_rate ' $SCRIPT_PATH | cut -d ':' -f1")

if [ -z "$LINE_NUMBERS" ]; then
    echo "No problematic lines found. The script might already be fixed or has a different syntax."
    exit 1
fi

echo "Found problematic lines at: $LINE_NUMBERS"

# Use sed to add commas after 'learning_rate' on each line
for LINE in $LINE_NUMBERS; do
    echo "Fixing line $LINE"
    ssh -i $KEY_PATH $SERVER "sed -i '${LINE}s/learning_rate /learning_rate, /g' $SCRIPT_PATH"
done

# Verify the fix
echo "Verifying the fix..."
ssh -i $KEY_PATH $SERVER "grep -n 'set_value.*learning_rate,' $SCRIPT_PATH"

# Kill any existing training processes
echo "Stopping any existing training processes..."
ssh -i $KEY_PATH $SERVER "pkill -f 'python.*fixed_v6_script_final.py' || true"

# Start the training with the fixed script
echo "Restarting training with fixed script..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/audio_emotion/wav2vec_direct_ssh_fix_${TIMESTAMP}.log"
ssh -i $KEY_PATH $SERVER "cd /home/ubuntu/audio_emotion && nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &"

echo "Comma fix applied and training restarted!"
echo "Training logs will be written to: $LOG_FILE"
echo "You can monitor the training with: ./monitor_direct_fix.sh"
