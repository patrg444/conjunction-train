#!/bin/bash

# --- Configuration ---
INSTANCE_IP="13.217.128.73" # Replace with your EC2 instance IP if different
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem" # Path to your PEM key file
LOCAL_SCRIPT_PATH="scripts/train_lstm_conv1d_cross_attention.py" # Path to the NEW training script
REMOTE_SCRIPT_NAME="train_lstm_conv1d_cross_attention.py" # Name for the script on the EC2 instance
REMOTE_DIR="/home/ec2-user/emotion_training" # Directory on EC2 for training files
LOG_PREFIX="training_lstm_conv1d_cross_attention" # Prefix for log files
PID_PREFIX="lstm_conv1d_cross_attention" # Prefix for PID files
# --- End Configuration ---

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    DEPLOYING LSTM/CONV1D + CROSS-ATTENTION MODEL SCRIPT    ${NC}" # Updated Title
echo -e "${BLUE}===================================================================${NC}"

# Check if local script exists
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Local training script not found at $LOCAL_SCRIPT_PATH${NC}"
    exit 1
fi

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: PEM key file not found at $KEY_FILE${NC}"
    exit 1
fi

# Create remote directory if it doesn't exist
echo -e "${YELLOW}Ensuring remote directory exists ($REMOTE_DIR)...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "mkdir -p $REMOTE_DIR"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create or access remote directory $REMOTE_DIR${NC}"
    exit 1
fi
echo "Remote directory check complete."

# Transfer the training script using SCP
echo -e "${YELLOW}Transferring LSTM/Conv1D + Cross-Attention script ($LOCAL_SCRIPT_PATH) to AWS instance...${NC}" # Updated description
scp -i "$KEY_FILE" "$LOCAL_SCRIPT_PATH" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Training script transfer failed.${NC}"
    exit 1
fi
echo -e "${GREEN}Training script transferred successfully!${NC}"

# Transfer the generator scripts using SCP (Needed for this model)
LOCAL_SYNC_GENERATOR_PATH="scripts/synchronized_data_generator.py"
REMOTE_SYNC_GENERATOR_NAME="synchronized_data_generator.py"
echo -e "${YELLOW}Transferring Synchronized Data Generator script ($LOCAL_SYNC_GENERATOR_PATH) to AWS instance...${NC}"
scp -i "$KEY_FILE" "$LOCAL_SYNC_GENERATOR_PATH" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_SYNC_GENERATOR_NAME"
if [ $? -ne 0 ]; then echo -e "${RED}Error: Synchronized generator script transfer failed.${NC}"; exit 1; fi
echo -e "${GREEN}Synchronized generator script transferred successfully!${NC}"

LOCAL_SEQ_GENERATOR_PATH="scripts/sequence_data_generator.py"
REMOTE_SEQ_GENERATOR_NAME="sequence_data_generator.py"
echo -e "${YELLOW}Transferring Sequence Data Generator script ($LOCAL_SEQ_GENERATOR_PATH) to AWS instance...${NC}"
scp -i "$KEY_FILE" "$LOCAL_SEQ_GENERATOR_PATH" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_SEQ_GENERATOR_NAME"
if [ $? -ne 0 ]; then echo -e "${RED}Error: Sequence generator script transfer failed.${NC}"; exit 1; fi
echo -e "${GREEN}Sequence generator script transferred successfully!${NC}"

# Set execute permissions on the remote training script
echo -e "${YELLOW}Setting execute permissions on the training script...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x $REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to set execute permissions on remote script.${NC}"
    # Attempting to continue, Python might still execute it
fi

# Check Python syntax on the remote script
echo -e "${YELLOW}Checking script for Python syntax errors...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "python3 -m py_compile $REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Python syntax check failed for $REMOTE_SCRIPT_NAME on the EC2 instance.${NC}"
    echo -e "${YELLOW}Please check the script for errors before launching.${NC}"
    exit 1
else
    echo -e "${GREEN}Syntax check passed! Script is valid Python.${NC}"
fi

# Create a timestamp for log and PID files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REMOTE_LOG_FILE="${LOG_PREFIX}_${TIMESTAMP}.log"
REMOTE_PID_FILE="${PID_PREFIX}_${TIMESTAMP}_pid.txt"

# Create a launch script on the remote instance
REMOTE_LAUNCH_SCRIPT="launch_${PID_PREFIX}.sh" # Use PID_PREFIX in launch script name
LAUNCH_SCRIPT_CONTENT="#!/bin/bash
cd $REMOTE_DIR || exit 1 # Change to the training directory
echo \"Starting LSTM/Conv1D + Cross-Attention model training...\" > $REMOTE_LOG_FILE # Updated description
echo \"PID will be written to: $REMOTE_PID_FILE\" >> $REMOTE_LOG_FILE
nohup python3 -u $REMOTE_SCRIPT_NAME >> $REMOTE_LOG_FILE 2>&1 &
PID=\$!
echo \$PID > $REMOTE_PID_FILE
echo \"Training process started with PID: \$PID\" >> $REMOTE_LOG_FILE
echo \"Logs are being written to: $REMOTE_LOG_FILE\"
# Optionally, print Python version
{ python3 --version >> $REMOTE_LOG_FILE 2>&1 || echo 'Failed to get Python version' >> $REMOTE_LOG_FILE; }

"

echo -e "${YELLOW}Creating launch script (~/$REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT) on AWS instance...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "echo '$LAUNCH_SCRIPT_CONTENT' > $REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT && chmod +x $REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create launch script on remote instance.${NC}"
    exit 1
fi
echo -e "${GREEN}Launch script created successfully!${NC}"

# Launch the training process using the launch script
echo -e "${YELLOW}Launching training process on AWS instance...${NC}"
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd $REMOTE_DIR && ./$REMOTE_LAUNCH_SCRIPT"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to execute launch script on remote instance.${NC}"
    exit 1
fi
echo -e "${GREEN}Training launch command executed.${NC}"
echo -e "${YELLOW}Please monitor the training progress using a specific monitoring script or by checking the logs on the EC2 instance.${NC}"
echo -e "${YELLOW}The specific log and PID files will have a timestamp like YYYYMMDD_HHMMSS and be named like '${LOG_PREFIX}_...' and '${PID_PREFIX}_...'.${NC}"


# Define content for a remote helper script that does the actual monitoring
REMOTE_MONITOR_HELPER_NAME="monitor_helper_${PID_PREFIX}_${TIMESTAMP}.sh" # Use PID_PREFIX
REMOTE_MONITOR_HELPER_CONTENT="#!/bin/bash
# This script runs on the EC2 instance to check status and tail logs

REMOTE_DIR=\"$REMOTE_DIR\"
LOG_FILE=\"$REMOTE_LOG_FILE\"
PID_FILE=\"$REMOTE_PID_FILE\"

# Check if training is active
if [ -f \"\${REMOTE_DIR}/\${PID_FILE}\" ]; then
    PID=\$(cat \"\${REMOTE_DIR}/\${PID_FILE}\")
    if [[ \"\$PID\" =~ ^[0-9]+\$ ]] && ps -p \"\$PID\" > /dev/null; then
        echo -e '\033[0;32mTraining process (PID: \$PID) is active\033[0m'
    else
        echo -e '\033[0;31mTraining process (PID: \$PID) is no longer running (or PID file invalid)\033[0m'
        if grep -q 'Training finished' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;32mTraining appears to have completed normally.\033[0m';
        elif grep -q 'Error:' \"\${REMOTE_DIR}/\${LOG_FILE}\" || grep -q 'Traceback' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'; fi
    fi
else
    echo -e '\033[0;31mPID file not found. Training might have finished or failed to start.\033[0m'
    if [ -f \"\${REMOTE_DIR}/\${LOG_FILE}\" ]; then
         if grep -q 'Training finished' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;32mLog file indicates training completed normally.\033[0m';
         elif grep -q 'Error:' \"\${REMOTE_DIR}/\${LOG_FILE}\" || grep -q 'Traceback' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m';
         else echo -e '\033[1;33mLog file exists but completion/error status unclear.\033[0m'; fi
    else echo -e '\033[0;31mLog file not found either.\033[0m'; fi
fi

# Check best accuracy
BEST_ACC_LINE=\$(grep 'val_accuracy improved from' \"\${REMOTE_DIR}/\${LOG_FILE}\" | tail -n 1)
if [ -n \"\$BEST_ACC_LINE\" ]; then
    echo -e '\033[1;33mLast recorded best validation accuracy improvement:\033[0m'; echo \"\$BEST_ACC_LINE\"
else
     BEST_ACC_LINE_INIT=\$(grep 'Epoch 00001: val_accuracy improved from' \"\${REMOTE_DIR}/\${LOG_FILE}\" | head -n 1)
     if [ -n \"\$BEST_ACC_LINE_INIT\" ]; then echo -e '\033[1;33mInitial validation accuracy recorded:\033[0m'; echo \"\$BEST_ACC_LINE_INIT\"
     else echo -e '\033[1;33mBest validation accuracy not found in log yet.\033[0m'; fi
fi

# Follow log file
echo -e '\033[0;34m===================================================================\033[0m'
echo -e '\033[0;32mFollowing log file output (tail -f):\033[0m'
echo -e '\033[0;34m===================================================================\033[0m'
tail -f \"\${REMOTE_DIR}/\${LOG_FILE}\"
"

# Create the remote helper script via SSH
echo -e "${YELLOW}Creating remote monitoring helper script ($REMOTE_MONITOR_HELPER_NAME)...${NC}"
# Use printf to handle potential special characters in the content safely
printf "%s" "$REMOTE_MONITOR_HELPER_CONTENT" | ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > $REMOTE_DIR/$REMOTE_MONITOR_HELPER_NAME && chmod +x $REMOTE_DIR/$REMOTE_MONITOR_HELPER_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create remote monitoring helper script.${NC}"
    exit 1
fi
echo -e "${GREEN}Remote monitoring helper script created successfully.${NC}"


# Generate a specific local monitoring script that calls the remote helper
MONITOR_SCRIPT_NAME="monitor_${PID_PREFIX}_${TIMESTAMP}.sh" # Use PID_PREFIX
MONITOR_SCRIPT_CONTENT="#!/bin/bash
# Monitoring script for LSTM/Conv1D + Cross-Attention model training # Updated description
# Generated specifically for run started around: ${TIMESTAMP}

# ANSI colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

INSTANCE_IP=\"$INSTANCE_IP\"
USERNAME=\"$USERNAME\"
KEY_FILE=\"/Users/patrickgloria/conjunction-train/aws-setup/emotion-recognition-key-fixed-20250323090016.pem\" # Use absolute path
REMOTE_DIR=\"$REMOTE_DIR\"
REMOTE_MONITOR_HELPER=\"$REMOTE_MONITOR_HELPER_NAME\" # Name of the helper script on EC2

echo -e \"\${BLUE}==================================================================\${NC}\"
echo -e \"\${GREEN}    MONITORING LSTM/CONV1D + CROSS-ATTENTION MODEL TRAINING (${TIMESTAMP})    \${NC}\" # Updated Title
echo -e \"\${BLUE}==================================================================\${NC}\"
echo -e \"\${YELLOW}Instance:\${NC} \$USERNAME@\$INSTANCE_IP\"
echo -e \"\${YELLOW}Executing remote helper:\${NC} \$REMOTE_DIR/\$REMOTE_MONITOR_HELPER\"
echo -e \"\${BLUE}==================================================================\${NC}\"

# Start continuous monitoring by executing the remote helper script
echo -e \"\${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.\${NC}\"
ssh -t -i \\\"\$KEY_FILE\\\" \\\"\$USERNAME@\$INSTANCE_IP\\\" \"cd \$REMOTE_DIR && ./$REMOTE_MONITOR_HELPER\"

"

echo -e "${YELLOW}Generating local monitoring script: $MONITOR_SCRIPT_NAME...${NC}"
echo "$MONITOR_SCRIPT_CONTENT" > "$MONITOR_SCRIPT_NAME"
chmod +x "$MONITOR_SCRIPT_NAME"
echo -e "${GREEN}Monitoring script '$MONITOR_SCRIPT_NAME' created successfully.${NC}"
echo -e "${YELLOW}You can run './$MONITOR_SCRIPT_NAME' to monitor the training.${NC}"


echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment and launch attempt completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
