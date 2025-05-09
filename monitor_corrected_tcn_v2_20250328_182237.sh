#!/bin/bash
# Monitoring script for the corrected TCN v2 model training
# Generated specifically for run started around: 20250328_182237

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

INSTANCE_IP="13.217.128.73"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed_v2_20250328_182237.log" # Specific log file
REMOTE_DIR="/home/ec2-user/emotion_training"
PID_FILE="fixed_tcn_large_v2_20250328_182237_pid.txt" # Specific PID file
MONITORING_INTERVAL=30

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}    MONITORING CORRECTED TCN V2 MODEL TRAINING (20250328_182237)    ${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo -e "${YELLOW}Instance:${NC} $USERNAME@$INSTANCE_IP"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo -e "${YELLOW}PID file:${NC} $PID_FILE"
echo -e "${BLUE}==================================================================${NC}"

# Start continuous monitoring
echo -e "${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.${NC}"
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
    # Check if training is active
    if [ -f \"${REMOTE_DIR}/${PID_FILE}\" ]; then
        PID=\$(cat \"${REMOTE_DIR}/${PID_FILE}\")
        # Check if PID is numeric and process exists
        if [[ \"\$PID\" =~ ^[0-9]+$ ]] && ps -p \"\$PID\" > /dev/null; then
            echo -e '\033[0;32mTraining process (PID: \$PID) is active\033[0m'
        else
            echo -e '\033[0;31mTraining process (PID: \$PID) is no longer running (or PID file invalid)\033[0m'
            # Check exit status if possible (might require more complex log parsing)
            if grep -q 'Training finished' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;32mTraining appears to have completed normally.\033[0m'
            elif grep -q 'Error:' \"${REMOTE_DIR}/${LOG_FILE}\" || grep -q 'Traceback' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'
            fi
        fi
    else
        echo -e '\033[0;31mPID file not found. Training might have finished or failed to start.\033[0m'
        # Check log file for completion or errors
        if [ -f \"${REMOTE_DIR}/${LOG_FILE}\" ]; then
             if grep -q 'Training finished' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;32mLog file indicates training completed normally.\033[0m'
             elif grep -q 'Error:' \"${REMOTE_DIR}/${LOG_FILE}\" || grep -q 'Traceback' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'
             else
                 echo -e '\033[1;33mLog file exists but completion/error status unclear.\033[0m'
             fi
        else
             echo -e '\033[0;31mLog file not found either.\033[0m'
        fi
    fi

    # Simplified check for best accuracy within the log file itself
    BEST_ACC_LINE=\$(grep 'val_accuracy improved from' \"${REMOTE_DIR}/${LOG_FILE}\" | tail -n 1)
    if [ -n \"\$BEST_ACC_LINE\" ]; then
        echo -e '\033[1;33mLast recorded best validation accuracy improvement:\033[0m'
        echo \"\$BEST_ACC_LINE\"
    else
         BEST_ACC_LINE_INIT=\$(grep 'Epoch 00001: val_accuracy improved from' \"${REMOTE_DIR}/${LOG_FILE}\" | head -n 1)
         if [ -n \"\$BEST_ACC_LINE_INIT\" ]; then
            echo -e '\033[1;33mInitial validation accuracy recorded:\033[0m'
            echo \"\$BEST_ACC_LINE_INIT\"
         else
            echo -e '\033[1;33mBest validation accuracy not found in log yet.\033[0m'
         fi
    fi

    # Now follow the log file
    echo -e '\033[0;34m===================================================================\033[0m'
    echo -e '\033[0;32mFollowing log file output (tail -f):\033[0m'
    echo -e '\033[0;34m===================================================================\033[0m'
    tail -f \"${REMOTE_DIR}/${LOG_FILE}\"
"
