#!/bin/bash
# Cross-platform monitoring script for TCN model training
# Generated specifically for model version: v20250327_160534

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed_v2_lr_increase_v20250327_160534.log" # Log file name passed from main script
REMOTE_DIR="/home/ec2-user/emotion_training"
PID_FILE="fixed_tcn_large_v20250327_160534_pid.txt" # Version specific PID file
MONITORING_INTERVAL=30

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}    MONITORING TCN MODEL TRAINING ($MODEL_VERSION)    ${NC}"
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
        PID=$(cat \"${REMOTE_DIR}/${PID_FILE}\")
        # Check if PID is numeric and process exists
        if [[ \"$PID\" =~ ^[0-9]+$ ]] && ps -p \"$PID\" > /dev/null; then
            echo -e '\033[0;32mTraining process (PID: $PID) is active\033[0m'
        else
            echo -e '\033[0;31mTraining process (PID: $PID) is no longer running (or PID file invalid)\033[0m'
            # Check exit status if possible (might require more complex log parsing)
            if grep -q 'Training completed' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;32mTraining appears to have completed normally.\033[0m'
            elif grep -q 'ERROR:' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'
            fi
        fi
    else
        echo -e '\033[0;31mPID file not found. Training might have finished or failed to start.\033[0m'
        # Check log file for completion or errors
        if [ -f \"${REMOTE_DIR}/${LOG_FILE}\" ]; then
             if grep -q 'Training completed' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;32mLog file indicates training completed normally.\033[0m'
             elif grep -q 'ERROR:' \"${REMOTE_DIR}/${LOG_FILE}\"; then
                 echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'
             else
                 echo -e '\033[1;33mLog file exists but completion/error status unclear.\033[0m'
             fi
        else
             echo -e '\033[0;31mLog file not found either.\033[0m'
        fi
    fi

    # Get best validation accuracy (version specific)
    BEST_ACC_FILE=\"best_val_accuracy_${MODEL_VERSION}.txt\"
    if [ -f \"${REMOTE_DIR}/${BEST_ACC_FILE}\" ]; then
        echo -e '\033[1;33mBest validation accuracy recorded:\033[0m'
        cat \"${REMOTE_DIR}/${BEST_ACC_FILE}\"
    else
        echo -e '\033[1;33mBest validation accuracy file (${BEST_ACC_FILE}) not found yet.\033[0m'
    fi

    # Now follow the log file
    echo -e '\033[0;34m===================================================================\033[0m'
    echo -e '\033[0;32mFollowing log file output (tail -f):\033[0m'
    echo -e '\033[0;34m===================================================================\033[0m'
    tail -f \"${REMOTE_DIR}/${LOG_FILE}\"
"
