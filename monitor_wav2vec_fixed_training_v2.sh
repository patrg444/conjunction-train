#!/bin/bash
# Enhanced monitoring script for wav2vec fixed training (v2)
# This script provides better visibility into training progress and numerical issues
# Updated to monitor the v2 version with the optimizer fix

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo -e "${BLUE}Enhanced Wav2Vec Training Monitor (v2)${NC}"
echo "================================================"

# Ensure the gpu key has the right permissions
chmod 400 ~/Downloads/gpu-key.pem

# Function to check if training is running
check_training_status() {
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "ps aux | grep train_wav2vec_audio_only_fixed_v2 | grep -v grep" > /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[✓] Training is currently RUNNING${NC}"
    else
        echo -e "${RED}[✗] Training process NOT FOUND${NC}"
        echo -e "  - Check for error messages in log file"
        echo -e "  - The process may have completed or crashed"
    fi
}

# Function to show GPU stats
show_gpu_stats() {
    echo -e "\n${BLUE}GPU Stats:${NC}"
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader"
}

# Function to analyze log file for common issues
analyze_log_file() {
    local LOG_FILE=$1
    echo -e "\n${BLUE}Log Analysis:${NC}"
    
    # Check for NaN values
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "grep -i 'nan' $LOG_FILE | tail -5" > /tmp/nan_check
    if [ -s /tmp/nan_check ]; then
        echo -e "${RED}[!] NaN values detected in training:${NC}"
        cat /tmp/nan_check
    else
        echo -e "${GREEN}[✓] No NaN values detected${NC}"
    fi
    
    # Check for numeric errors
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "grep -i 'error\|exception\|fail' $LOG_FILE | tail -5" > /tmp/error_check
    if [ -s /tmp/error_check ]; then
        echo -e "${RED}[!] Errors detected in training:${NC}"
        cat /tmp/error_check
    fi
    
    # Get recent loss values to look for trends
    echo -e "\n${BLUE}Recent loss values:${NC}"
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "grep -i 'loss:' $LOG_FILE | tail -10"
    
    # Get recent accuracy values
    echo -e "\n${BLUE}Recent accuracy values:${NC}"
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "grep -i 'accuracy:' $LOG_FILE | tail -10"
    
    # Get current learning rate
    echo -e "\n${BLUE}Current learning rate:${NC}"
    ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "grep -i 'Learning rate set to' $LOG_FILE | tail -1"
}

# Find the log file (most recent one matching the pattern)
LOG_FILE=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "ls -t /home/ubuntu/audio_emotion/wav2vec_fixed_training_v2_*.log 2>/dev/null | head -1")

if [ -z "$LOG_FILE" ]; then
    echo -e "${RED}No training log file found.${NC}"
    echo "Has the training been started? Check if deploy_fixed_wav2vec_training_v2.sh was run."
    exit 1
fi

echo -e "Monitoring log file: ${YELLOW}$LOG_FILE${NC}"

# Check if training is running
check_training_status

# Show GPU stats
show_gpu_stats

# Analyze log file
analyze_log_file "$LOG_FILE"

# Get training duration
START_TIME=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "head -20 $LOG_FILE | grep 'Starting' | head -1")
if [ ! -z "$START_TIME" ]; then
    echo -e "\n${BLUE}Training started at:${NC} $START_TIME"
    CURRENT_TIME=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "date")
    echo -e "${BLUE}Current time:${NC} $CURRENT_TIME"
fi

# Show TensorBoard status
TB_RUNNING=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "ps aux | grep tensorboard | grep -v grep" > /dev/null && echo "yes" || echo "no")
if [ "$TB_RUNNING" == "yes" ]; then
    echo -e "\n${GREEN}[✓] TensorBoard is running${NC}"
    echo -e "To view TensorBoard, run:"
    echo -e "${YELLOW}ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77${NC}"
    echo -e "Then open http://localhost:6006 in your browser"
else
    echo -e "\n${RED}[✗] TensorBoard is not running${NC}"
    echo -e "To start TensorBoard, run:"
    echo -e "${YELLOW}ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \"cd /home/ubuntu/audio_emotion && nohup tensorboard --logdir=logs --port=6006 --host=0.0.0.0 > tensorboard.log 2>&1 &\"${NC}"
fi

echo -e "\n${BLUE}To tail the log file continuously, run:${NC}"
echo -e "${YELLOW}ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \"tail -f $LOG_FILE\"${NC}"

echo -e "\n${GREEN}Monitoring complete${NC}"
