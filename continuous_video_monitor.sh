#!/bin/bash

# Continuous monitoring script for video training
# Provides real-time updates on training progress and GPU status

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Training output directory where logs are stored
OUTPUT_DIR="/home/ubuntu/emotion_full_video"

# Find the most recent training session
LATEST_DIR=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "video_full_*" | sort -r | head -n 1)
if [ -z "$LATEST_DIR" ]; then
    echo -e "${RED}No training session found in ${OUTPUT_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}==== Continuous Video Training Monitor ====${NC}"
echo -e "${BLUE}Monitoring latest training session in: ${LATEST_DIR}${NC}"

# Find the latest log file
LOG_FILE="${LATEST_DIR}/training.log"

# If log file doesn't exist, create a temp file to redirect stdout/stderr from tmux
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE="/tmp/video_training_output.log"
    # Capture tmux output to the temp file
    tmux capture-pane -pt video_training -S - | grep -v "^$" > "$LOG_FILE" 2>/dev/null
    echo -e "${YELLOW}No dedicated log file found. Using tmux output capture.${NC}"
fi

# Function to display GPU stats
show_gpu_stats() {
    while true; do
        # Get GPU stats using nvidia-smi
        local GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader)
        
        # Clear the line and print the GPU stats
        echo -ne "\r\033[K${BLUE}GPU: ${GPU_STATS}${NC}"
        
        # Sleep for 5 seconds
        sleep 5
    done
}

# Start GPU monitoring in the background
show_gpu_stats &
GPU_MONITOR_PID=$!

# Trap to kill the GPU monitoring background process when the script exits
trap "kill $GPU_MONITOR_PID 2>/dev/null; echo -e '\n${GREEN}Monitoring stopped.${NC}'; exit" INT TERM EXIT

echo -e "\n${GREEN}Training progress (press Ctrl+C to stop monitoring):${NC}\n"

# Use tail to continuously monitor the log file
# Main training progress comes from the training script's output
if [ -f "$LOG_FILE" ]; then
    tail -n 20 -f "$LOG_FILE"
else
    # If the log file doesn't exist, continuously capture tmux output as a fallback
    while true; do
        tmux capture-pane -pt video_training -S - | grep -v "^$" | tail -n 20
        sleep 2
    done
fi
