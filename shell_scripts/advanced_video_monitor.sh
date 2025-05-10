#!/bin/bash

# Advanced Video Training Monitor
# Provides continuous real-time monitoring with better formatting and visuals

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Training output directory
OUTPUT_DIR="/home/ubuntu/emotion_full_video"

# Find the most recent training session
LATEST_DIR=$(find ${OUTPUT_DIR} -maxdepth 1 -type d -name "video_full_*" | sort -r | head -n 1)
if [ -z "$LATEST_DIR" ]; then
    echo -e "${RED}No training session found in ${OUTPUT_DIR}${NC}"
    exit 1
fi

# Clear screen and print header
clear
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}             ADVANCED VIDEO TRAINING MONITOR               ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${CYAN}Monitoring session in: ${YELLOW}${LATEST_DIR}${NC}\n"

# Get training start time from directory name
SESSION_NAME=$(basename "$LATEST_DIR")
START_TIME=$(echo "$SESSION_NAME" | cut -d'_' -f3-4)
echo -e "${CYAN}Training started: ${YELLOW}${START_TIME}${NC}\n"

# Function to show a progress bar
show_progress_bar() {
    local percent=$1
    local width=50
    local num_filled=$(($width * $percent / 100))
    local num_empty=$(($width - $num_filled))
    local bar=""

    # Build the progress bar
    for ((i=0; i<$num_filled; i++)); do
        bar="${bar}█"
    done
    for ((i=0; i<$num_empty; i++)); do
        bar="${bar} "
    done

    echo -e "${BLUE}[${bar}] ${percent}%${NC}"
}

# Function to extract current training progress
extract_progress() {
    local log_content=$1
    local epoch_line=$(echo "$log_content" | grep -o "Epoch [0-9]*/30" | tail -1)
    local progress_line=$(echo "$log_content" | grep -o "Training:.*%" | tail -1)
    
    if [[ -n "$epoch_line" ]]; then
        echo -e "${CYAN}$epoch_line${NC}"
    fi
    
    if [[ -n "$progress_line" ]]; then
        local percent=$(echo "$progress_line" | grep -o "[0-9]*%" | sed 's/%//')
        local stats=$(echo "$progress_line" | grep -o "loss=.*acc=.*\]" | sed 's/\]//g')
        
        if [[ -n "$percent" ]]; then
            show_progress_bar "$percent"
        fi
        
        if [[ -n "$stats" ]]; then
            echo -e "${YELLOW}$stats${NC}"
        fi
    fi
}

# Function to extract validation metrics
extract_validation() {
    local log_content=$1
    local val_line=$(echo "$log_content" | grep "Val Loss:" | tail -1)
    
    if [[ -n "$val_line" ]]; then
        echo -e "\n${PURPLE}${BOLD}Validation Results:${NC}"
        echo -e "${PURPLE}$val_line${NC}"
        
        # Extract best validation line if present
        local best_line=$(echo "$log_content" | grep "best validation accuracy" | tail -1)
        if [[ -n "$best_line" ]]; then
            echo -e "${GREEN}$best_line${NC}"
        fi
    fi
}

# Create a temporary file for GPU stats
GPU_STATS_FILE=$(mktemp)

# Function to update GPU stats in the background
update_gpu_stats() {
    while true; do
        # Get GPU stats from nvidia-smi
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader > "$GPU_STATS_FILE"
        sleep 2
    done
}

# Start GPU monitoring in background
update_gpu_stats &
GPU_MONITOR_PID=$!

# Trap to kill the GPU monitoring process when the script is terminated
trap "kill $GPU_MONITOR_PID 2>/dev/null; rm -f $GPU_STATS_FILE; echo -e '\n${GREEN}Monitoring stopped.${NC}'; exit" INT TERM EXIT

# Main monitoring loop
while true; do
    # Clear screen each iteration
    clear
    
    # Print header
    echo -e "${BOLD}${GREEN}===========================================================${NC}"
    echo -e "${BOLD}${GREEN}             ADVANCED VIDEO TRAINING MONITOR               ${NC}"
    echo -e "${BOLD}${GREEN}===========================================================${NC}"
    echo -e "${CYAN}Monitoring session in: ${YELLOW}${LATEST_DIR}${NC}"
    echo -e "${CYAN}Training started: ${YELLOW}${START_TIME}${NC}\n"
    
    # Show current time
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${CYAN}Current Time: ${YELLOW}$current_time${NC}\n"
    
    # Display GPU stats
    if [ -f "$GPU_STATS_FILE" ]; then
        gpu_data=$(cat "$GPU_STATS_FILE")
        gpu_util=$(echo "$gpu_data" | awk -F', ' '{print $1}')
        gpu_mem_util=$(echo "$gpu_data" | awk -F', ' '{print $2}')
        gpu_mem_used=$(echo "$gpu_data" | awk -F', ' '{print $3}')
        gpu_mem_total=$(echo "$gpu_data" | awk -F', ' '{print $4}')
        
        echo -e "${BOLD}${BLUE}GPU Statistics:${NC}"
        echo -e "${BLUE}GPU Utilization: ${YELLOW}$gpu_util${NC}"
        echo -e "${BLUE}Memory Utilization: ${YELLOW}$gpu_mem_util${NC}"
        echo -e "${BLUE}Memory Used/Total: ${YELLOW}$gpu_mem_used / $gpu_mem_total${NC}\n"
    fi
    
    # Get the tmux output
    tmux_output=$(tmux capture-pane -pt video_training -S - | grep -v "^$")
    
    # Extract and display the current training progress
    echo -e "${BOLD}${CYAN}Training Progress:${NC}"
    extract_progress "$tmux_output"
    
    # Extract and display validation metrics if available
    extract_validation "$tmux_output"
    
    # Display any warnings or errors
    warnings=$(echo "$tmux_output" | grep -i "warning\|error" | tail -3)
    if [[ -n "$warnings" ]]; then
        echo -e "\n${YELLOW}Recent Warnings:${NC}"
        echo -e "${YELLOW}$warnings${NC}"
    fi
    
    echo -e "\n${GREEN}Press Ctrl+C to stop monitoring${NC}"
    
    # Sleep before the next update
    sleep 5
done
