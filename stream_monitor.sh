#!/bin/bash

# Stream Monitor - Provides continuous real-time streaming of training logs
# This script connects to the training session and maintains a persistent connection

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Clear screen and display header
clear
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}          CONTINUOUS STREAMING TRAINING MONITOR            ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${CYAN}Connecting to continuous training stream...${NC}\n"

# Check if tmux session exists
if ! tmux has-session -t video_training 2>/dev/null; then
    echo -e "${RED}Error: Training session 'video_training' not found.${NC}"
    echo -e "${YELLOW}Make sure your training script is running in a tmux session named 'video_training'.${NC}"
    exit 1
fi

# First, create a monitoring window that runs the GPU stats in the background
echo -e "${CYAN}Setting up GPU monitoring...${NC}"

# Create a new tmux session for monitoring
tmux new-session -d -s monitor_session

# Send GPU monitoring command to the session
tmux send-keys -t monitor_session "while true; do clear; echo -e '${BOLD}${GREEN}GPU Statistics:${NC}'; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv; echo; echo -e '${BOLD}${GREEN}Training Process:${NC}'; ps -o pid,%cpu,%mem,cmd -p \$(pgrep -f train_video_full.py) | grep -v grep; echo; sleep 5; done" C-m

# Split the window vertically - top will be GPU stats, bottom will be training output
tmux split-window -v -t monitor_session

# Show the user we're about to start continuous monitoring
echo -e "${YELLOW}Attaching to training output stream...${NC}"
echo -e "${YELLOW}Press Ctrl+C to detach from the stream${NC}\n"
sleep 2

# Send command to attach to the video_training session in read-only mode to the bottom pane
tmux send-keys -t monitor_session:0.1 "tmux attach-session -t video_training -r" C-m

# Attach to our monitoring session
echo -e "${GREEN}Connected! Displaying continuous training output:${NC}\n"
tmux attach-session -t monitor_session

# When the user detaches (Ctrl+C), clean up the monitoring session
tmux kill-session -t monitor_session 2>/dev/null

echo -e "\n${GREEN}Monitoring stopped.${NC}"
