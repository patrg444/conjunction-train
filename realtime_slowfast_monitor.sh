#!/bin/bash

# REALTIME SlowFast Training Monitor
# This script directly attaches to the tmux session running the training process
# for true real-time monitoring without any buffering or delays

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

KEY_PATH="${KEY_PATH:-$HOME/Downloads/gpu-key.pem}"
EC2_HOST="${EC2_HOST:-ubuntu@54.162.134.77}"

echo -e "${BLUE}R3D-18 EMOTION RECOGNITION REALTIME MONITOR${NC}"
echo "============================================================="
echo
echo -e "${YELLOW}Connecting to EC2 instance...${NC}"
echo -e "${GREEN}Attaching directly to training session${NC}"
echo -e "${RED}Press Ctrl+B then D to detach without stopping training${NC}"
echo

# Directly attach to the tmux session on the remote server
# This gives a true real-time view of the training output
ssh -t -i "$KEY_PATH" "$EC2_HOST" "tmux attach -t slowfast_training"
