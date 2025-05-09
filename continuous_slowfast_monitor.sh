#!/bin/bash

# Continuous streaming monitor for SlowFast training
# This script provides a direct stream of training logs without interruption

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

KEY_PATH="${KEY_PATH:-$HOME/Downloads/gpu-key.pem}"
EC2_HOST="${EC2_HOST:-ubuntu@54.162.134.77}"

echo -e "${BLUE}R3D-18 EMOTION RECOGNITION CONTINUOUS MONITOR${NC}"
echo "============================================================="
echo

echo -e "${YELLOW}Connecting to EC2 instance...${NC}"
echo -e "${GREEN}Streaming continuous training logs:${NC}"
echo

# Use direct streaming with stdbuf to avoid buffering
ssh -i "$KEY_PATH" "$EC2_HOST" "stdbuf -oL -eL cat /home/ubuntu/monitor_logs/slowfast_training_stream.log"
