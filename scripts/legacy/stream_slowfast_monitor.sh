#!/bin/bash

# STREAM SlowFast Training Monitor
# A non-interactive version that streams the training output in real-time
# without attaching to the tmux session

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

KEY_PATH="${KEY_PATH:-$HOME/Downloads/gpu-key.pem}"
EC2_HOST="${EC2_HOST:-ubuntu@54.162.134.77}"

echo -e "${BLUE}R3D-18 EMOTION RECOGNITION STREAM MONITOR${NC}"
echo "============================================================="
echo
echo -e "${YELLOW}Connecting to EC2 instance...${NC}"
echo -e "${GREEN}Streaming training output in real-time${NC}"
echo -e "${CYAN}Press Ctrl+C to stop monitoring (training will continue)${NC}"
echo

# This version captures the current pane content and then sets up a continuous pipe
# It's non-interactive but shows real-time output
ssh -i "$KEY_PATH" "$EC2_HOST" \
  "tmux capture-pane -pt slowfast_training -S -1000 -J; \
   echo '${YELLOW}[Previous output above - Real-time streaming below]${NC}'; \
   echo '${YELLOW}==========================================================${NC}'; \
   tmux pipe-pane -t slowfast_training -o 'cat'; \
   tail -f /home/ubuntu/monitor_logs/slowfast_training_stream.log"
