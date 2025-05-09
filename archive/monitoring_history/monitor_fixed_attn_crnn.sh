#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Monitoring ATTN-CRNN training on EC2 GPU instance...${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check GPU usage
if [ "$1" = "-g" ] || [ "$1" = "--gpu" ]; then
    echo -e "${YELLOW}Checking GPU status...${NC}"
    ssh -i "$PEM" ubuntu@$IP "nvidia-smi"
    exit 0
fi

# Check if process is running
if [ "$1" = "-p" ] || [ "$1" = "--process" ]; then
    echo -e "${YELLOW}Checking if training process is running...${NC}"
    ssh -i "$PEM" ubuntu@$IP "ps aux | grep fixed_attn_crnn.py | grep -v grep && echo -e '${GREEN}Process is RUNNING${NC}' || echo -e '${YELLOW}Process NOT found${NC}'"
    exit 0
fi

# Tail logs
if [ "$1" = "-l" ] || [ "$1" = "--logs" ]; then
    echo -e "${YELLOW}Fetching latest logs...${NC}"
    ssh -i "$PEM" ubuntu@$IP "tail -n 100 ~/emotion_project/train_log.txt || echo 'No log file found'"
    exit 0
fi

# Check how many files were processed
if [ "$1" = "-c" ] || [ "$1" = "--count" ]; then
    echo -e "${YELLOW}Counting processed files...${NC}"
    ssh -i "$PEM" ubuntu@$IP "cd ~/emotion_project && tmux capture-pane -p -t attn_train | grep -i 'loaded' | tail -n 10 || echo 'No output found'"
    exit 0
fi

# Check training progress directly from memory
if [ "$1" = "-m" ] || [ "$1" = "--memory" ]; then
    echo -e "${YELLOW}Capturing full tmux output...${NC}"
    ssh -i "$PEM" ubuntu@$IP "tmux capture-pane -p -t attn_train"
    exit 0
fi

# Download model (when training is completed)
if [ "$1" = "-d" ] || [ "$1" = "--download" ]; then
    echo -e "${YELLOW}Downloading trained model...${NC}"
    mkdir -p ./models
    scp -i "$PEM" ubuntu@$IP:~/emotion_project/models/attn_crnn_model.h5 ./models/
    scp -i "$PEM" ubuntu@$IP:~/emotion_project/models/class_map.npy ./models/
    echo -e "${GREEN}Model downloaded to ./models/attn_crnn_model.h5${NC}"
    exit 0
fi

# Default: Show a summary of everything
echo -e "${YELLOW}Checking if training process is running...${NC}"
ssh -i "$PEM" ubuntu@$IP "ps aux | grep fixed_attn_crnn.py | grep -v grep && echo -e '${GREEN}Process is RUNNING${NC}' || echo -e '${YELLOW}Process NOT found${NC}'"

echo -e "${YELLOW}\nChecking GPU status...${NC}"
ssh -i "$PEM" ubuntu@$IP "nvidia-smi | grep -A 3 'GPU'"

echo -e "${YELLOW}\nLatest log entries (last 10 lines):${NC}"
ssh -i "$PEM" ubuntu@$IP "tail -n 10 ~/emotion_project/train_log.txt || echo 'No log file found'"

echo -e "${BLUE}\n=========================================${NC}"
echo -e "For more details, use:"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -g${NC}  # GPU status"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -p${NC}  # Process check"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -l${NC}  # View logs"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -m${NC}  # Check tmux output"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -c${NC}  # Count processed files"
echo -e "  ${GREEN}./monitor_fixed_attn_crnn.sh -d${NC}  # Download model"
echo -e "${BLUE}=========================================${NC}"
