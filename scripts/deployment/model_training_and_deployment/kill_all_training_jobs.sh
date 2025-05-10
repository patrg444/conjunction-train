#!/bin/bash
# Script to kill all running training processes on the EC2 instance

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo -e "${RED}Terminating All Training Jobs${NC}"
echo "================================================"

# Ensure the gpu key has the right permissions
chmod 400 ~/Downloads/gpu-key.pem

# Function to kill processes by pattern
kill_processes() {
    local pattern=$1
    local description=$2
    
    echo -e "\n${BLUE}Checking for running $description processes...${NC}"
    
    # Get process IDs
    PIDs=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "pgrep -f \"$pattern\"" 2>/dev/null)
    
    if [ -z "$PIDs" ]; then
        echo -e "${YELLOW}No $description processes found.${NC}"
    else
        echo -e "${GREEN}Found $description processes. Terminating...${NC}"
        ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "for pid in $PIDs; do echo \"Killing PID: \$pid\"; sudo kill -9 \$pid; done"
        echo -e "${GREEN}$description processes terminated.${NC}"
    fi
}

# Kill TensorBoard
kill_processes "tensorboard" "TensorBoard"

# Kill wav2vec training
kill_processes "train_wav2vec" "wav2vec training"

# Kill all python processes (in case the specific pattern didn't catch everything)
echo -e "\n${BLUE}Checking for any remaining Python processes...${NC}"
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "pkill -9 python" 2>/dev/null
echo -e "${GREEN}All remaining Python processes terminated.${NC}"

# Kill screen sessions
echo -e "\n${BLUE}Terminating screen sessions...${NC}"
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "screen -ls | grep -o '[0-9]\{1,\}\..*' | cut -d. -f1 | xargs -I{} screen -S {} -X quit" 2>/dev/null
echo -e "${GREEN}All screen sessions terminated.${NC}"

# Verify everything is terminated
echo -e "\n${BLUE}Verifying all processes are terminated...${NC}"
REMAINING=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "ps aux | grep -E 'python|train_wav2vec|tensorboard' | grep -v grep")

if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}All training jobs successfully terminated!${NC}"
else
    echo -e "${YELLOW}Some processes may still be running:${NC}"
    echo "$REMAINING"
    echo -e "${YELLOW}You may need to manually terminate these processes.${NC}"
fi

# Check GPU status
echo -e "\n${BLUE}Current GPU status:${NC}"
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader"

echo -e "\n${GREEN}Killing jobs complete.${NC}"
