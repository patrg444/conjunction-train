#!/bin/bash

# Setup script for video training monitoring tools
# This script installs all monitoring tools on the AWS instance and makes them executable

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}           SETUP VIDEO TRAINING MONITORING TOOLS           ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"

# Upload all monitoring scripts to EC2
echo -e "\n${CYAN}Uploading monitoring scripts to EC2...${NC}"

# Set default values for key and EC2 instance
DEFAULT_KEY="$HOME/Downloads/gpu-key.pem"
DEFAULT_EC2="ubuntu@54.162.134.77"

# Prompt for the SSH key path
read -p "Enter path to SSH key (default: $DEFAULT_KEY): " KEY_PATH
KEY_PATH=${KEY_PATH:-$DEFAULT_KEY}

# Prompt for the EC2 instance address
read -p "Enter EC2 instance address (default: $DEFAULT_EC2): " EC2_INSTANCE
EC2_INSTANCE=${EC2_INSTANCE:-$DEFAULT_EC2}

# Check if key exists
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}Error: SSH key not found at $KEY_PATH${NC}"
    exit 1
fi

# Upload all monitoring scripts
echo -e "${YELLOW}Uploading monitor_video_training.sh...${NC}"
scp -i "$KEY_PATH" monitor_video_training.sh "$EC2_INSTANCE:~/"

echo -e "${YELLOW}Uploading continuous_video_monitor.sh...${NC}"
scp -i "$KEY_PATH" continuous_video_monitor.sh "$EC2_INSTANCE:~/"

echo -e "${YELLOW}Uploading advanced_video_monitor.sh...${NC}"
scp -i "$KEY_PATH" advanced_video_monitor.sh "$EC2_INSTANCE:~/"

echo -e "${YELLOW}Uploading direct_stream_monitor.sh...${NC}"
scp -i "$KEY_PATH" direct_stream_monitor.sh "$EC2_INSTANCE:~/"

echo -e "${YELLOW}Uploading stream_monitor.sh...${NC}"
scp -i "$KEY_PATH" stream_monitor.sh "$EC2_INSTANCE:~/"

echo -e "${YELLOW}Uploading VIDEO_TRAINING_MONITORING.md...${NC}"
scp -i "$KEY_PATH" VIDEO_TRAINING_MONITORING.md "$EC2_INSTANCE:~/"

# Make scripts executable
echo -e "\n${CYAN}Making scripts executable...${NC}"
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "chmod +x ~/*.sh"

# Create monitoring directory if it doesn't exist
echo -e "\n${CYAN}Setting up monitoring directory...${NC}"
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "mkdir -p ~/monitor_logs"

# Create quick access functions in .bashrc
echo -e "\n${CYAN}Setting up quick access functions...${NC}"
ssh -i "$KEY_PATH" "$EC2_INSTANCE" "cat >> ~/.bashrc << 'EOF'

# Video training monitoring functions
function monitor-basic() {
    ./monitor_video_training.sh
}

function monitor-continuous() {
    ./continuous_video_monitor.sh
}

function monitor-advanced() {
    ./advanced_video_monitor.sh
}

function monitor-background() {
    ./direct_stream_monitor.sh
}

function monitor-stream() {
    ./stream_monitor.sh
}

function monitor-logs() {
    tail -f ~/monitor_logs/video_training_stream.log
}

function monitor-gpu() {
    tail -f ~/monitor_logs/gpu_stats.log
}

function monitor-stop() {
    pkill -f \"bash.*direct_stream_monitor.sh\"
    tmux kill-session -t monitor_* 2>/dev/null
    echo 'Background monitoring stopped'
}

EOF"

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\n${YELLOW}Available on EC2:${NC}"
echo -e "  ${BOLD}./monitor_video_training.sh${NC} - Basic monitoring"
echo -e "  ${BOLD}./continuous_video_monitor.sh${NC} - Continuous GPU monitoring"
echo -e "  ${BOLD}./advanced_video_monitor.sh${NC} - Advanced visual monitoring"
echo -e "  ${BOLD}./direct_stream_monitor.sh${NC} - Background logging"
echo -e "  ${BOLD}./stream_monitor.sh${NC} - TMUX-based streaming"
echo -e "\n${YELLOW}Quick access functions (after logging in again):${NC}"
echo -e "  ${BOLD}monitor-basic${NC} - Run basic monitor"
echo -e "  ${BOLD}monitor-continuous${NC} - Run continuous monitor"
echo -e "  ${BOLD}monitor-advanced${NC} - Run advanced monitor"
echo -e "  ${BOLD}monitor-background${NC} - Start background logging"
echo -e "  ${BOLD}monitor-stream${NC} - Run tmux-based streaming monitor"
echo -e "  ${BOLD}monitor-logs${NC} - View training logs"
echo -e "  ${BOLD}monitor-gpu${NC} - View GPU stats"
echo -e "  ${BOLD}monitor-stop${NC} - Stop background monitoring"
