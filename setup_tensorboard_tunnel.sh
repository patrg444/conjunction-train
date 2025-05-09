#!/bin/bash
# Script to set up TensorBoard monitoring for wav2vec training

# Settings
SSH_KEY="$1"
AWS_IP="$2"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="/home/$SSH_USER/audio_emotion"
TENSORBOARD_PORT=6006

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
  echo "Usage: $0 <path-to-key.pem> <ec2-ip-address>"
  echo "Example: $0 ~/Downloads/gpu-key.pem 54.162.134.77"
  exit 1
}

# Check parameters
if [[ -z "$SSH_KEY" || -z "$AWS_IP" ]]; then
  echo -e "${RED}Error: Missing required parameters${NC}"
  show_usage
fi

# Check if key file exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo -e "${RED}Error: SSH key file does not exist: $SSH_KEY${NC}"
  exit 1
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          TENSORBOARD SETUP FOR WAV2VEC TRAINING           ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}$AWS_IP${NC}"
echo -e "TensorBoard port: ${CYAN}$TENSORBOARD_PORT${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if TensorBoard is already running on remote server
echo -e "\n${YELLOW}Checking if TensorBoard is already running...${NC}"
tb_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f tensorboard || echo 'not_running'")

if [[ "$tb_running" != "not_running" ]]; then
  echo -e "${GREEN}✅ TensorBoard already running (PID: $tb_running)${NC}"
else
  echo -e "${YELLOW}TensorBoard not running. Starting it now...${NC}"
  
  # Create logs directory if it doesn't exist
  ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p $EC2_PROJECT_PATH/logs"
  
  # Start TensorBoard in the background
  ssh -i "$SSH_KEY" "$SSH_HOST" "cd $EC2_PROJECT_PATH && nohup tensorboard --logdir=logs --port=$TENSORBOARD_PORT --bind_all > tensorboard.log 2>&1 &"
  
  # Check if it started successfully
  sleep 2
  tb_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f tensorboard || echo 'not_running'")
  
  if [[ "$tb_running" != "not_running" ]]; then
    echo -e "${GREEN}✅ TensorBoard started successfully (PID: $tb_running)${NC}"
  else
    echo -e "${RED}⚠️ Failed to start TensorBoard. Check tensorboard.log on the remote server.${NC}"
    exit 1
  fi
fi

# Create SSH tunnel in a new terminal window
echo -e "\n${YELLOW}Setting up SSH tunnel for TensorBoard...${NC}"
echo -e "${GREEN}Run the following command in a new terminal to create the tunnel:${NC}"
echo -e "${CYAN}ssh -i $SSH_KEY -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} $SSH_HOST${NC}"

echo -e "\n${YELLOW}Once the tunnel is established, open TensorBoard in your browser:${NC}"
echo -e "${CYAN}http://localhost:${TENSORBOARD_PORT}${NC}"

# Provide command for checking TensorBoard logs if needed
echo -e "\n${YELLOW}To check TensorBoard logs if there are issues:${NC}"
echo -e "${CYAN}ssh -i $SSH_KEY $SSH_HOST \"cat $EC2_PROJECT_PATH/tensorboard.log\"${NC}"

# Display command to kill TensorBoard if needed
echo -e "\n${YELLOW}If you need to restart TensorBoard, run:${NC}"
echo -e "${CYAN}ssh -i $SSH_KEY $SSH_HOST \"pkill -f tensorboard\"${NC}"
echo -e "${CYAN}Then run this script again.${NC}"
