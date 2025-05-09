#!/bin/bash

# Improved Stream Monitor - Shows formatted training progress
# This script connects to the training session and reformats the output for better readability

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}          IMPROVED TRAINING PROGRESS MONITOR               ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${CYAN}Connecting to training session...${NC}\n"

# Create monitoring directory if it doesn't exist
mkdir -p ~/monitor_logs

# Create a script to reformat the tqdm output
cat > /tmp/reformat_progress.awk <<'EOL'
#!/usr/bin/awk -f
{
  # If line contains training progress with loss and acc
  if ($0 ~ /Training:.*\|.*loss=.*acc=/) {
    # Replace space with comma between loss and acc
    gsub(/loss=/, "loss=", $0)
    gsub(/acc=/, ", acc=", $0)
    
    # Replace brackets with commas
    gsub(/\[/, "", $0)
    gsub(/\]/, "", $0)
    
    # Add commas after it/s
    gsub(/it\/s /, "it/s, ", $0)
    
    print $0
  } else {
    # Print all other lines unchanged
    print $0
  }
}
EOL

chmod +x /tmp/reformat_progress.awk

# Check if tmux session exists
if ! tmux has-session -t video_training 2>/dev/null; then
    echo -e "${RED}Error: Training session 'video_training' not found.${NC}"
    echo -e "${YELLOW}Make sure your training script is running in a tmux session named 'video_training'.${NC}"
    exit 1
fi

# Start GPU monitoring in the background
echo -e "${CYAN}Setting up GPU monitoring...${NC}"
(while true; do 
  gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader)
  echo "GPU: $gpu_info" >> ~/monitor_logs/gpu_stats.log
  sleep 5
done) &
GPU_MONITOR_PID=$!

# Trap to kill the background processes when the script exits
trap "kill $GPU_MONITOR_PID 2>/dev/null; echo -e '${GREEN}Monitoring stopped.${NC}'; exit" INT TERM EXIT

# Continuously capture the training output, reformat and display it
echo -e "${YELLOW}Starting continuous monitoring...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop monitoring${NC}\n"

# Capture training output with formatting
tmux capture-pane -pt video_training -S - > ~/monitor_logs/video_training_stream.log

# Display continuously with formatting
tail -f ~/monitor_logs/video_training_stream.log | awk -f /tmp/reformat_progress.awk
