#!/bin/bash

# Direct formatter for training progress
# This script directly formats and displays the output from the training log

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}          DIRECT TRAINING PROGRESS FORMATTER               ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${CYAN}Starting direct formatting...${NC}\n"

# Create a simple sed script to add commas
cat > /tmp/format_progress.sed <<'EOL'
# Replace space with comma between loss and acc
s/loss=/loss=/g
s/acc=/, acc=/g
# Add comma after it/s
s/it\/s /it\/s, /g
EOL

# Monitor training output with formatting
echo -e "${YELLOW}Formatting continuous training output...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop monitoring${NC}\n"

# SSH to server and run the formatting command
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "
  # Continuously capture training output from the tmux session
  tmux capture-pane -pt video_training -S -
  
  # Set up a continuous watcher on the tmux session
  while true; do
    # Capture the current content of the tmux pane
    tmux capture-pane -pt video_training -S - > /tmp/current_output.txt
    
    # Extract lines with training progress
    grep -E 'Training:.*\|.*loss=.*acc=' /tmp/current_output.txt | tail -1 | 
    # Format with commas
    sed -e 's/loss=/loss=/g' -e 's/acc=/, acc=/g' -e 's/it\/s /it\/s, /g'
    
    # Extract lines with validation results
    grep -E 'Train Loss:|Val Loss:' /tmp/current_output.txt | tail -2 |
    # Format validation results
    sed -e 's/$/,/g' -e '\$ s/,$//'
    
    # Sleep briefly before next capture
    sleep 2
  done
"
