#!/bin/bash
#
# Continuous monitoring script for ATTN-CRNN training
# Provides real-time updates on training progress, errors, and model checkpoints
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"
TMUX_NAME="audio_train"
PROJECT_DIR="~/emotion_project"
CHECKPOINT="$PROJECT_DIR/best_attn_crnn_model.h5"
INTERVAL=180 # Default check interval in seconds

# Terminal colors
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
CYAN=$(tput setaf 6)
RESET=$(tput sgr0)

# Variables
ONESHOT=false

# Catch Ctrl+C for clean exit
trap "echo -e \"\n${CYAN}Exiting monitor.${RESET}\"; exit" INT

# Help text
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -i SECONDS   Set check interval (default: $INTERVAL seconds)"
  echo "  -1           Run once and exit"
  echo "  -s           Show GPU status and exit"
  echo "  -h           Show this help"
  echo
  echo "Examples:"
  echo "  $0           # Run continuous monitoring with default interval"
  echo "  $0 -i 60     # Run continuous monitoring, check every minute"
  echo "  $0 -1        # Run a single check and exit"
  echo "  $0 -s        # Show GPU status and exit"
}

# Parse command line options
while getopts "i:1sh" opt; do
  case $opt in
    i)
      INTERVAL=$OPTARG
      ;;
    1)
      ONESHOT=true
      ;;
    s)
      echo "${CYAN}=== GPU Status ===${RESET}"
      ssh -i "$PEM" ubuntu@$IP "nvidia-smi"
      exit 0
      ;;
    h|*)
      show_help
      exit 0
      ;;
  esac
done

print_header() {
  local current_time=$(date "+%Y-%m-%d %H:%M:%S")
  echo "${CYAN}=========================================================="
  echo "  ATTN-CRNN TRAINING MONITOR"
  echo "=========================================================="
  echo "${RESET}Last check: $current_time"
  echo "Next check in $INTERVAL seconds (Ctrl+C to exit)"
  echo
}

run_monitoring_check() {
  # Connect to remote server for all checks in a single SSH session
  ssh -i "$PEM" ubuntu@$IP <<EOF
    # Check tmux session
    if tmux has-session -t $TMUX_NAME 2>/dev/null; then
      echo "${GREEN}✓ tmux session '$TMUX_NAME' is active${RESET}"
    else
      echo "${RED}✗ WARNING: tmux session '$TMUX_NAME' not found${RESET}"
    fi

    # Check Python process
    if pgrep -f train_attn_crnn.py >/dev/null; then 
      echo "${GREEN}✓ Python training process is active${RESET}"
      
      # Get process start time
      PROCESS_TIME=\$(ps -eo pid,etime,cmd | grep train_attn_crnn.py | grep -v grep | awk '{print \$2}')
      echo "Training duration: \$PROCESS_TIME"
    else
      echo "${RED}✗ WARNING: training process not running${RESET}"
    fi
    
    echo "${CYAN}=== Latest Training Progress ===${RESET}"
    tmux capture-pane -pt $TMUX_NAME | grep -E "Epoch|val_loss|val_accuracy" | tail -10
    
    echo
    echo "${CYAN}=== Error Detection ===${RESET}"
    if tmux capture-pane -pt $TMUX_NAME | tail -50 | grep -iE "error|traceback|exception|failed"; then
      echo "${RED}*** ERRORS DETECTED IN RECENT LOGS ***${RESET}"
    else
      echo "${GREEN}✓ No errors found in recent logs${RESET}"
    fi
    
    # Look for specific issues with WAV2VEC data loading
    if tmux capture-pane -pt $TMUX_NAME | grep -i "no .npz files found"; then
      echo "${RED}*** WARNING: NPZ files not found error detected ***${RESET}"
    fi
    
    echo
    echo "${CYAN}=== Checkpoint Status ===${RESET}"
    if [ -f "$CHECKPOINT" ]; then
      ls -lh "$CHECKPOINT"
      echo "${GREEN}✓ Model checkpoint exists${RESET}"
    else
      echo "${YELLOW}Model checkpoint not found yet${RESET}"
    fi
    
    # Count .npz files to help with data diagnostics
    echo
    echo "${CYAN}=== Dataset Info ===${RESET}"
    echo "WAV2VEC files count: \$(find $PROJECT_DIR -name '*.npz' | wc -l)"
EOF
}

# Main execution loop
while true; do
  clear
  print_header
  run_monitoring_check
  
  if [ "$ONESHOT" = true ]; then
    break
  fi
  
  sleep "$INTERVAL"
done

echo
echo "${CYAN}Monitor session ended${RESET}"
