#!/bin/bash

# Smart Streaming monitor for ATTN-CRNN training
# Enhanced feature detection to prevent unnecessary uploads
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"
TMUX_NAME="audio_train"
FILTER_MODE="training"  # Options: "all" "training" "validation"

# Terminal colors
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
CYAN=$(tput setaf 6)
MAGENTA=$(tput setaf 5)
RESET=$(tput sgr0)
BOLD=$(tput bold)

# Catch Ctrl+C for clean exit
trap "echo -e \"\n${CYAN}Exiting streaming monitor.${RESET}\"; exit" INT

# Help text
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -a         Show all output (default: filter for training info)"
  echo "  -t         Filter for training/epoch lines only"
  echo "  -v         Filter for validation metrics only"
  echo "  -c         Check session health before streaming"
  echo "  -f         Force detailed WAV2VEC feature check (slower)"
  echo "  -h         Show this help"
  echo
  echo "Examples:"
  echo "  $0         # Stream filtered training output"
  echo "  $0 -a      # Stream all output without filtering"
  echo "  $0 -v      # Focus on validation metrics only"
  echo "  $0 -c -f   # Do health check with thorough feature search"
}

# Default to not doing health check and not doing forced feature check
HEALTH_CHECK=false
FORCE_FEATURE_CHECK=false

# Parse command line options
while getopts "atcvfh" opt; do
  case $opt in
    a)
      FILTER_MODE="all"
      ;;
    t)
      FILTER_MODE="training"
      ;;
    v)
      FILTER_MODE="validation"
      ;;
    c)
      HEALTH_CHECK=true
      ;;
    f)
      FORCE_FEATURE_CHECK=true
      ;;
    h|*)
      show_help
      exit 0
      ;;
  esac
done

# Quick health check if requested
if $HEALTH_CHECK; then
  echo "${CYAN}${BOLD}=== ATTN-CRNN TRAINING HEALTH CHECK ===${RESET}"

  # Connect to remote server for health check
  echo "${CYAN}Checking training status...${RESET}"
  
  # Check tmux session existence
  TMUX_EXISTS=$(ssh -i "$PEM" ubuntu@$IP "tmux has-session -t $TMUX_NAME 2>/dev/null && echo 'yes' || echo 'no'")
  if [[ "$TMUX_EXISTS" == "yes" ]]; then
    echo "${GREEN}✓ tmux session '$TMUX_NAME' is active${RESET}"
  else
    echo "${RED}✗ WARNING: tmux session '$TMUX_NAME' not found${RESET}"
  fi

  # Check Python process
  PROCESS_RUNNING=$(ssh -i "$PEM" ubuntu@$IP "pgrep -f train_attn_crnn.py >/dev/null && echo 'yes' || echo 'no'")
  if [[ "$PROCESS_RUNNING" == "yes" ]]; then
    echo "${GREEN}✓ Python training process is active${RESET}"
    
    # Get process start time
    PROCESS_TIME=$(ssh -i "$PEM" ubuntu@$IP "ps -eo pid,etime,cmd | grep train_attn_crnn.py | grep -v grep | awk '{print \$2}'")
    echo "Training duration: $PROCESS_TIME"
  else
    echo "${RED}✗ WARNING: training process not running${RESET}"
  fi

  # Use a more reliable approach to WAV2VEC feature detection
  if $FORCE_FEATURE_CHECK; then
    echo "${CYAN}Performing thorough WAV2VEC feature search (this may take a moment)...${RESET}"
    FEATURE_COUNT=$(ssh -i "$PEM" ubuntu@$IP "find /home/ubuntu -path '*wav2vec*' -name '*.npz' 2>/dev/null | wc -l")
    if [ "$FEATURE_COUNT" -gt 0 ]; then
      echo "${GREEN}✓ Found $FEATURE_COUNT WAV2VEC feature files${RESET}"
      ssh -i "$PEM" ubuntu@$IP "find /home/ubuntu -path '*wav2vec*' -name '*.npz' 2>/dev/null | head -n 3" | \
        awk '{printf "  - %s\n", $0}'
      echo "  ... and $(($FEATURE_COUNT-3)) more"
    else
      echo "${RED}✗ No WAV2VEC feature files found${RESET}"
    fi
  else
    # Quick check of known locations
    echo "${CYAN}Checking common WAV2VEC feature locations...${RESET}"
    FEATURE_LOCATIONS=$(ssh -i "$PEM" ubuntu@$IP "
      find -L /home/ubuntu/audio_emotion/models/wav2vec -name '*.npz' 2>/dev/null | wc -l
      find -L /home/ubuntu/emotion_project/wav2vec_features -name '*.npz' 2>/dev/null | wc -l
      find -L /home/ubuntu/wav2vec_sample -name '*.npz' 2>/dev/null | wc -l
      find -L /home/ubuntu/audio_emotion/data/*wav2vec* -name '*.npz' 2>/dev/null | wc -l
    ")
    
    # Convert multiline string to array
    IFS=$'\n' read -rd '' -a COUNTS <<< "$FEATURE_LOCATIONS"
    
    TOTAL=0
    LOCATIONS=("/home/ubuntu/audio_emotion/models/wav2vec" 
               "/home/ubuntu/emotion_project/wav2vec_features" 
               "/home/ubuntu/wav2vec_sample"
               "/home/ubuntu/audio_emotion/data/*wav2vec*")
    
    for i in "${!COUNTS[@]}"; do
      if [ "${COUNTS[$i]}" -gt 0 ]; then
        echo "  - ${GREEN}${COUNTS[$i]} files in ${LOCATIONS[$i]}${RESET}"
        TOTAL=$((TOTAL + COUNTS[$i]))
      fi
    done
    
    if [ "$TOTAL" -gt 0 ]; then
      echo "${GREEN}✓ Total WAV2VEC feature files: $TOTAL${RESET}"
    else
      echo "${RED}✗ No WAV2VEC feature files found in common locations${RESET}"
      echo "${YELLOW}Use -f option for a deeper search${RESET}"
    fi
  fi

  # Check for model checkpoint
  MODEL_EXISTS=$(ssh -i "$PEM" ubuntu@$IP "test -f /home/ubuntu/emotion_project/best_attn_crnn_model.h5 && echo 'yes' || echo 'no'")
  if [[ "$MODEL_EXISTS" == "yes" ]]; then
    MODEL_SIZE=$(ssh -i "$PEM" ubuntu@$IP "ls -lh /home/ubuntu/emotion_project/best_attn_crnn_model.h5 | awk '{print \$5}'")
    echo "${GREEN}✓ Model checkpoint exists (size: $MODEL_SIZE)${RESET}"
  else
    echo "${YELLOW}Model checkpoint not found yet${RESET}"
  fi

  echo
  echo "${CYAN}${BOLD}=== STARTING CONTINUOUS STREAM ===${RESET}"
  echo "${CYAN}Press Ctrl+C to exit${RESET}"
  echo
  # Sleep a bit so user can see health check results before streaming starts
  sleep 2
fi

# Function to highlight important patterns in the output
highlight_output() {
  # Highlight validation accuracy and loss
  sed -E "s/(val_accuracy: [0-9.]+)/${GREEN}\\1${RESET}/g" |
  sed -E "s/(val_loss: [0-9.]+)/${MAGENTA}\\1${RESET}/g" |
  # Highlight epoch info
  sed -E "s/(Epoch [0-9]+\/[0-9]+)/${CYAN}\\1${RESET}/g" |
  # Highlight errors and warnings
  sed -E "s/(Error|ERROR|Exception|EXCEPTION|Warning|WARNING)/${RED}\\1${RESET}/g" |
  # Highlight NaN values which indicate training problems
  sed -E "s/(NaN)/${RED}${BOLD}\\1${RESET}/g"
}

# Ensure clean start with a new log file
ssh -i "$PEM" ubuntu@$IP "rm -f /tmp/tmux_output.log"

# Stream the tmux session content
case $FILTER_MODE in
  "all")
    echo "${CYAN}Streaming all training output...${RESET}"
    ssh -i "$PEM" ubuntu@$IP "tmux pipe-pane -t $TMUX_NAME 'cat >> /tmp/tmux_output.log'; tail -f /tmp/tmux_output.log" | highlight_output
    ;;
  "training")
    echo "${CYAN}Streaming filtered training output (epoch updates)...${RESET}"
    ssh -i "$PEM" ubuntu@$IP "tmux pipe-pane -t $TMUX_NAME 'cat >> /tmp/tmux_output.log'; tail -f /tmp/tmux_output.log | grep -E 'Epoch|val_loss|val_accuracy|Error|Exception|WARNING'" | highlight_output
    ;;
  "validation")
    echo "${CYAN}Streaming validation metrics only...${RESET}"
    ssh -i "$PEM" ubuntu@$IP "tmux pipe-pane -t $TMUX_NAME 'cat >> /tmp/tmux_output.log'; tail -f /tmp/tmux_output.log | grep -E 'val_loss|val_accuracy'" | highlight_output
    ;;
esac
