#!/bin/bash
#
# Streaming monitor for ATTN-CRNN training
# Continuously streams training output in real-time
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"
TMUX_NAME="audio_train"
FILTER_MODE="training"  # Options: "all", "training", "validation"

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
  echo "  -h         Show this help"
  echo
  echo "Examples:"
  echo "  $0         # Stream filtered training output"
  echo "  $0 -a      # Stream all output without filtering"
  echo "  $0 -v      # Focus on validation metrics only"
}

# Default to not doing health check
HEALTH_CHECK=false

# Parse command line options
while getopts "atcvh" opt; do
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
  ssh -i "$PEM" ubuntu@$IP <<EOF
    # Check tmux session
    if tmux has-session -t $TMUX_NAME 2>/dev/null; then
      echo "${GREEN}✓ tmux session '$TMUX_NAME' is active${RESET}"
    else
      echo "${RED}✗ WARNING: tmux session '$TMUX_NAME' not found${RESET}"
      exit 1
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
    
    # Count WAV2VEC files directly without using arrays
    TOTAL_COUNT=0
    echo "Checking for WAV2VEC feature files..."
    
    # Use a simpler direct approach to check each directory
    COUNT1=$(find -L ~/emotion_project/wav2vec_features -name '*.npz' 2>/dev/null | wc -l)
    if [ "$COUNT1" -gt 0 ]; then
      echo "   Found $COUNT1 feature files in ~/emotion_project/wav2vec_features"
      TOTAL_COUNT=$((TOTAL_COUNT + COUNT1))
    fi
    
    COUNT2=$(find -L ~/audio_emotion/models/wav2vec -name '*.npz' 2>/dev/null | wc -l)
    if [ "$COUNT2" -gt 0 ]; then
      echo "   Found $COUNT2 feature files in ~/audio_emotion/models/wav2vec"
      TOTAL_COUNT=$((TOTAL_COUNT + COUNT2))
    fi
    
    COUNT3=$(find -L ~/emotion-recognition/crema_d_features_facenet -name '*.npz' 2>/dev/null | wc -l)
    if [ "$COUNT3" -gt 0 ]; then
      echo "   Found $COUNT3 feature files in ~/emotion-recognition/crema_d_features_facenet"
      TOTAL_COUNT=$((TOTAL_COUNT + COUNT3))
    fi
    
    COUNT4=$(find -L ~/emotion-recognition/npz_files/CREMA-D -name '*.npz' 2>/dev/null | wc -l)
    if [ "$COUNT4" -gt 0 ]; then
      echo "   Found $COUNT4 feature files in ~/emotion-recognition/npz_files/CREMA-D"
      TOTAL_COUNT=$((TOTAL_COUNT + COUNT4))
    fi
    
    COUNT5=$(find -L ~/emotion-recognition/crema_d_features_audio -name '*.npz' 2>/dev/null | wc -l)
    if [ "$COUNT5" -gt 0 ]; then
      echo "   Found $COUNT5 feature files in ~/emotion-recognition/crema_d_features_audio"
      TOTAL_COUNT=$((TOTAL_COUNT + COUNT5))
    fi
    
    echo "WAV2VEC feature files found: \$TOTAL_COUNT"
    
    # Check for model checkpoint
    if [ -f ~/emotion_project/best_attn_crnn_model.h5 ]; then
      echo "${GREEN}✓ Model checkpoint exists${RESET}"
      ls -lh ~/emotion_project/best_attn_crnn_model.h5
    else
      echo "${YELLOW}Model checkpoint not found yet${RESET}"
    fi
EOF

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
