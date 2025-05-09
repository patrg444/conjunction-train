#!/bin/bash
#
# Monitoring script for ATTN-CRNN training
#

set -eo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

function print_usage() {
  echo "ATTN-CRNN Training Monitoring Tool"
  echo ""
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -l, --logs      Show recent training logs"
  echo "  -c, --check     Check training status"
  echo "  -s, --stats     Show GPU and memory usage"
  echo "  -m, --model     Check if model has been saved"
  echo "  -t, --tmux      Attach to tmux session directly (use Ctrl+b d to detach)"
  echo "  -h, --help      Show this help message"
  echo ""
  echo "Example: $0 -l    # Show recent training logs"
}

function show_logs() {
  echo "=== RECENT TRAINING LOGS ==="
  ssh -i "$PEM" ubuntu@$IP "tail -n 150 /home/ubuntu/emotion_project/logs/attn_crnn_training_v2_*.log"
}

function check_status() {
  echo "=== CHECKING TRAINING STATUS ==="
  ssh -i "$PEM" ubuntu@$IP << 'EOF'
    set -eo pipefail

    # Check if tmux session exists
    if tmux has-session -t audio_train 2>/dev/null; then
      echo "✓ tmux session 'audio_train' is running"
      TMUX_PID=$(tmux list-panes -t audio_train -F "#{pane_pid}" 2>/dev/null || echo "")
      if [ -n "$TMUX_PID" ]; then
        CHILD_PIDS=$(pgrep -P "$TMUX_PID" || echo "")
        if [ -n "$CHILD_PIDS" ]; then
          echo "✓ Training process is active"
          
          # Check for Python process
          PYTHON_PID=$(pgrep -P "$TMUX_PID" -l | grep python | awk '{print $1}')
          if [ -n "$PYTHON_PID" ]; then
            echo "✓ Python training process found (PID: $PYTHON_PID)"
            RUNTIME=$(ps -o etime= -p "$PYTHON_PID")
            echo "  Running time: $RUNTIME"
          else
            echo "✗ Warning: No Python process found in tmux session"
          fi
        else
          echo "✗ Warning: No active processes in tmux session"
        fi
      else
        echo "✗ Warning: Could not determine tmux session PID"
      fi
    else
      echo "✗ Error: tmux session 'audio_train' not found"
    fi

    # Check latest log file
    LATEST_LOG=$(ls -t /home/ubuntu/emotion_project/logs/attn_crnn_training_v2_*.log 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
      echo ""
      echo "Latest log file: $LATEST_LOG"
      LAST_UPDATED=$(stat -c %y "$LATEST_LOG")
      echo "Last updated: $LAST_UPDATED"
      
      echo ""
      echo "Last few lines from the log:"
      tail -n 3 "$LATEST_LOG"
    else
      echo ""
      echo "No training logs found"
    fi
EOF
}

function show_stats() {
  echo "=== GPU AND MEMORY USAGE ==="
  ssh -i "$PEM" ubuntu@$IP << 'EOF'
    set -eo pipefail
    
    echo "GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    
    echo ""
    echo "Memory Usage:"
    free -h
    
    echo ""
    echo "Top processes by memory usage:"
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 6
EOF
}

function check_model() {
  echo "=== CHECKING MODEL FILES ==="
  ssh -i "$PEM" ubuntu@$IP << 'EOF'
    set -eo pipefail
    
    MODEL_FILE="/home/ubuntu/emotion_project/best_attn_crnn_model.h5"
    if [ -f "$MODEL_FILE" ]; then
      FILE_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
      LAST_MODIFIED=$(stat -c %y "$MODEL_FILE")
      echo "✓ Model file exists: $MODEL_FILE"
      echo "  Size: $FILE_SIZE"
      echo "  Last modified: $LAST_MODIFIED"
    else
      echo "✗ Model file not found at $MODEL_FILE"
    fi
    
    # Check for TensorBoard logs
    LOG_DIR="/home/ubuntu/emotion_project/logs"
    TB_DIRS=$(find "$LOG_DIR" -type d -name "*attn_crnn*" 2>/dev/null)
    if [ -n "$TB_DIRS" ]; then
      echo ""
      echo "TensorBoard log directories found:"
      for DIR in $TB_DIRS; do
        echo "  $DIR"
      done
    fi
EOF
}

function attach_tmux() {
  echo "=== ATTACHING TO TMUX SESSION ==="
  echo "Connecting to tmux session 'audio_train'..."
  echo "NOTE: To detach from the session without stopping it, press Ctrl+b then d"
  ssh -t -i "$PEM" ubuntu@$IP "tmux attach -t audio_train"
}

# Process arguments
if [ $# -eq 0 ]; then
  print_usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--logs)
      show_logs
      shift
      ;;
    -c|--check)
      check_status
      shift
      ;;
    -s|--stats)
      show_stats
      shift
      ;;
    -m|--model)
      check_model
      shift
      ;;
    -t|--tmux)
      attach_tmux
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done
