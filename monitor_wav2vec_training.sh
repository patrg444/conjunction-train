#!/bin/bash
# Comprehensive wav2vec training monitoring script
# Combines log monitoring, GPU stats, and error detection

# Settings
SSH_KEY="$1"
AWS_IP="$2"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="/home/$SSH_USER/audio_emotion"
REMOTE_LOG="${EC2_PROJECT_PATH}/train_wav2vec_audio_only.log"
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

clear
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          WAV2VEC AUDIO EMOTION TRAINING MONITOR           ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}$AWS_IP${NC}"
echo -e "Remote path: ${CYAN}$REMOTE_LOG${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if training process is running
echo -e "\n${YELLOW}Checking if wav2vec training is running...${NC}"
is_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f 'train_wav2vec_audio_only.py' || echo 'not_running'")

if [[ "$is_running" == "not_running" ]]; then
  echo -e "${RED}⚠️  WARNING: Training process is not running!${NC}"
  echo -e "You can restart training with: ./run_wav2vec_audio_only.sh"
  
  # Check if log file exists and was recently updated
  log_exists=$(ssh -i "$SSH_KEY" "$SSH_HOST" "test -f $REMOTE_LOG && echo 'exists' || echo 'missing'")
  
  if [[ "$log_exists" == "exists" ]]; then
    last_modified=$(ssh -i "$SSH_KEY" "$SSH_HOST" "stat -c %Y $REMOTE_LOG 2>/dev/null || stat -f %m $REMOTE_LOG 2>/dev/null")
    current_time=$(date +%s)
    time_diff=$((current_time - last_modified))
    
    if [[ $time_diff -lt 3600 ]]; then  # Less than 1 hour
      echo -e "${YELLOW}Training may have finished or crashed recently.${NC}"
      echo -e "Showing last 15 lines of log file:"
      ssh -i "$SSH_KEY" "$SSH_HOST" "tail -15 $REMOTE_LOG"
    else
      echo -e "${YELLOW}Log file exists but hasn't been updated for $(($time_diff / 3600)) hours.${NC}"
    fi
  else
    echo -e "${RED}Log file not found: $REMOTE_LOG${NC}"
  fi
else
  echo -e "${GREEN}✅ Training process active (PID: $is_running)${NC}"
  
  # Check GPU utilization
  echo -e "\n${YELLOW}Checking GPU utilization...${NC}"
  gpu_info=$(ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
  
  # Parse GPU info
  gpu_util=$(echo "$gpu_info" | cut -d, -f1 | tr -d ' ')
  mem_util=$(echo "$gpu_info" | cut -d, -f2 | tr -d ' ')
  mem_used=$(echo "$gpu_info" | cut -d, -f3 | tr -d ' ')
  mem_total=$(echo "$gpu_info" | cut -d, -f4 | tr -d ' ')
  gpu_temp=$(echo "$gpu_info" | cut -d, -f5 | tr -d ' ')
  
  # Display GPU info with color coding
  if [[ -z "$gpu_util" ]]; then
    echo -e "${RED}Error retrieving GPU information${NC}"
  else
    # Display GPU utilization with appropriate color
    if [[ "$gpu_util" -lt 30 ]]; then
      echo -e "GPU Utilization: ${RED}$gpu_util%${NC} (low - should be 60-90%)"
    elif [[ "$gpu_util" -lt 60 ]]; then
      echo -e "GPU Utilization: ${YELLOW}$gpu_util%${NC} (medium)"
    else
      echo -e "GPU Utilization: ${GREEN}$gpu_util%${NC} (good)"
    fi
    
    # Check memory usage - for batch size 64 should use ~6-8GB
    if [[ "$mem_used" -lt 4000 ]]; then
      echo -e "Memory Usage: ${RED}$mem_used MB / $mem_total MB${NC} (low - check batch size)"
    elif [[ "$mem_used" -gt 12000 && "$mem_total" -lt 15000 ]]; then
      echo -e "Memory Usage: ${RED}$mem_used MB / $mem_total MB${NC} (high - risk of OOM)"
    else
      echo -e "Memory Usage: ${GREEN}$mem_used MB / $mem_total MB${NC} (${mem_util}%)"
    fi
    
    echo -e "GPU Temperature: ${gpu_temp}°C"
  fi
  
  # Check if TensorBoard is running
  echo -e "\n${YELLOW}Checking if TensorBoard is running...${NC}"
  tb_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f tensorboard || echo 'not_running'")
  
  if [[ "$tb_running" == "not_running" ]]; then
    echo -e "${YELLOW}TensorBoard not running.${NC}"
    echo -e "You can start TensorBoard with: ./setup_tensorboard_tunnel.sh $SSH_KEY $AWS_IP"
  else
    echo -e "${GREEN}✅ TensorBoard running (PID: $tb_running)${NC}"
    echo -e "To access TensorBoard, open a separate terminal and run:"
    echo -e "${CYAN}ssh -i $SSH_KEY -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} $SSH_HOST${NC}"
    echo -e "Then visit: ${CYAN}http://localhost:${TENSORBOARD_PORT}${NC} in your browser"
  fi
  
  # Extract training progress information
  echo -e "\n${YELLOW}Extracting training progress...${NC}"
  
  # Check if log file exists
  log_exists=$(ssh -i "$SSH_KEY" "$SSH_HOST" "test -f $REMOTE_LOG && echo 'exists' || echo 'missing'")
  
  if [[ "$log_exists" == "exists" ]]; then
    # Extract epoch information with retries in case the log is being written
    echo -e "\n${BLUE}Training Progress:${NC}"
    
    # Try to find the latest epoch info
    epoch_info=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -E 'Epoch [0-9]+/[0-9]+' $REMOTE_LOG | tail -1")
    
    if [[ -n "$epoch_info" ]]; then
      echo -e "$epoch_info"
      
      # Extract metrics if available
      echo -e "\n${BLUE}Recent Metrics:${NC}"
      
      # Get the last few accuracy metrics
      val_acc=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -E 'val_accuracy: [0-9.]+' $REMOTE_LOG | tail -3")
      
      if [[ -n "$val_acc" ]]; then
        echo -e "${GREEN}Validation Accuracy:${NC}"
        echo -e "$val_acc"
      else
        echo -e "${YELLOW}No validation metrics found yet.${NC}"
      fi
      
      # Check for errors or warnings
      echo -e "\n${BLUE}Checking for Errors:${NC}"
      errors=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -i 'error\\|exception\\|failed\\|OOM\\|out of memory' $REMOTE_LOG | wc -l")
      
      if [[ "$errors" -gt 0 ]]; then
        echo -e "${RED}Found $errors error-related messages.${NC}"
        echo -e "${RED}Last error:${NC}"
        ssh -i "$SSH_KEY" "$SSH_HOST" "grep -i 'error\\|exception\\|failed\\|OOM\\|out of memory' $REMOTE_LOG | tail -1"
      else
        echo -e "${GREEN}No errors detected in log.${NC}"
      fi
    else
      echo -e "${YELLOW}Could not find epoch information. Training may have just started.${NC}"
    fi
    
    echo -e "\n${BLUE}Live Log Stream (press Ctrl+C to exit):${NC}"
    ssh -i "$SSH_KEY" "$SSH_HOST" "tail -f $REMOTE_LOG"
  else
    echo -e "${RED}Log file not found at: $REMOTE_LOG${NC}"
  fi
fi
