#!/usr/bin/env bash
# Continuous monitoring script for G5 training
# This script continuously monitors:
# - GPU utilization and memory
# - Feature directory sizes
# - Training progress and metrics
# - Error detection
#
# Usage: ./continuous_g5_monitor.sh [interval_seconds=30]
#   - interval_seconds: Polling interval (default: 30 seconds)

# Set constants
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="/home/$SSH_USER/emotion-recognition"
INTERVAL=${1:-30}  # Default polling interval: 30 seconds
TENSORBOARD_PORT=6006

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          CONTINUOUS G5 TRAINING MONITOR${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Monitoring G5 instance: $AWS_IP"
echo "Polling interval: $INTERVAL seconds"
echo "Press Ctrl+C to exit"
echo -e "${BLUE}============================================================${NC}"

# Function to draw a progress bar
# Args: $1 = current value, $2 = total value, $3 = width
progress_bar() {
    local current=$1
    local total=$2
    local width=${3:-50}
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "["
    printf "%${filled}s" | tr ' ' '='
    printf ">"
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# Function to format byte sizes
format_size() {
    local size=$1
    local unit="B"
    
    if [ $size -ge 1073741824 ]; then
        printf "%.1f GB" $(echo "scale=1; $size / 1073741824" | bc)
    elif [ $size -ge 1048576 ]; then
        printf "%.1f MB" $(echo "scale=1; $size / 1048576" | bc)
    elif [ $size -ge 1024 ]; then
        printf "%.1f KB" $(echo "scale=1; $size / 1024" | bc)
    else
        printf "%d B" $size
    fi
}

# Function to check if TensorBoard is running, start if not
check_tensorboard() {
    local tb_running
    tb_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f tensorboard || echo 'not_running'")
    
    if [[ "$tb_running" == "not_running" ]]; then
        echo -e "${YELLOW}TensorBoard not running. Starting it...${NC}"
        ssh -i "$SSH_KEY" "$SSH_HOST" "cd $EC2_PROJECT_PATH && \
            mkdir -p logs/tensorboard && \
            nohup tensorboard --logdir=logs/tensorboard --port=$TENSORBOARD_PORT --host=localhost > ~/tensorboard.log 2>&1 &"
        
        echo -e "${YELLOW}To view TensorBoard, open a new terminal and run:${NC}"
        echo -e "${CYAN}  ssh -i $SSH_KEY -L $TENSORBOARD_PORT:localhost:$TENSORBOARD_PORT $SSH_HOST${NC}"
        echo -e "${CYAN}  Then open http://localhost:$TENSORBOARD_PORT in your browser${NC}"
    else
        echo -e "${GREEN}TensorBoard running (PID: $tb_running)${NC}"
    fi
}

# Function to check if training is running
check_training_status() {
    local training_pid
    training_pid=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f 'train_audio_pooling_lstm_with_laughter.py' || echo 'not_running'")
    
    if [[ "$training_pid" == "not_running" ]]; then
        echo -e "${RED}⚠️  WARNING: Training process is not running!${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Training process active (PID: $training_pid)${NC}"
        return 0
    fi
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BLUE}======== G5 TRAINING MONITOR - $(date) ========${NC}"
    
    # 1. Check if training is running
    echo -e "${CYAN}[1/5] TRAINING PROCESS STATUS${NC}"
    check_training_status
    training_running=$?
    
    # 2. Check data presence and sizes
    echo -e "\n${CYAN}[2/5] DATASET STATUS${NC}"
    dataset_info=$(ssh -i "$SSH_KEY" "$SSH_HOST" "cd $EC2_PROJECT_PATH && \
        echo -n 'RAVDESS: ' && du -sb ravdess_features_facenet/ 2>/dev/null | cut -f1 || echo '0' && \
        echo -n 'CREMA-D: ' && du -sb crema_d_features_facenet/ 2>/dev/null | cut -f1 || echo '0' && \
        echo -n 'Manifest entries: ' && [ -f datasets/manifests/laughter_v1.csv ] && wc -l < datasets/manifests/laughter_v1.csv || echo '0' && \
        echo -n 'Normalization files: ' && find models -name '*normalization_stats.pkl' | wc -l")
    
        # Parse dataset info (macOS compatible)
    ravdess_bytes=$(echo "$dataset_info" | grep -o 'RAVDESS: [0-9]*' | sed 's/RAVDESS: //')
    cremad_bytes=$(echo "$dataset_info" | grep -o 'CREMA-D: [0-9]*' | sed 's/CREMA-D: //')
    manifest_count=$(echo "$dataset_info" | grep -o 'Manifest entries: [0-9]*' | sed 's/Manifest entries: //')
    norm_files=$(echo "$dataset_info" | grep -o 'Normalization files: [0-9]*' | sed 's/Normalization files: //')
    
    # Set defaults if parsing failed
    ravdess_bytes=${ravdess_bytes:-0}
    cremad_bytes=${cremad_bytes:-0}
    manifest_count=${manifest_count:-0}
    norm_files=${norm_files:-0}
    
    # Format sizes
    ravdess_size=$(format_size $ravdess_bytes)
    cremad_size=$(format_size $cremad_bytes)
    
    # Display dataset info with color coding (with additional error handling)
    if [ -z "$ravdess_bytes" ] || [ "$ravdess_bytes" -lt 1000000 ]; then
        echo -e "RAVDESS features: ${RED}$(format_size ${ravdess_bytes:-0})${NC} (should be ~1.6 GB)"
    else
        echo -e "RAVDESS features: ${GREEN}$(format_size $ravdess_bytes)${NC}"
    fi
    
    if [ -z "$cremad_bytes" ] || [ "$cremad_bytes" -lt 1000000 ]; then
        echo -e "CREMA-D features: ${RED}$(format_size ${cremad_bytes:-0})${NC} (should be ~1.0 GB)"
    else
        echo -e "CREMA-D features: ${GREEN}$(format_size $cremad_bytes)${NC}"
    fi
    
    if [ -z "$manifest_count" ] || [ "$manifest_count" -lt 10 ]; then
        echo -e "Laughter manifest: ${RED}${manifest_count:-0} entries${NC} (should be hundreds of entries)"
    else
        echo -e "Laughter manifest: ${GREEN}$manifest_count entries${NC}"
    fi
    
    if [ -z "$norm_files" ] || [ "$norm_files" -lt 2 ]; then
        echo -e "Normalization files: ${RED}${norm_files:-0}${NC} (need at least 2 files)"
    else
        echo -e "Normalization files: ${GREEN}$norm_files${NC}"
    fi
    
    # 3. Check GPU status
    echo -e "\n${CYAN}[3/5] GPU STATUS${NC}"
    gpu_info=$(ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
    
    # Parse GPU info
    gpu_util=$(echo "$gpu_info" | cut -d, -f1 | tr -d ' ')
    mem_util=$(echo "$gpu_info" | cut -d, -f2 | tr -d ' ')
    mem_used=$(echo "$gpu_info" | cut -d, -f3 | tr -d ' ')
    mem_total=$(echo "$gpu_info" | cut -d, -f4 | tr -d ' ')
    gpu_temp=$(echo "$gpu_info" | cut -d, -f5 | tr -d ' ')
    
    # Display GPU info with color coding
    if [ "$gpu_util" -lt 30 ]; then
        echo -e "GPU Utilization: ${RED}$gpu_util%${NC} (low - should be 60-90%)"
    elif [ "$gpu_util" -lt 60 ]; then
        echo -e "GPU Utilization: ${YELLOW}$gpu_util%${NC} (medium)"
    else
        echo -e "GPU Utilization: ${GREEN}$gpu_util%${NC} (good)"
    fi
    
    echo -e "Memory Usage: $mem_used MB / $mem_total MB (${mem_util}%)"
    echo -e "GPU Temperature: ${gpu_temp}°C"
    
    # 4. Show training progress
    echo -e "\n${CYAN}[4/5] TRAINING PROGRESS${NC}"
    if [ $training_running -eq 0 ]; then
        # Find the latest training log
        latest_log=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ls -t $EC2_PROJECT_PATH/logs/train_laugh_*.log 2>/dev/null | head -1 || echo ''")
        
        if [ -n "$latest_log" ]; then
            # Extract current epoch
            current_epoch=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -o 'Epoch [0-9]*/100' $latest_log | tail -1 | cut -d' ' -f2 | cut -d'/' -f1")
            
            if [ -n "$current_epoch" ]; then
                # Extract more metrics
                latest_loss=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -o 'loss: [0-9.]*' $latest_log | tail -1 | cut -d' ' -f2")
                latest_acc=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -o 'emotion_output_accuracy: [0-9.]*' $latest_log | tail -1 | cut -d' ' -f2")
                latest_laugh_acc=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -o 'laugh_output_accuracy: [0-9.]*' $latest_log | tail -1 | cut -d' ' -f2")
                
                # Extract time info
                start_time=$(ssh -i "$SSH_KEY" "$SSH_HOST" "stat -c %Y $latest_log")
                current_time=$(date +%s)
                elapsed_seconds=$((current_time - start_time))
                elapsed_hours=$(echo "scale=1; $elapsed_seconds/3600" | bc)
                
                # Calculate estimates
                if [ "$current_epoch" -gt 0 ]; then
                    avg_epoch_time=$(echo "scale=2; $elapsed_seconds/$current_epoch" | bc 2>/dev/null)
                    remaining_epochs=$((100 - current_epoch))
                    remaining_seconds=$(echo "$avg_epoch_time * $remaining_epochs" | bc 2>/dev/null)
                    remaining_hours=$(echo "scale=1; $remaining_seconds/3600" | bc)
                    
                    # Display progress
                    echo -e "Progress: $(progress_bar $current_epoch 100)"
                    echo -e "Running for: ${elapsed_hours}h, Remaining: ${remaining_hours}h"
                    echo -e "Current loss: ${latest_loss}, Emotion acc: ${latest_acc}, Laugh acc: ${latest_laugh_acc}"
                    
                    # Show estimated completion time
                    estimated_completion=$(date -d "+$remaining_seconds seconds" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -v "+${remaining_seconds}S" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
                    if [ -n "$estimated_completion" ]; then
                        echo -e "Estimated completion: $estimated_completion"
                    fi
                else
                    echo -e "${YELLOW}Training just started, waiting for first epoch to complete...${NC}"
                fi
            else
                echo -e "${YELLOW}Waiting for first epoch to start...${NC}"
            fi
            
            # Check for errors in the log
            error_count=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -i 'error\|exception\|failed' $latest_log | wc -l")
            if [ "$error_count" -gt 0 ]; then
                echo -e "${RED}⚠️  WARNING: $error_count error/exception mentions found in log!${NC}"
                echo -e "${RED}Last error:${NC}"
                ssh -i "$SSH_KEY" "$SSH_HOST" "grep -i 'error\|exception\|failed' $latest_log | tail -1"
            fi
        else
            echo -e "${YELLOW}No training logs found yet.${NC}"
        fi
    else
        echo -e "${RED}Training process not running. Cannot monitor progress.${NC}"
    fi
    
    # 5. Check TensorBoard
    echo -e "\n${CYAN}[5/5] TENSORBOARD STATUS${NC}"
    check_tensorboard
    
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "Next update in $INTERVAL seconds... (Press Ctrl+C to exit)"
    echo -e "${BLUE}============================================================${NC}"
    
    sleep $INTERVAL
done
