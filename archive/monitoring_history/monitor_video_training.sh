#!/bin/bash
# Monitor the video training process
# This script:
# 1. Checks if the training session is running
# 2. Shows GPU status (if available)
# 3. Shows the training logs

SESSION_NAME="video_training"
REFRESH_INTERVAL=10  # seconds

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if session exists
check_session() {
    if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
        echo -e "${GREEN}Training session '${SESSION_NAME}' is running.${NC}"
        return 0
    else
        echo -e "${RED}Training session '${SESSION_NAME}' is not running.${NC}"
        echo "You can start it with: ./launch_video_full_training.sh"
        return 1
    fi
}

# Show GPU status if available
show_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n${CYAN}=== GPU Status ===${NC}"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv
    else
        echo -e "\n${YELLOW}No GPU detected.${NC}"
    fi
}

# Show disk space
show_disk_space() {
    echo -e "\n${CYAN}=== Disk Space ===${NC}"
    df -h /home/ubuntu | tail -n 1
}

# Show top processes
show_top_processes() {
    echo -e "\n${CYAN}=== Top Processes ===${NC}"
    ps aux | grep -E 'python|Python' | grep -v grep | sort -rn -k 3 | head -5
}

# Get the training logs
get_training_logs() {
    echo -e "\n${CYAN}=== Training Logs (last 20 lines) ===${NC}"
    # Capture tmux content to a temp file and show the last 20 lines
    tmux capture-pane -pt ${SESSION_NAME} -S - | tail -20
}

# Main monitoring loop
monitor() {
    clear
    while true; do
        clear
        echo -e "${CYAN}====== Video Training Monitor ======${NC}"
        echo -e "${CYAN}$(date)${NC}"
        
        if ! check_session; then
            break
        fi
        
        show_gpu_status
        show_disk_space
        show_top_processes
        get_training_logs
        
        echo -e "\n${YELLOW}Press Ctrl+C to stop monitoring.${NC}"
        echo -e "${YELLOW}Refreshing in ${REFRESH_INTERVAL} seconds...${NC}"
        sleep ${REFRESH_INTERVAL}
    done
}

# Function to capture and save logs
save_logs() {
    if ! check_session; then
        return 1
    fi
    
    LOG_FILE="video_training_$(date +%Y%m%d_%H%M%S).log"
    echo -e "Capturing full logs to ${LOG_FILE}..."
    
    # Capture the entire tmux history
    tmux capture-pane -pt ${SESSION_NAME} -S - > ${LOG_FILE}
    
    echo -e "${GREEN}Logs saved to ${LOG_FILE}${NC}"
    return 0
}

# Parse command line arguments
case "$1" in
    logs)
        save_logs
        ;;
    gpu)
        show_gpu_status
        ;;
    *)
        monitor
        ;;
esac

exit 0
