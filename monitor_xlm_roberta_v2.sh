#!/bin/bash
# Enhanced monitoring script for XLM-RoBERTa v2 training

# Configuration
LOG_DIR="training_logs_humor"
EXP_NAME="xlm-roberta-large_optimized"
TENSORBOARD_PORT=6006
TAIL_LINES=50

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== XLM-RoBERTa v2 Training Monitor ===${NC}"
echo -e "${YELLOW}Monitoring experiment: ${EXP_NAME}${NC}"

# Check if training logs directory exists
if [ ! -d "${LOG_DIR}/${EXP_NAME}" ]; then
    echo -e "${RED}Training logs directory not found: ${LOG_DIR}/${EXP_NAME}${NC}"
    echo "Make sure training has started and the log directory exists."
    exit 1
fi

# Function to display most recent metrics
display_metrics() {
    echo -e "\n${BLUE}=== Recent Training Metrics ===${NC}"
    
    # Find the most recent metrics log file
    METRICS_FILE=$(find ${LOG_DIR}/${EXP_NAME} -name "*.log" -type f -print | sort -r | head -n 1)
    
    if [ -z "$METRICS_FILE" ]; then
        echo -e "${YELLOW}No metrics log file found. Training may not have started yet.${NC}"
    else
        echo -e "${CYAN}Latest metrics from: ${METRICS_FILE}${NC}"
        echo -e "${GREEN}--- Training Metrics ---${NC}"
        grep "train_loss\|train_acc" "$METRICS_FILE" | tail -n $TAIL_LINES
        
        echo -e "\n${GREEN}--- Validation Metrics ---${NC}"
        grep "val_loss\|val_acc\|val_f1\|val_precision\|val_recall" "$METRICS_FILE" | tail -n $TAIL_LINES
    fi
}

# Function to display checkpoints
display_checkpoints() {
    echo -e "\n${BLUE}=== Saved Checkpoints ===${NC}"
    CHECKPOINT_DIR="${LOG_DIR}/${EXP_NAME}/checkpoints"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo -e "${YELLOW}No checkpoints directory found yet.${NC}"
    else
        echo -e "${CYAN}Checkpoints saved at: ${CHECKPOINT_DIR}${NC}"
        ls -lh "$CHECKPOINT_DIR" | grep -v "total" | awk '{print $9, "("$5")"}'
    fi
}

# Function to display TensorBoard instructions
display_tensorboard() {
    echo -e "\n${BLUE}=== TensorBoard ===${NC}"
    echo -e "${GREEN}To start TensorBoard, run:${NC}"
    echo -e "tensorboard --logdir=${LOG_DIR}/${EXP_NAME} --port=${TENSORBOARD_PORT}"
    echo -e "${GREEN}Then open in browser:${NC} http://localhost:${TENSORBOARD_PORT}"
}

# Function to display system status
display_system_status() {
    echo -e "\n${BLUE}=== System Status ===${NC}"
    echo -e "${CYAN}GPU Usage:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    else
        echo "nvidia-smi not available - GPU status unknown"
    fi
    
    echo -e "\n${CYAN}CPU and Memory Usage:${NC}"
    top -bn1 | head -5
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BLUE}=== XLM-RoBERTa v2 Training Monitor ===${NC}"
    echo -e "${YELLOW}Monitoring experiment: ${EXP_NAME}${NC}"
    echo -e "${YELLOW}Time: $(date)${NC}"
    
    display_metrics
    display_checkpoints
    display_tensorboard
    display_system_status
    
    echo -e "\n${BLUE}=== Control Options ===${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit monitoring${NC}"
    
    sleep 30
done
