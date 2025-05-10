#!/bin/bash
# Advanced continuous monitoring script for TCN large model training (Cross-platform version)
# This script provides real-time metrics, progress tracking, and alerts
# Works on both macOS and Linux systems

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed.log"
REMOTE_DIR="~/emotion_training"
MONITORING_INTERVAL=30 # seconds between monitoring cycles
ALERT_THRESHOLD=90 # minutes without progress before alerting

# Detect OS for date compatibility
OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
fi

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    for cmd in awk grep ssh date cut; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}Error: $cmd is not installed.${NC}"
            exit 1
        fi
    done
    
    if [ ! -f "$KEY_FILE" ]; then
        echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
        echo "Please ensure the key file path is correct."
        exit 1
    fi
    
    echo -e "${GREEN}All dependencies satisfied!${NC}"
    echo -e "${YELLOW}OS detected:${NC} $OS_TYPE"
}

# Cross-platform date conversion function
convert_to_seconds() {
    local timestamp="$1"
    
    if [[ -z "$timestamp" ]]; then
        echo "0"
        return
    fi
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS date command format
        date -j -f "%Y-%m-%d %H:%M:%S" "$timestamp" +%s 2>/dev/null || echo "0"
    elif [[ "$OS_TYPE" == "linux" ]]; then
        # Linux date command format
        date -d "$timestamp" +%s 2>/dev/null || echo "0"
    else
        # Default fallback - can't compute time difference
        echo "0"
    fi
}

display_banner() {
    clear
    echo -e "${BLUE}===========================================================================${NC}"
    echo -e "${BOLD}${GREEN}    CONTINUOUS MONITORING: FIXED TCN LARGE MODEL TRAINING    ${NC}"
    echo -e "${BLUE}===========================================================================${NC}"
    echo -e "${YELLOW}Instance:${NC} $USERNAME@$INSTANCE_IP"
    echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
    echo -e "${YELLOW}Time:${NC} $(date)"
    echo -e "${BLUE}===========================================================================${NC}"
    echo ""
}

check_training_active() {
    local pid_output=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat ${REMOTE_DIR}/fixed_tcn_large_pid.txt 2>/dev/null || echo 'No PID file found'")
    
    if [[ "$pid_output" == "No PID file found" ]]; then
        echo -e "${RED}No training process PID file found.${NC}"
        return 1
    fi
    
    local process_check=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "ps -p $pid_output -o comm= 2>/dev/null || echo 'Process not running'")
    
    if [[ "$process_check" == "Process not running" ]]; then
        echo -e "${RED}Training process (PID: $pid_output) is no longer running.${NC}"
        return 1
    else
        echo -e "${GREEN}Training process active (PID: $pid_output, Process: $process_check)${NC}"
        return 0
    fi
}

get_latest_metrics() {
    # Extract the latest epoch metrics from the log file
    local metrics=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Epoch [0-9]+/|loss:|accuracy:|val_loss:|val_accuracy:' ${REMOTE_DIR}/$LOG_FILE | tail -15")
    
    # Get latest validation accuracy
    local val_acc=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'val_accuracy:' ${REMOTE_DIR}/$LOG_FILE | tail -1 | awk '{print \$NF}'")
    
    # Get best validation accuracy so far
    local best_val_acc=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'val_accuracy:' ${REMOTE_DIR}/$LOG_FILE | awk '{print \$NF}' | sort -nr | head -1")
    
    # Count total epochs completed
    local epochs_completed=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Epoch [0-9]+/[0-9]+ ' ${REMOTE_DIR}/$LOG_FILE | wc -l")
    
    # Get latest learning rate
    local learning_rate=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Learning rate set to' ${REMOTE_DIR}/$LOG_FILE | tail -1 | awk '{print \$NF}'")
    
    # Get starting time
    local start_time=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Starting optimized balanced model training' ${REMOTE_DIR}/$LOG_FILE | head -1 | awk '{print \$1, \$2}'")
    
    # Check if there are any errors
    local error_count=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'ERROR|Error|Exception|Traceback' ${REMOTE_DIR}/$LOG_FILE | wc -l")
    
    # Get last 3 progress timestamps to check if training is still progressing
    local last_timestamps=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Epoch [0-9]+/[0-9]+ ' ${REMOTE_DIR}/$LOG_FILE | tail -3 | awk '{print \$1, \$2}'")
    
    # Return all metrics as a string
    echo -e "METRICS:$metrics\nVAL_ACC:$val_acc\nBEST_VAL_ACC:$best_val_acc\nEPOCHS:$epochs_completed\nLR:$learning_rate\nSTART:$start_time\nERRORS:$error_count\nTIMESTAMPS:$last_timestamps"
}

check_system_status() {
    # Get CPU, memory, and GPU utilization from the remote instance
    local cpu_usage=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "top -bn1 | grep 'Cpu(s)' | awk '{print \$2 + \$4}' | cut -d. -f1")
    local mem_usage=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "free -m | awk 'NR==2{printf \"%.1f%%\", \$3*100/\$2}'")
    local disk_usage=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "df -h | awk '\$NF==\"/\"{printf \"%s\", \$5}'")
    
    # Return system stats
    echo "CPU: ${cpu_usage}% | Memory: ${mem_usage} | Disk: ${disk_usage}"
}

display_training_summary() {
    local metrics="$1"
    
    # Extract individual metrics
    local val_acc=$(echo "$metrics" | grep "VAL_ACC:" | cut -d':' -f2)
    local best_val_acc=$(echo "$metrics" | grep "BEST_VAL_ACC:" | cut -d':' -f2)
    local epochs=$(echo "$metrics" | grep "EPOCHS:" | cut -d':' -f2)
    local lr=$(echo "$metrics" | grep "LR:" | cut -d':' -f2)
    local start=$(echo "$metrics" | grep "START:" | cut -d':' -f2)
    local errors=$(echo "$metrics" | grep "ERRORS:" | cut -d':' -f2)
    
    # Calculate elapsed time if start time is available
    local elapsed="Unknown"
    if [[ -n "$start" ]]; then
        local start_seconds=$(convert_to_seconds "$start")
        if [[ "$start_seconds" != "0" ]]; then
            local current_seconds=$(date +%s)
            local elapsed_seconds=$((current_seconds - start_seconds))
            local elapsed_hours=$((elapsed_seconds / 3600))
            local elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 ))
            elapsed="${elapsed_hours}h ${elapsed_minutes}m"
        fi
    fi
    
    # Display summary
    echo -e "${CYAN}${BOLD}Training Summary:${NC}"
    echo -e "${YELLOW}Epochs Completed:${NC} $epochs/125"
    echo -e "${YELLOW}Current Val Accuracy:${NC} $val_acc"
    echo -e "${YELLOW}Best Val Accuracy:${NC} ${GREEN}$best_val_acc${NC}"
    echo -e "${YELLOW}Current Learning Rate:${NC} $lr"
    echo -e "${YELLOW}Training Duration:${NC} $elapsed"
    
    if [[ "$errors" -gt "0" ]]; then
        echo -e "${RED}Warning: $errors errors detected in log${NC}"
    else
        echo -e "${GREEN}No errors detected${NC}"
    fi
    
    # Get system status
    local system_status=$(check_system_status)
    echo -e "${YELLOW}System Status:${NC} $system_status"
}

display_recent_progress() {
    local metrics="$1"
    
    # Extract recent metrics
    local recent_metrics=$(echo "$metrics" | grep "METRICS:" | cut -d':' -f2-)
    
    echo -e "${CYAN}${BOLD}Recent Training Progress:${NC}"
    echo "$recent_metrics" | grep -E 'Epoch [0-9]+/|val_accuracy:'
    echo ""
}

check_progress_stall() {
    local metrics="$1"
    local timestamps=$(echo "$metrics" | grep "TIMESTAMPS:" | cut -d':' -f2-)
    
    if [[ -z "$timestamps" ]]; then
        return 0  # No timestamps, can't check for stall
    fi
    
    # Get the last timestamp
    local last_timestamp=$(echo "$timestamps" | tail -1)
    local last_timestamp_seconds=$(convert_to_seconds "$last_timestamp")
    
    if [[ "$last_timestamp_seconds" == "0" ]]; then
        return 0  # Couldn't parse timestamp
    fi
    
    local current_seconds=$(date +%s)
    local elapsed_seconds=$((current_seconds - last_timestamp_seconds))
    local elapsed_minutes=$((elapsed_seconds / 60))
    
    if [[ "$elapsed_minutes" -gt "$ALERT_THRESHOLD" ]]; then
        echo -e "${RED}${BOLD}ALERT: No progress for $elapsed_minutes minutes!${NC}"
        return 1
    fi
    
    return 0
}

show_tail_log() {
    echo -e "${CYAN}${BOLD}Last 10 lines of log:${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -10 ${REMOTE_DIR}/$LOG_FILE"
    echo ""
}

monitor_training() {
    local continuous=true
    local cycle_count=0
    
    while $continuous; do
        display_banner
        
        # Check if the training process is still active
        check_training_active
        training_active=$?
        
        if [[ "$training_active" -eq 1 ]]; then
            echo -e "${RED}Training process appears to have stopped. Checking log for completion...${NC}"
            # Check if training completed successfully
            local completed=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -E 'Training completed in|Final model saved to:' ${REMOTE_DIR}/$LOG_FILE | wc -l")
            
            if [[ "$completed" -gt 1 ]]; then
                echo -e "${GREEN}${BOLD}Training completed successfully!${NC}"
                show_tail_log
                continuous=false
                break
            else
                echo -e "${RED}${BOLD}Training appears to have crashed or stopped unexpectedly.${NC}"
                show_tail_log
                read -p "Continue monitoring? (y/n): " continue_choice
                if [[ "${continue_choice,,}" != "y" ]]; then
                    continuous=false
                    break
                fi
            fi
        fi
        
        # Get the latest metrics
        local metrics=$(get_latest_metrics)
        
        # Display training summary
        display_training_summary "$metrics"
        echo ""
        
        # Display recent progress
        display_recent_progress "$metrics"
        
        # Check for progress stall
        check_progress_stall "$metrics"
        
        # Show tail of log file
        show_tail_log
        
        # Increment cycle count and show monitoring info
        ((cycle_count++))
        echo -e "${BLUE}Monitoring cycle: $cycle_count | Refreshing every ${MONITORING_INTERVAL}s | Press Ctrl+C to stop${NC}"
        
        # Wait for next monitoring cycle
        sleep $MONITORING_INTERVAL
    done
}

# Main execution
check_dependencies
monitor_training

echo -e "${GREEN}${BOLD}Monitoring complete.${NC}"
echo -e "${BLUE}===========================================================================${NC}"
