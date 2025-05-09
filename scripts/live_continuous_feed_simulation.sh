#!/bin/bash
# This script simulates a continuous feed from AWS while the real deployment completes
# It provides a visual indication of progress similar to what you would see with 
# the live_continuous_monitor.sh script once deployment completes

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Clear screen and set cursor to home position
clear_screen() {
    printf "\033c"
}

# Print centered text with color
print_centered() {
    local text="$1"
    local color="$2"
    local width=$(tput cols)
    local padding=$(( (width - ${#text}) / 2 ))
    printf "%${padding}s" ""
    echo -e "${color}${text}${NC}"
}

# Print a horizontal line
print_line() {
    local width=$(tput cols)
    local line=$(printf "%${width}s" | tr " " "=")
    echo -e "${BLUE}${line}${NC}"
}

# Print banner
print_banner() {
    clear_screen
    print_line
    print_centered "REAL-TIME SIMULATED LSTM ATTENTION MODEL TRAINING" "${BOLD}${MAGENTA}"
    print_centered "AWS CONTINUOUS MONITORING SIMULATION" "${BOLD}${MAGENTA}"
    print_line
    echo ""
}

# Simulated progress data
training_steps=200
current_step=0
epochs=15
current_epoch=1
gpu_utilization=()
for i in {1..100}; do
    gpu_utilization+=($((RANDOM % 40 + 40)))  # Random values between 40-80%
done

# Function to simulate live output
simulate_training_output() {
    clear_screen
    print_banner
    
    # Simulate multiple panes
    
    # Pane 1: Training Log
    echo -e "${CYAN}=== TRAINING LOG ===${NC}"
    echo -e "${GREEN}Epoch $current_epoch/$epochs${NC}"
    
    # Show last few log entries
    for ((i = max(0, current_step-10); i <= current_step; i++)); do
        if [ $i -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training with batch size 16, loss=focal"
        elif [ $i -eq 1 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Model compiled with Adam optimizer (lr=0.0005)"
        elif [ $i -eq 2 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Loading dataset with proper masking for variable-length sequences"
        elif [ $i -eq 3 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Found 1440 training samples, 360 validation samples"
        elif [ $i -eq $current_step ]; then
            loss=$(echo "scale=4; 1.0 - ($current_epoch * 0.1) - ($i / $training_steps * 0.2) + (0.$RANDOM / 10000)" | bc)
            acc=$(echo "scale=4; 0.4 + ($current_epoch * 0.07) + ($i / $training_steps * 0.15) + (0.$RANDOM / 10000)" | bc)
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Epoch $current_epoch, Step $i/$training_steps - loss: $loss, accuracy: $acc"
        elif [ $(($i % 20)) -eq 0 ]; then
            loss=$(echo "scale=4; 1.0 - ($current_epoch * 0.1) - ($i / $training_steps * 0.2) + (0.$RANDOM / 10000)" | bc)
            acc=$(echo "scale=4; 0.4 + ($current_epoch * 0.07) + ($i / $training_steps * 0.15) + (0.$RANDOM / 10000)" | bc)
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Epoch $current_epoch, Step $i/$training_steps - loss: $loss, accuracy: $acc"
        fi
    done
    echo ""
    
    # Pane 2: GPU Metrics
    echo -e "${MAGENTA}=== GPU METRICS ===${NC}"
    util_idx=$((current_step % ${#gpu_utilization[@]}))
    memory_used=$((3000 + RANDOM % 5000))
    echo "GPU Utilization: ${gpu_utilization[$util_idx]}%"
    echo "Memory Used: $memory_used MB / 16000 MB"
    echo "GPU Temperature: $((50 + RANDOM % 10))°C"
    echo ""
    
    # Pane 3: System Resources
    echo -e "${YELLOW}=== SYSTEM RESOURCES ===${NC}"
    echo "CPU Usage: $((20 + RANDOM % 30))%"
    echo "Memory: $((6 + RANDOM % 4))GB / 16GB"
    echo "Disk Space: 32GB / 100GB used"
    echo "Active Python Processes: 2"
    echo ""
    
    # Pane 4: Model Directory
    echo -e "${GREEN}=== MODEL DIRECTORY ===${NC}"
    if [ $current_epoch -gt 1 ]; then
        echo "- models/"
        echo "  - attention_focal_loss/"
        for ((e=1; e<current_epoch; e++)); do
            echo "    - model_epoch_$e.keras"
        done
        if [ $current_step -eq $training_steps ]; then
            echo "    - model_epoch_$current_epoch.keras"
        fi
        if [ $current_epoch -gt 5 ]; then
            echo "    - model_best.keras"
        fi
    else
        echo "No model checkpoints saved yet."
    fi
    echo ""
    
    # Progress bar
    echo -e "${CYAN}Overall Progress:${NC}"
    total_steps=$((epochs * training_steps))
    current_total_step=$(( (current_epoch-1) * training_steps + current_step ))
    percent=$((current_total_step * 100 / total_steps))
    
    # Create progress bar
    width=50
    filled_width=$((width * percent / 100))
    bar=$(printf "%${filled_width}s" | tr " " "#")
    empty=$(printf "%$((width - filled_width))s" | tr " " "-")
    echo -e "[${GREEN}${bar}${NC}${empty}] $percent%"
    
    # Deployment status note
    echo ""
    echo -e "${RED}NOTE: This is a simulation while AWS deployment completes.${NC}"
    echo -e "${RED}Run './scripts/check_deployment_status.sh' to check the real deployment status.${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    # Update steps for next iteration
    current_step=$((current_step + 1))
    if [ $current_step -gt $training_steps ]; then
        current_step=0
        current_epoch=$((current_epoch + 1))
    fi
    
    if [ $current_epoch -gt $epochs ]; then
        clear_screen
        print_banner
        echo -e "${GREEN}Simulation complete!${NC}"
        echo ""
        echo -e "${CYAN}This was a simulation of the continuous monitoring.${NC}"
        echo -e "${CYAN}To check the real AWS deployment status, run:${NC}"
        echo "  ./scripts/check_deployment_status.sh"
        echo ""
        echo -e "${CYAN}Once deployment completes, you can use:${NC}"
        echo "  ./aws-setup/live_continuous_monitor.sh all"
        echo -e "${CYAN}to see the real continuous feed from AWS.${NC}"
        echo ""
        print_line
        exit 0
    fi
    
    sleep 0.3  # Update frequency
}

# Main loop
print_banner
echo -e "${YELLOW}Starting simulated continuous feed...${NC}"
echo -e "${RED}Press Ctrl+C to exit${NC}"
sleep 2

# Trap Ctrl+C to exit cleanly
trap "clear_screen; echo -e '${GREEN}Simulation stopped.${NC}'; exit 0" INT

# Run continuously
while true; do
    simulate_training_output
done
