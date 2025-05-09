#!/bin/bash
# Enhanced monitoring script for SlowFast training with formatting
# This script provides a clean, formatted view of training progress metrics

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
LOG_PATH="/home/ubuntu/monitor_logs/slowfast_training_stream.log"

# Define terminal colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print header
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}          SLOWFAST EMOTION RECOGNITION MONITOR            ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e ""
echo -e "${CYAN}Connecting to EC2 instance...${NC}"
echo -e "${CYAN}Training metrics will be color-coded and formatted for readability${NC}"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
echo -e ""

# Function to extract and format metrics line
format_line() {
    line="$1"
    
    # Format epoch headers
    if [[ $line == *"Epoch"* && $line == *"/"* ]]; then
        echo -e "${BOLD}${BLUE}$line${NC}"
        return
    fi
    
    # Format training progress
    if [[ $line == *"Training:"* ]]; then
        # Extract percentage
        pct=$(echo "$line" | grep -o '[0-9]*%' | head -1)
        
        # Extract metrics
        metrics=$(echo "$line" | grep -o 'loss=[0-9.]*' | sed 's/loss=/Loss: /')
        acc=$(echo "$line" | grep -o 'acc=[0-9.]*' | sed 's/acc=/Accuracy: /')
        
        if [[ ! -z "$pct" && ! -z "$metrics" && ! -z "$acc" ]]; then
            echo -e "Training: ${pct} - ${YELLOW}${metrics}${NC}, ${GREEN}${acc}%${NC}"
        else
            echo -e "$line"
        fi
        return
    fi
    
    # Format validation progress
    if [[ $line == *"Validation:"* ]]; then
        echo -e "${CYAN}$line${NC}"
        return
    fi
    
    # Format training/validation final results
    if [[ $line == *"Train Loss:"* || $line == *"Val Loss:"* ]]; then
        # Extract metrics
        loss=$(echo "$line" | grep -o 'Loss: [0-9.]*' | cut -d' ' -f2)
        acc=$(echo "$line" | grep -o 'Acc: [0-9.]*' | cut -d' ' -f2)
        
        if [[ ! -z "$loss" && ! -z "$acc" ]]; then
            if [[ $line == *"Train Loss:"* ]]; then
                echo -e "${BOLD}Train Results: ${YELLOW}Loss: $loss${NC}, ${GREEN}Accuracy: $acc%${NC}"
            else
                echo -e "${BOLD}${BLUE}Val Results:   ${YELLOW}Loss: $loss${NC}, ${GREEN}Accuracy: $acc%${NC}"
            fi
        else
            echo -e "$line"
        fi
        return
    fi
    
    # Format best validation results
    if [[ $line == *"New best validation accuracy:"* ]]; then
        acc=$(echo "$line" | grep -o '[0-9.]*%')
        echo -e "${BOLD}${GREEN}$line${NC}"
        return
    fi
    
    # Format no improvement lines
    if [[ $line == *"No improvement for"* ]]; then
        echo -e "${RED}$line${NC}"
        return
    fi
    
    # Default formatting for other lines
    echo "$line"
}

# Main monitoring loop
ssh -i $KEY $EC2_HOST "tail -f $LOG_PATH" | while read -r line; do
    format_line "$line"
done
