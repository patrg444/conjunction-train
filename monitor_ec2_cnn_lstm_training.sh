#!/bin/bash

# This script helps monitor the CNN-LSTM model training on EC2

# Get EC2 IP from the config file
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/emotion_project"

# Function to find the latest log file
find_latest_log() {
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $REMOTE_DIR -name 'cnn_lstm_training_*.log' -type f | sort -r | head -n1"
}

# Function to check GPU usage
check_gpu_usage() {
    echo "Checking GPU usage..."
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "nvidia-smi"
    echo ""
}

# Function to stream log file
stream_log() {
    local LOG_FILE=$1
    if [[ -z "$LOG_FILE" ]]; then
        LOG_FILE=$(find_latest_log)
    fi
    
    if [[ -z "$LOG_FILE" ]]; then
        echo "No training log file found. Has training started?"
        return 1
    fi
    
    echo "Streaming log file: $LOG_FILE"
    echo "Press Ctrl+C to exit"
    echo "======================================"
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "tail -f $LOG_FILE"
}

# Function to show model summary
show_model_summary() {
    echo "Checking for trained models..."
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $REMOTE_DIR/models -name 'cnn_lstm_fixed_*' -type d | sort -r"
    
    # Check if training is complete or still running
    LOG_FILE=$(find_latest_log)
    if [[ -n "$LOG_FILE" ]]; then
        echo ""
        echo "Checking training status..."
        ssh -i "$SSH_KEY" ubuntu@$EC2_IP "grep -E 'Success|Error occurred' $LOG_FILE || echo 'Training still in progress'"
        
        # Show final model validation accuracy if available
        echo ""
        echo "Looking for validation accuracy results..."
        ssh -i "$SSH_KEY" ubuntu@$EC2_IP "grep -A 3 'Model Performance Comparison' $LOG_FILE"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "======= CNN-LSTM Training Monitor ======="
    echo "1. Check GPU usage"
    echo "2. Stream training log"
    echo "3. Show model summary and results"
    echo "4. Exit"
    echo "========================================"
    read -p "Select an option (1-4): " option
    
    case $option in
        1) check_gpu_usage; show_menu ;;
        2) stream_log; show_menu ;;
        3) show_model_summary; show_menu ;;
        4) echo "Exiting."; exit 0 ;;
        *) echo "Invalid option"; show_menu ;;
    esac
}

# Check if EC2 instance is accessible
echo "Checking connection to EC2 instance at $EC2_IP..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 ubuntu@$EC2_IP "echo 'Connected!'"; then
    echo "Connection successful!"
    show_menu
else
    echo "Failed to connect to EC2 instance. Check your connection or instance status."
    exit 1
fi
