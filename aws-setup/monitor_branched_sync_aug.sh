#!/bin/bash
# Monitor the branched model with synchronized augmentation training on EC2
# This script provides real-time monitoring of:
# - Training logs
# - CPU usage
# - Memory consumption

# EC2 instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="../aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="/home/ec2-user/emotion_training/training_branched_sync_aug.log"

echo "======================================================"
echo "  MONITORING BRANCHED MODEL WITH SYNCHRONIZED AUGMENTATION"
echo "======================================================"
echo "Target: $USERNAME@$INSTANCE_IP"
echo "Using key: $KEY_FILE"
echo "Log file: $LOG_FILE"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    exit 1
fi

# Check if process is running
echo "Checking if the training process is running..."
PROCESS_INFO=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "ps aux | grep train_branched_sync_aug.py | grep -v grep")

if [ -z "$PROCESS_INFO" ]; then
    echo "Warning: Training process is not running"
    echo "To start the training, use: ./aws-setup/deploy_branched_sync_aug.sh"
    exit 1
fi

echo "Process is running:"
echo "$PROCESS_INFO"
echo ""

# Function to display menu
display_menu() {
    echo "======================================================"
    echo "  MONITORING OPTIONS"
    echo "======================================================"
    echo "1. View last 50 lines of training log"
    echo "2. Stream training log in real-time (Ctrl+C to exit)"
    echo "3. Check system resources (CPU, memory)"
    echo "4. Check validation metrics" 
    echo "5. Check training time estimate"
    echo "6. Exit"
    echo "======================================================"
    echo "Enter your choice (1-6): "
}

# Main monitoring loop
while true; do
    display_menu
    read -r choice

    case $choice in
        1)
            echo "Fetching last 50 lines of training log..."
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -n 50 $LOG_FILE"
            ;;
        2)
            echo "Streaming training log in real-time (Ctrl+C to exit)..."
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -f $LOG_FILE"
            ;;
        3)
            echo "Checking system resources..."
            echo "CPU Usage:"
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "top -bn1 | head -20"
            echo ""
            echo "Memory Usage:"
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "free -h"
            echo ""
            echo "Disk Usage:"
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "df -h"
            ;;
        4)
            echo "Checking validation metrics..."
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'val_accuracy' $LOG_FILE | tail -n 10"
            echo ""
            echo "Best validation metrics:"
            ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'Best validation accuracy' $LOG_FILE | tail -n 1"
            ;;
        5)
            echo "Checking training time estimate..."
            EPOCH_TIMES=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -a 'Epoch [0-9]' $LOG_FILE | grep -o '[0-9]\+s' | grep -o '[0-9]\+'")
            
            if [ -z "$EPOCH_TIMES" ]; then
                echo "No training epoch data available yet."
            else
                # Calculate average epoch time
                TOTAL=0
                COUNT=0
                for time in $EPOCH_TIMES; do
                    TOTAL=$((TOTAL + time))
                    COUNT=$((COUNT + 1))
                done
                
                if [ $COUNT -gt 0 ]; then
                    AVG=$((TOTAL / COUNT))
                    echo "Average epoch time: $AVG seconds"
                    echo "Estimated total training time (50 epochs): $((AVG * 50 / 60)) minutes"
                    echo "Estimated remaining time: $((AVG * (50 - COUNT) / 60)) minutes"
                else
                    echo "No training epoch data available yet."
                fi
            fi
            ;;
        6)
            echo "Exiting monitor..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read -r
    clear
done
