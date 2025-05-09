#!/bin/bash
# Enhanced script to monitor training progress with better formatting

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m   ENHANCED TRAINING MONITOR - COMBINED DATASETS\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# SSH to the instance and run a more sophisticated monitoring script
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'

# Function to extract dataset info from the log
function show_datasets_info() {
    echo -e "\033[1;33m=== DATASETS INFORMATION ===\033[0m"
    
    # Get RAVDESS samples count
    RAVDESS_COUNT=$(grep "Added RAVDESS" ~/emotion_training/training.log | tail -1 | grep -oP '\d+(?= samples)')
    if [ ! -z "$RAVDESS_COUNT" ]; then
        echo -e "\033[1;32m✓ RAVDESS:\033[0m $RAVDESS_COUNT samples"
    else
        echo -e "\033[1;31m✗ RAVDESS: Not loaded\033[0m"
    fi
    
    # Get CREMA-D samples count
    CREMA_COUNT=$(grep "Added CREMA-D" ~/emotion_training/training.log | tail -1 | grep -oP '\d+(?= samples)')
    if [ ! -z "$CREMA_COUNT" ]; then
        echo -e "\033[1;32m✓ CREMA-D:\033[0m $CREMA_COUNT samples" 
    else
        echo -e "\033[1;31m✗ CREMA-D: Not loaded\033[0m"
    fi
    
    # Get combined count
    TOTAL_COUNT=$(grep "Combined:" ~/emotion_training/training.log | tail -1 | grep -oP '\d+(?= total)')
    if [ ! -z "$TOTAL_COUNT" ]; then
        echo -e "\033[1;32m✓ COMBINED:\033[0m $TOTAL_COUNT total samples"
    fi
    
    echo ""
}

# Function to show the latest epoch progress
function show_latest_epoch() {
    echo -e "\033[1;33m=== LATEST TRAINING PROGRESS ===\033[0m"
    
    # Get the current epoch
    CURRENT_EPOCH=$(grep -oP "Epoch \d+/\d+" ~/emotion_training/training.log | tail -1)
    if [ ! -z "$CURRENT_EPOCH" ]; then
        echo -e "\033[1;34mCurrent: $CURRENT_EPOCH\033[0m"
        
        # Get the latest accuracy and loss
        LATEST_METRICS=$(grep -A 1 "Epoch.*val_loss" ~/emotion_training/training.log | tail -1)
        
        if [ ! -z "$LATEST_METRICS" ]; then
            # Extract values
            TRAIN_LOSS=$(echo "$LATEST_METRICS" | grep -oP "loss: \K[0-9.]+")
            TRAIN_ACC=$(echo "$LATEST_METRICS" | grep -oP "accuracy: \K[0-9.]+")
            VAL_LOSS=$(echo "$LATEST_METRICS" | grep -oP "val_loss: \K[0-9.]+")
            VAL_ACC=$(echo "$LATEST_METRICS" | grep -oP "val_accuracy: \K[0-9.]+")
            
            if [ ! -z "$TRAIN_LOSS" ] && [ ! -z "$TRAIN_ACC" ]; then
                echo -e "\033[1mTraining:   \033[0m loss = \033[1;34m$TRAIN_LOSS\033[0m, accuracy = \033[1;34m$TRAIN_ACC\033[0m"
            fi
            
            if [ ! -z "$VAL_LOSS" ] && [ ! -z "$VAL_ACC" ]; then
                VAL_ACC_PERCENT=$(awk "BEGIN {printf \"%.1f\", $VAL_ACC * 100}")
                echo -e "\033[1mValidation: \033[0m loss = \033[1;34m$VAL_LOSS\033[0m, accuracy = \033[1;34m$VAL_ACC\033[0m ($VAL_ACC_PERCENT%)"
            fi
        else
            echo "No validation metrics available yet"
        fi
    else
        echo "No epochs completed yet"
    fi
    
    echo ""
}

# Function to show training errors if any
function show_errors() {
    ERRORS=$(grep -i "error\|exception\|failed" ~/emotion_training/training.log | tail -5)
    
    if [ ! -z "$ERRORS" ]; then
        echo -e "\033[1;33m=== ERRORS ===\033[0m"
        echo -e "\033[1;31m$ERRORS\033[0m"
        echo ""
    fi
}

# Continuously monitor
while true; do
    clear
    echo -e "\033[1;36m========================================================\033[0m"
    echo -e "\033[1;36m   ENHANCED TRAINING MONITOR - COMBINED DATASETS\033[0m"
    echo -e "\033[1;36m========================================================\033[0m"
    echo -e "\033[0;90mLast updated: $(date)\033[0m"
    echo ""
    
    # Show all sections
    show_datasets_info
    show_latest_epoch
    show_errors
    
    echo -e "\033[1;36m========================================================\033[0m"
    echo -e "\033[0;90mMonitoring training log - updating every 5 seconds\033[0m"
    
    # Wait before refreshing
    sleep 5
done
EOF
