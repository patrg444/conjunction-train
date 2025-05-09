#!/bin/bash
# All-in-one tool for monitoring and analyzing TCN model training

echo "=============================================================================="
echo "  BRANCHED REGULARIZATION SYNC AUG TCN MODEL TRACKING TOOL"
echo "=============================================================================="

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
# Default to regular model
MODEL_TYPE="regular"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: SSH key file not found: $KEY_FILE"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Function to get log file based on model type
get_log_file() {
    if [ "$MODEL_TYPE" == "large" ]; then
        LOG_FILE="training_branched_regularization_sync_aug_tcn_large.log"
    else
        LOG_FILE="training_branched_regularization_sync_aug_tcn.log"
    fi
    echo "Using log file: $LOG_FILE for $MODEL_TYPE model"
}

# Set initial log file
get_log_file

# Function to show model selection menu
select_model() {
    echo
    echo "Select model type to track:"
    echo "1) Regular TCN model"
    echo "2) Large TCN model"
    echo
    echo -n "Enter your choice (1/2): "
    read model_choice
    
    case "$model_choice" in
        1)
            MODEL_TYPE="regular"
            ;;
        2)
            MODEL_TYPE="large"
            ;;
        *)
            echo "Invalid choice. Defaulting to regular model."
            MODEL_TYPE="regular"
            ;;
    esac
    
    get_log_file
}

# Function to show options menu
show_menu() {
    echo
    echo "Please select an option:"
    echo "0) Switch model type (current: $MODEL_TYPE)"
    echo "1) Monitor live training logs"
    echo "2) Download training logs"
    echo "3) Analyze training progress"
    echo "4) Download and analyze (2+3)"
    echo "q) Quit"
    echo
    echo -n "Enter your choice: "
}

# Function to monitor live logs
monitor_logs() {
    echo "Starting continuous monitoring of $MODEL_TYPE model... (Press Ctrl+C to stop)"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -f ~/emotion_training/$LOG_FILE"
}

# Function to download logs
download_logs() {
    echo "Downloading training logs for $MODEL_TYPE model..."
    scp -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP:~/emotion_training/$LOG_FILE" "./$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Log file downloaded successfully."
    else
        echo "Error: Failed to download log file."
        return 1
    fi
}

# Function to analyze logs
analyze_logs() {
    if [ ! -f "./$LOG_FILE" ]; then
        echo "Error: Log file not found. Please download the logs first."
        return 1
    fi
    
    echo "Analyzing training progress..."
    ./extract_tcn_model_progress.py "./$LOG_FILE"
}

# Main loop
while true; do
    show_menu
    read choice
    
    case "$choice" in
        0)
            select_model
            ;;
        1)
            echo "=============================================================================="
            echo "  LIVE MONITORING - $MODEL_TYPE TCN MODEL TRAINING"
            echo "=============================================================================="
            monitor_logs
            ;;
        2)
            echo "=============================================================================="
            echo "  DOWNLOADING LOGS - $MODEL_TYPE TCN MODEL TRAINING"
            echo "=============================================================================="
            download_logs
            ;;
        3)
            echo "=============================================================================="
            echo "  ANALYZING PROGRESS - $MODEL_TYPE TCN MODEL TRAINING"
            echo "=============================================================================="
            analyze_logs
            ;;
        4)
            echo "=============================================================================="
            echo "  DOWNLOADING AND ANALYZING - $MODEL_TYPE TCN MODEL TRAINING"
            echo "=============================================================================="
            download_logs && analyze_logs
            ;;
        q|Q)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done
