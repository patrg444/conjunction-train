#!/bin/bash
# Comprehensive script to retest TCN models with improved logging and validation accuracy tracking
# Simplified version: Removes code appending, relies on base script's __main__ block.

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# AWS instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="/home/ec2-user/emotion_training" # Use explicit path

# Timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_VERSION="v${TIMESTAMP}"
# --- CONFIGURATION START --- (This section will be modified before each run)
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed_${MODEL_VERSION}.log"
TRAINING_SCRIPT="scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py"
MODEL_DIR_NAME="branched_regularization_sync_aug_tcn_large_fixed" # Used for Python script env var
# --- CONFIGURATION END ---
REMOTE_TRAINING_SCRIPT_PATH="${REMOTE_DIR}/scripts/$(basename $TRAINING_SCRIPT)"

# Ensure key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Function to detect OS for cross-platform compatibility
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)
echo -e "${BLUE}Detected operating system: ${YELLOW}$OS_TYPE${NC}"

# Function to validate SSH connection
validate_ssh_connection() {
    echo -e "${YELLOW}Validating SSH connection to AWS instance...${NC}"

    if ssh -i "$KEY_FILE" -o ConnectTimeout=5 "$USERNAME@$INSTANCE_IP" "echo 'Connection successful'" &>/dev/null; then
        echo -e "${GREEN}SSH connection successful!${NC}"
        return 0
    else
        echo -e "${RED}Failed to connect to AWS instance.${NC}"
        echo -e "${RED}Please check your connection and SSH key.${NC}"
        return 1
    fi
}

# Function to stop any existing training process
stop_existing_training() {
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${YELLOW}    STOPPING ANY EXISTING TRAINING PROCESSES    ${NC}"
    echo -e "${BLUE}=================================================================${NC}"

    # First try to stop using PID files
    for pid_file in fixed_tcn_large_pid.txt branched_regularization_sync_aug_tcn_large_pid.txt fixed_tcn_large_*_pid.txt; do
        ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
            if [ -f ${REMOTE_DIR}/${pid_file} ]; then
                PID=\$(cat ${REMOTE_DIR}/${pid_file})
                echo 'Stopping process from ${pid_file} (PID: '\$PID')'
                kill -9 \$PID 2>/dev/null || true
                rm ${REMOTE_DIR}/${pid_file} 2>/dev/null || true
            fi
        "
    done

    # Now use more aggressive approach to make sure we clean everything
    echo -e "${YELLOW}Ensuring all Python training processes are terminated...${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
        ps aux | grep 'python.*train_branched_regularization_sync_aug_tcn' | grep -v grep | awk '{print \$2}' | xargs -r kill -9
    "

    # Verify no processes are still running
    VERIFY_COMMAND="ps aux | grep 'train_branched_regularization_sync_aug_tcn' | grep -v grep"
    RUNNING_PROCESSES=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$VERIFY_COMMAND")

    if [ -z "$RUNNING_PROCESSES" ]; then
        echo -e "${GREEN}All training processes successfully terminated.${NC}"
    else
        echo -e "${RED}Warning: Some processes may still be running:${NC}"
        echo "$RUNNING_PROCESSES"
        echo -e "${YELLOW}Attempting force termination of all Python processes...${NC}"

        ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
            pkill -9 python || true
        "

        echo -e "${GREEN}All Python processes terminated.${NC}"
    fi

    echo -e "${BLUE}=================================================================${NC}"
}

# Function to deploy TCN model with enhanced logging
deploy_model() {
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${GREEN}    DEPLOYING TCN MODEL VARIANT WITH ENHANCED LOGGING ($MODEL_VERSION)    ${NC}"
    echo -e "${GREEN}    Using script: $TRAINING_SCRIPT    ${NC}"
    echo -e "${BLUE}=================================================================${NC}"

    # Make sure the script is executable locally (though not strictly necessary for scp)
    chmod +x "$TRAINING_SCRIPT"

    echo -e "${YELLOW}Transferring training script to AWS instance...${NC}"
    scp -i "$KEY_FILE" "$TRAINING_SCRIPT" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

    # Create the simplified run script locally
    cat > run_retest_script.sh << EOL
#!/bin/bash

# Error handling
set -e

cd ~/emotion_training

# Get version and script path from arguments
MODEL_VERSION="\$1"
LOG_FILE="\$2"
REMOTE_TRAINING_SCRIPT_PATH="\$3" # Get the script path as an argument
MODEL_DIR_NAME_ARG="\$4" # Get model directory name

echo "Setting up environment for model version: \$MODEL_VERSION"
echo "Using training script: \$REMOTE_TRAINING_SCRIPT_PATH"
echo "Log file: \$LOG_FILE"
echo "Model directory name: \$MODEL_DIR_NAME_ARG"

# No code appending needed here anymore

echo "Setting permissions..."
chmod +x "\$REMOTE_TRAINING_SCRIPT_PATH"

# Start the training with full logging
echo "Starting TCN large model training with version \$MODEL_VERSION..."
echo "\$(date): Starting training for model version \$MODEL_VERSION using script \$REMOTE_TRAINING_SCRIPT_PATH" > "\$LOG_FILE"
echo "Environment information:" >> "\$LOG_FILE"
python3 --version >> "\$LOG_FILE" 2>&1
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" >> "\$LOG_FILE" 2>&1
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')" >> "\$LOG_FILE" 2>&1

# Create empty files for logging validation accuracy (version specific)
# These might be created by the Python script itself now, but touching them is harmless
touch "val_accuracy_log_\${MODEL_VERSION}.txt" 2>/dev/null || true
touch "best_val_accuracy_\${MODEL_VERSION}.txt" 2>/dev/null || true

# Export variables needed by the Python script's __main__ block (if it uses them)
export MODEL_VERSION_ENV="\$MODEL_VERSION"
export MODEL_DIR_NAME_ENV="\$MODEL_DIR_NAME_ARG"

# Start the training process using the specified script
nohup python3 "\$REMOTE_TRAINING_SCRIPT_PATH" >> "\$LOG_FILE" 2>&1 &

# Save the PID (version specific)
PID_FILE="fixed_tcn_large_\${MODEL_VERSION}_pid.txt"
echo \$! > "\$PID_FILE"
echo "Training process started with PID \$(cat "\$PID_FILE")"
echo "Log file: \$LOG_FILE"
echo "PID file: \$PID_FILE"

echo "Deployment completed successfully!"
EOL

    # Transfer the run script
    echo -e "${YELLOW}Transferring run script to AWS instance...${NC}"
    scp -i "$KEY_FILE" run_retest_script.sh "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/"

    # Run the deployment, passing the remote script path and model dir name
    echo -e "${YELLOW}Running model deployment on AWS instance...${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${REMOTE_DIR}/run_retest_script.sh && ${REMOTE_DIR}/run_retest_script.sh '$MODEL_VERSION' '$LOG_FILE' '$REMOTE_TRAINING_SCRIPT_PATH' '$MODEL_DIR_NAME'"

    # Clean up local temporary file
    rm run_retest_script.sh

    echo -e "${GREEN}Model deployment complete! (Version: $MODEL_VERSION)${NC}"
    echo -e "${BLUE}=================================================================${NC}"
}

# Function to create a cross-platform monitoring script
create_monitoring_script() {
    local monitor_script="monitor_tcn_${MODEL_VERSION}.sh"

    cat > "$monitor_script" << EOL
#!/bin/bash
# Cross-platform monitoring script for TCN model training
# Generated specifically for model version: $MODEL_VERSION

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

INSTANCE_IP="$INSTANCE_IP"
USERNAME="$USERNAME"
KEY_FILE="$KEY_FILE"
LOG_FILE="$LOG_FILE" # Log file name passed from main script
REMOTE_DIR="$REMOTE_DIR"
PID_FILE="fixed_tcn_large_${MODEL_VERSION}_pid.txt" # Version specific PID file
MONITORING_INTERVAL=30

echo -e "\${BLUE}==================================================================\${NC}"
echo -e "\${GREEN}    MONITORING TCN MODEL TRAINING (\$MODEL_VERSION)    \${NC}"
echo -e "\${BLUE}==================================================================\${NC}"
echo -e "\${YELLOW}Instance:\${NC} \$USERNAME@\$INSTANCE_IP"
echo -e "\${YELLOW}Log file:\${NC} \$LOG_FILE"
echo -e "\${YELLOW}PID file:\${NC} \$PID_FILE"
echo -e "\${BLUE}==================================================================\${NC}"

# Start continuous monitoring
echo -e "\${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.\${NC}"
ssh -t -i "\$KEY_FILE" "\$USERNAME@\$INSTANCE_IP" "
    # Check if training is active
    if [ -f \"\${REMOTE_DIR}/\${PID_FILE}\" ]; then
        PID=\$(cat \"\${REMOTE_DIR}/\${PID_FILE}\")
        # Check if PID is numeric and process exists
        if [[ \"\$PID\" =~ ^[0-9]+$ ]] && ps -p \"\$PID\" > /dev/null; then
            echo -e '${GREEN}Training process (PID: \$PID) is active${NC}'
        else
            echo -e '${RED}Training process (PID: \$PID) is no longer running (or PID file invalid)${NC}'
            # Check exit status if possible (might require more complex log parsing)
            if grep -q 'Training completed' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then
                 echo -e '${GREEN}Training appears to have completed normally.${NC}'
            elif grep -q 'ERROR:' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then
                 echo -e '${RED}Errors detected in log file. Please review.${NC}'
            fi
        fi
    else
        echo -e '${RED}PID file not found. Training might have finished or failed to start.${NC}'
        # Check log file for completion or errors
        if [ -f \"\${REMOTE_DIR}/\${LOG_FILE}\" ]; then
             if grep -q 'Training completed' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then
                 echo -e '${GREEN}Log file indicates training completed normally.${NC}'
             elif grep -q 'ERROR:' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then
                 echo -e '${RED}Errors detected in log file. Please review.${NC}'
             else
                 echo -e '${YELLOW}Log file exists but completion/error status unclear.${NC}'
             fi
        else
             echo -e '${RED}Log file not found either.${NC}'
        fi
    fi

    # Get best validation accuracy (version specific)
    BEST_ACC_FILE=\"best_val_accuracy_\${MODEL_VERSION}.txt\"
    if [ -f \"\${REMOTE_DIR}/\${BEST_ACC_FILE}\" ]; then
        echo -e '${YELLOW}Best validation accuracy recorded:${NC}'
        cat \"\${REMOTE_DIR}/\${BEST_ACC_FILE}\"
    else
        echo -e '${YELLOW}Best validation accuracy file (\${BEST_ACC_FILE}) not found yet.${NC}'
    fi

    # Now follow the log file
    echo -e '${BLUE}===================================================================${NC}'
    echo -e '${GREEN}Following log file output (tail -f):${NC}'
    echo -e '${BLUE}===================================================================${NC}'
    tail -f \"\${REMOTE_DIR}/\${LOG_FILE}\"
"
EOL

    chmod +x "$monitor_script"
    echo -e "${GREEN}Created monitoring script: ${YELLOW}$monitor_script${NC}"
}

# Main execution
echo -e "${BLUE}===========================================================================${NC}"
echo -e "${GREEN}${BOLD}TCN MODEL RETESTING WITH ENHANCED LOGGING${NC}"
echo -e "${BLUE}===========================================================================${NC}"
echo -e "${YELLOW}Model version:${NC} $MODEL_VERSION"
echo -e "${YELLOW}Training script:${NC} $TRAINING_SCRIPT"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo -e "${BLUE}===========================================================================${NC}"

# Step 1: Validate connection
if ! validate_ssh_connection; then
    exit 1
fi

# Step 2: Stop any existing training (COMMENTED OUT as requested)
# stop_existing_training

# Step 3: Deploy and run model with enhanced logging
deploy_model

# Step 4: Create monitoring script
create_monitoring_script

echo -e "${BLUE}===========================================================================${NC}"
echo -e "${GREEN}Model retesting deployment complete!${NC}"
echo -e "${YELLOW}To monitor training, run:${NC} ./monitor_tcn_${MODEL_VERSION}.sh"
echo -e "${BLUE}===========================================================================${NC}"
