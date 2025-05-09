#!/bin/bash

# --- Configuration ---
INSTANCE_IP="18.206.48.166" # Replace with your EC2 instance IP if different
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem" # Path to your PEM key file
LOCAL_SCRIPT_PATH="scripts/train_spectrogram_cnn_lstm.py" # Path to the NEW training script
REMOTE_SCRIPT_NAME="train_spectrogram_cnn_lstm.py" # Name for the script on the EC2 instance
REMOTE_DIR="/home/ec2-user/emotion_training" # Directory on EC2 for training files
LOG_PREFIX="training_precomputed_cnn_lstm" # Prefix for log files (Updated)
PID_PREFIX="precomputed_cnn_lstm" # Prefix for PID files (Updated)
# --- End Configuration ---

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN} DEPLOYING PRECOMPUTED CNN AUDIO + LSTM MODEL SCRIPT ${NC}" # Updated Title
echo -e "${BLUE}===================================================================${NC}"

# Check if local script exists
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Local training script not found at $LOCAL_SCRIPT_PATH${NC}"
    exit 1
fi

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: PEM key file not found at $KEY_FILE${NC}"
    exit 1
fi

# Create remote directory if it doesn't exist
echo -e "${YELLOW}Ensuring remote directory exists ($REMOTE_DIR)...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "mkdir -p $REMOTE_DIR"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create or access remote directory $REMOTE_DIR${NC}"
    exit 1
fi
echo "Remote directory check complete."

# Install dependencies on remote instance (ensure tensorflow, numpy, tqdm are there)
# Assuming base AMI has Python/pip. Add others if needed by preprocessing/training.
echo -e "${YELLOW}Ensuring core dependencies (tensorflow, numpy, tqdm) are installed on EC2 instance...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "pip install tensorflow numpy tqdm"
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Failed to install core dependencies via pip on remote instance. Continuing, but script might fail.${NC}"
fi
echo "Dependency check/install attempt complete."

# --- Run CNN Feature Preprocessing on EC2 (if needed) ---
# 1. Transfer the necessary scripts
LOCAL_CNN_PREPROCESS_SCRIPT="scripts/preprocess_cnn_audio_features.py"
REMOTE_CNN_PREPROCESS_SCRIPT="preprocess_cnn_audio_features.py"
LOCAL_CNN_GENERATOR_DEP="scripts/spectrogram_cnn_pooling_generator.py" # Dependency
REMOTE_CNN_GENERATOR_DEP="spectrogram_cnn_pooling_generator.py"
LOCAL_SPEC_PREPROCESS_DEP="scripts/preprocess_spectrograms.py" # Dependency for N_MELS import
REMOTE_SPEC_PREPROCESS_DEP="preprocess_spectrograms.py"

echo -e "${YELLOW}Transferring CNN preprocessing script ($LOCAL_CNN_PREPROCESS_SCRIPT) and dependencies to AWS instance...${NC}"
scp -i "$KEY_FILE" "$LOCAL_CNN_PREPROCESS_SCRIPT" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_CNN_PREPROCESS_SCRIPT"
if [ $? -ne 0 ]; then echo -e "${RED}Error: CNN Preprocessing script transfer failed.${NC}"; exit 1; fi
scp -i "$KEY_FILE" "$LOCAL_CNN_GENERATOR_DEP" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_CNN_GENERATOR_DEP"
if [ $? -ne 0 ]; then echo -e "${RED}Error: CNN Generator dependency transfer failed.${NC}"; exit 1; fi
scp -i "$KEY_FILE" "$LOCAL_SPEC_PREPROCESS_DEP" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_SPEC_PREPROCESS_DEP"
if [ $? -ne 0 ]; then echo -e "${RED}Error: Spectrogram Preprocessing dependency transfer failed.${NC}"; exit 1; fi
echo -e "${GREEN}CNN Preprocessing scripts transferred successfully.${NC}"

# 2. Check if target directories exist and UPLOAD local features if necessary
REMOTE_TARGET_DATA_DIR="$REMOTE_DIR/data" # Target data dir relative to training dir
REMOTE_CNN_RAVDESS_OUT="$REMOTE_TARGET_DATA_DIR/ravdess_features_cnn_audio"
REMOTE_CNN_CREMAD_OUT="$REMOTE_TARGET_DATA_DIR/crema_d_features_cnn_audio"
echo -e "${YELLOW}Checking if precomputed CNN features exist on EC2 at $REMOTE_TARGET_DATA_DIR...${NC}"
# Check command: test if directory exists AND is not empty (-A checks for any file, including hidden)
CHECK_CMD="if [ -d \"$REMOTE_CNN_RAVDESS_OUT\" ] && [ \"\$(ls -A $REMOTE_CNN_RAVDESS_OUT)\" ] && [ -d \"$REMOTE_CNN_CREMAD_OUT\" ] && [ \"\$(ls -A $REMOTE_CNN_CREMAD_OUT)\" ]; then echo 'exists'; else echo 'missing'; fi"
EXISTENCE_STATUS=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$CHECK_CMD")

if [ "$EXISTENCE_STATUS" == "missing" ]; then
    echo -e "${YELLOW}Precomputed CNN features not found or directories empty in $REMOTE_TARGET_DATA_DIR on EC2.${NC}"
    LOCAL_CNN_RAVDESS_IN="data/ravdess_features_cnn_audio"
    LOCAL_CNN_CREMAD_IN="data/crema_d_features_cnn_audio"

    # Check if local directories exist
    if [ ! -d "$LOCAL_CNN_RAVDESS_IN" ] || [ ! -d "$LOCAL_CNN_CREMAD_IN" ]; then
        echo -e "${RED}Error: Local precomputed CNN feature directories not found ($LOCAL_CNN_RAVDESS_IN, $LOCAL_CNN_CREMAD_IN). Cannot upload.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Uploading local precomputed CNN features to $REMOTE_TARGET_DATA_DIR on EC2 (this might take some time)...${NC}"
    # Ensure the target remote data directory exists inside the training dir
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "mkdir -p $REMOTE_TARGET_DATA_DIR"
    # Use scp -r to copy directories recursively into the target directory
    scp -r -i "$KEY_FILE" "$LOCAL_CNN_RAVDESS_IN" "$USERNAME@$INSTANCE_IP:$REMOTE_TARGET_DATA_DIR/"
    if [ $? -ne 0 ]; then echo -e "${RED}Error: Failed to upload $LOCAL_CNN_RAVDESS_IN.${NC}"; exit 1; fi
    scp -r -i "$KEY_FILE" "$LOCAL_CNN_CREMAD_IN" "$USERNAME@$INSTANCE_IP:$REMOTE_TARGET_DATA_DIR/"
    if [ $? -ne 0 ]; then echo -e "${RED}Error: Failed to upload $LOCAL_CNN_CREMAD_IN.${NC}"; exit 1; fi
    echo -e "${GREEN}Local precomputed CNN features uploaded successfully to $REMOTE_TARGET_DATA_DIR.${NC}"
else
    echo -e "${GREEN}Precomputed CNN features found on EC2 in $REMOTE_TARGET_DATA_DIR. Skipping upload.${NC}"
fi
# --- End CNN Feature Preprocessing ---


# Transfer the training script using SCP
echo -e "${YELLOW}Transferring Precomputed CNN Audio + LSTM script ($LOCAL_SCRIPT_PATH) to AWS instance...${NC}" # Updated description
scp -i "$KEY_FILE" "$LOCAL_SCRIPT_PATH" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Training script transfer failed.${NC}"
    exit 1
fi
echo -e "${GREEN}Training script transferred successfully!${NC}"

# Transfer the CORRECT generator script using SCP
LOCAL_GENERATOR_PATH="scripts/precomputed_cnn_audio_generator.py" # Path to the CORRECT generator
REMOTE_GENERATOR_NAME="precomputed_cnn_audio_generator.py" # Name for the generator on EC2
echo -e "${YELLOW}Transferring Precomputed CNN Audio Generator script ($LOCAL_GENERATOR_PATH) to AWS instance...${NC}" # Updated description
scp -i "$KEY_FILE" "$LOCAL_GENERATOR_PATH" "$USERNAME@$INSTANCE_IP:$REMOTE_DIR/$REMOTE_GENERATOR_NAME"
if [ $? -ne 0 ]; then echo -e "${RED}Error: Precomputed CNN Audio generator script transfer failed.${NC}"; exit 1; fi # Updated error message
echo -e "${GREEN}Precomputed CNN Audio generator script transferred successfully!${NC}" # Updated success message

# NOTE: Do NOT transfer old generator scripts (synchronized_data_generator.py, sequence_data_generator.py)

# Set execute permissions on the remote training script
echo -e "${YELLOW}Setting execute permissions on the training script...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x $REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to set execute permissions on remote script.${NC}"
    # Attempting to continue, Python might still execute it
fi

# Check Python syntax on the remote script
echo -e "${YELLOW}Checking script for Python syntax errors...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "python3 -m py_compile $REMOTE_DIR/$REMOTE_SCRIPT_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Python syntax check failed for $REMOTE_SCRIPT_NAME on the EC2 instance.${NC}"
    echo -e "${YELLOW}Please check the script for errors before launching.${NC}"
    exit 1
else
    echo -e "${GREEN}Syntax check passed! Script is valid Python.${NC}"
fi

# Create a timestamp for log and PID files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REMOTE_LOG_FILE="${LOG_PREFIX}_${TIMESTAMP}.log"
REMOTE_PID_FILE="${PID_PREFIX}_${TIMESTAMP}_pid.txt"

# Create a launch script on the remote instance
REMOTE_LAUNCH_SCRIPT="launch_${PID_PREFIX}.sh" # Use PID_PREFIX in launch script name
LAUNCH_SCRIPT_CONTENT="#!/bin/bash
cd $REMOTE_DIR || exit 1 # Change to the training directory
echo \"Starting Precomputed CNN Audio + LSTM model training...\" > $REMOTE_LOG_FILE # Updated description
echo \"PID will be written to: $REMOTE_PID_FILE\" >> $REMOTE_LOG_FILE
nohup python3 -u $REMOTE_SCRIPT_NAME >> $REMOTE_LOG_FILE 2>&1 &
PID=\$!
echo \$PID > $REMOTE_PID_FILE
echo \"Training process started with PID: \$PID\" >> $REMOTE_LOG_FILE
echo \"Logs are being written to: $REMOTE_LOG_FILE\"
# Optionally, print Python version
{ python3 --version >> $REMOTE_LOG_FILE 2>&1 || echo 'Failed to get Python version' >> $REMOTE_LOG_FILE; }

"

echo -e "${YELLOW}Creating launch script (~/$REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT) on AWS instance...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "echo '$LAUNCH_SCRIPT_CONTENT' > $REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT && chmod +x $REMOTE_DIR/$REMOTE_LAUNCH_SCRIPT"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create launch script on remote instance.${NC}"
    exit 1
fi
echo -e "${GREEN}Launch script created successfully!${NC}"

# Launch the training process using the launch script
echo -e "${YELLOW}Launching training process on AWS instance...${NC}"
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd $REMOTE_DIR && ./$REMOTE_LAUNCH_SCRIPT"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to execute launch script on remote instance.${NC}"
    exit 1
fi
echo -e "${GREEN}Training launch command executed.${NC}"
echo -e "${YELLOW}Please monitor the training progress using a specific monitoring script or by checking the logs on the EC2 instance.${NC}"
echo -e "${YELLOW}The specific log and PID files will have a timestamp like YYYYMMDD_HHMMSS and be named like '${LOG_PREFIX}_...' and '${PID_PREFIX}_...'.${NC}"


# Define content for a remote helper script that does the actual monitoring
REMOTE_MONITOR_HELPER_NAME="monitor_helper_${PID_PREFIX}_${TIMESTAMP}.sh" # Use PID_PREFIX (already updated)
REMOTE_MONITOR_HELPER_CONTENT="#!/bin/bash
# This script runs on the EC2 instance to check status and tail logs

REMOTE_DIR=\"$REMOTE_DIR\"
LOG_FILE=\"$REMOTE_LOG_FILE\"
PID_FILE=\"$REMOTE_PID_FILE\"

# Check if training is active
if [ -f \"\${REMOTE_DIR}/\${PID_FILE}\" ]; then
    PID=\$(cat \"\${REMOTE_DIR}/\${PID_FILE}\")
    if [[ \"\$PID\" =~ ^[0-9]+\$ ]] && ps -p \"\$PID\" > /dev/null; then
        echo -e '\033[0;32mTraining process (PID: \$PID) is active\033[0m'
    else
        echo -e '\033[0;31mTraining process (PID: \$PID) is no longer running (or PID file invalid)\033[0m'
        if grep -q 'Training finished' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;32mTraining appears to have completed normally.\033[0m';
        elif grep -q 'Error:' \"\${REMOTE_DIR}/\${LOG_FILE}\" || grep -q 'Traceback' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m'; fi
    fi
else
    echo -e '\033[0;31mPID file not found. Training might have finished or failed to start.\033[0m'
    if [ -f \"\${REMOTE_DIR}/\${LOG_FILE}\" ]; then
         if grep -q 'Training finished' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;32mLog file indicates training completed normally.\033[0m';
         elif grep -q 'Error:' \"\${REMOTE_DIR}/\${LOG_FILE}\" || grep -q 'Traceback' \"\${REMOTE_DIR}/\${LOG_FILE}\"; then echo -e '\033[0;31mErrors detected in log file. Please review.\033[0m';
         else echo -e '\033[1;33mLog file exists but completion/error status unclear.\033[0m'; fi
    else echo -e '\033[0;31mLog file not found either.\033[0m'; fi
fi

# Check best accuracy
BEST_ACC_LINE=\$(grep 'val_accuracy improved from' \"\${REMOTE_DIR}/\${LOG_FILE}\" | tail -n 1)
if [ -n \"\$BEST_ACC_LINE\" ]; then
    echo -e '\033[1;33mLast recorded best validation accuracy improvement:\033[0m'; echo \"\$BEST_ACC_LINE\"
else
     BEST_ACC_LINE_INIT=\$(grep 'Epoch 00001: val_accuracy improved from' \"\${REMOTE_DIR}/\${LOG_FILE}\" | head -n 1)
     if [ -n \"\$BEST_ACC_LINE_INIT\" ]; then echo -e '\033[1;33mInitial validation accuracy recorded:\033[0m'; echo \"\$BEST_ACC_LINE_INIT\"
     else echo -e '\033[1;33mBest validation accuracy not found in log yet.\033[0m'; fi
fi

# Follow log file
echo -e '\033[0;34m===================================================================\033[0m'
echo -e '\033[0;32mFollowing log file output (tail -f):\033[0m'
echo -e '\033[0;34m===================================================================\033[0m'
tail -f \"\${REMOTE_DIR}/\${LOG_FILE}\"
"

# Create the remote helper script via SSH
echo -e "${YELLOW}Creating remote monitoring helper script ($REMOTE_MONITOR_HELPER_NAME)...${NC}"
# Use printf to handle potential special characters in the content safely
printf "%s" "$REMOTE_MONITOR_HELPER_CONTENT" | ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > $REMOTE_DIR/$REMOTE_MONITOR_HELPER_NAME && chmod +x $REMOTE_DIR/$REMOTE_MONITOR_HELPER_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create remote monitoring helper script.${NC}"
    exit 1
fi
echo -e "${GREEN}Remote monitoring helper script created successfully.${NC}"


# Generate a specific local monitoring script that calls the remote helper
MONITOR_SCRIPT_NAME="monitor_${PID_PREFIX}_${TIMESTAMP}.sh" # Use PID_PREFIX
MONITOR_SCRIPT_CONTENT="#!/bin/bash
# Monitoring script for Precomputed CNN Audio + LSTM model training # Updated description
# Generated specifically for run started around: ${TIMESTAMP}

# ANSI colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

INSTANCE_IP=\"18.206.48.166\" # Updated IP
USERNAME=\"$USERNAME\"
KEY_FILE=\"/Users/patrickgloria/conjunction-train/aws-setup/emotion-recognition-key-fixed-20250323090016.pem\" # Use absolute path
REMOTE_DIR=\"$REMOTE_DIR\"
REMOTE_MONITOR_HELPER=\"$REMOTE_MONITOR_HELPER_NAME\" # Name of the helper script on EC2

echo -e \"\${BLUE}==================================================================\${NC}\"
echo -e \"\${GREEN} MONITORING PRECOMPUTED CNN AUDIO + LSTM MODEL TRAINING (${TIMESTAMP}) \${NC}\" # Updated Title
echo -e \"\${BLUE}==================================================================\${NC}\"
echo -e \"\${YELLOW}Instance:\${NC} \$USERNAME@\$INSTANCE_IP\"
echo -e \"\${YELLOW}Executing remote helper:\${NC} \$REMOTE_DIR/\$REMOTE_MONITOR_HELPER\"
echo -e \"\${BLUE}==================================================================\${NC}\"

# Start continuous monitoring by explicitly using bash to execute the remote helper script (single command string - trying again)
echo -e \"\${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.\${NC}\"
ssh -t -i \"\$KEY_FILE\" \"\$USERNAME@\$INSTANCE_IP\" \"bash $REMOTE_DIR/$REMOTE_MONITOR_HELPER\" # Execute bash with script path as single command string

"

echo -e "${YELLOW}Generating local monitoring script: $MONITOR_SCRIPT_NAME...${NC}"
echo "$MONITOR_SCRIPT_CONTENT" > "$MONITOR_SCRIPT_NAME"
chmod +x "$MONITOR_SCRIPT_NAME"
echo -e "${GREEN}Monitoring script '$MONITOR_SCRIPT_NAME' created successfully.${NC}"
echo -e "${YELLOW}You can run './$MONITOR_SCRIPT_NAME' to monitor the training.${NC}"


echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment and launch attempt completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
