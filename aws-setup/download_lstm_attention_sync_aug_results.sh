#!/bin/bash
# Download the LSTM attention model with synchronized augmentation results from AWS EC2

# Check if we have connection information file
CONNECTION_INFO_FILE="aws-setup/lstm_attention_model_connection.txt"
if [ ! -f "$CONNECTION_INFO_FILE" ]; then
    echo "ERROR: Connection info file not found: $CONNECTION_INFO_FILE"
    echo "Make sure you've run deploy_lstm_attention_sync_aug.sh first."
    exit 1
fi

# Source the connection info
source "$CONNECTION_INFO_FILE"

# Validate connection info
if [ -z "$INSTANCE_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "ERROR: Connection information incomplete."
    echo "INSTANCE_IP or KEY_FILE is missing from $CONNECTION_INFO_FILE"
    exit 1
fi

echo "================================================================="
echo "  DOWNLOADING LSTM ATTENTION MODEL WITH SYNCHRONIZED AUGMENTATION"
echo "================================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Key file: $KEY_FILE"

# Create output directory
OUTPUT_DIR="models/lstm_attention_sync_aug"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Check if instance is running
INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
if [ "$INSTANCE_STATE" != "running" ]; then
    echo "ERROR: Instance $INSTANCE_ID is not running (current state: $INSTANCE_STATE)"
    echo "Please start the instance before downloading results."
    exit 1
fi

# Check if SSH is responsive
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@$INSTANCE_IP exit &>/dev/null; then
    echo "ERROR: Cannot connect to the instance via SSH."
    echo "Please check that the instance is running and SSH is available."
    exit 1
fi

echo "Checking if training is complete..."
TRAINING_PID=$(ssh -i "$KEY_FILE" ec2-user@$INSTANCE_IP "pgrep -f train_branched_attention_sync_aug.py || echo ''")
if [ ! -z "$TRAINING_PID" ]; then
    echo "WARNING: Training process is still running (PID: $TRAINING_PID)"
    read -p "Do you want to download current results anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download canceled. Run again after training completes."
        exit 0
    fi
fi

echo "Downloading training log..."
scp -i "$KEY_FILE" ec2-user@$INSTANCE_IP:~/emotion_training/training_lstm_attention_sync_aug.log "$OUTPUT_DIR/logs/training_log.txt"

echo "Checking for trained models..."
MODEL_COUNT=$(ssh -i "$KEY_FILE" ec2-user@$INSTANCE_IP "ls -1 ~/emotion_training/models/attention_focal_loss_sync_aug/*.keras 2>/dev/null | wc -l")

if [ "$MODEL_COUNT" -eq "0" ]; then
    echo "WARNING: No trained models found. Training may not have produced a model yet."
    read -p "Do you want to continue downloading other files? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download canceled."
        exit 0
    fi
else
    echo "Found $MODEL_COUNT model file(s). Downloading..."
    
    # Create local directory
    mkdir -p "$OUTPUT_DIR/models"
    
    # Download models
    scp -i "$KEY_FILE" ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss_sync_aug/*.keras "$OUTPUT_DIR/models/"
    
    # Get the best model accuracy from the log file
    if [ -f "$OUTPUT_DIR/logs/training_log.txt" ]; then
        BEST_ACCURACY=$(grep -oP "val_accuracy: \K[0-9]\.[0-9]+" "$OUTPUT_DIR/logs/training_log.txt" | sort -nr | head -1)
        if [ ! -z "$BEST_ACCURACY" ]; then
            echo "Best validation accuracy: $BEST_ACCURACY"
            
            # Create model_info.json with accuracy
            MODEL_INFO="{
  \"model_type\": \"lstm_attention_synchronized_augmentation\",
  \"description\": \"LSTM model with attention mechanism and synchronized audio-visual augmentation\",
  \"best_accuracy\": $BEST_ACCURACY,
  \"training_completed\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"
}"
            echo "$MODEL_INFO" > "$OUTPUT_DIR/model_info.json"
            echo "Created model_info.json with accuracy information"
        fi
    fi
fi

# Download any custom files that may have been generated
echo "Checking for additional result files..."
CUSTOM_FILES_COUNT=$(ssh -i "$KEY_FILE" ec2-user@$INSTANCE_IP "find ~/emotion_training -name \"*sync_aug*\" -not -path \"*/models/*\" -not -path \"*/scripts/*\" | wc -l")

if [ "$CUSTOM_FILES_COUNT" -gt "0" ]; then
    echo "Found $CUSTOM_FILES_COUNT additional file(s). Downloading..."
    
    # Create directory for other files
    mkdir -p "$OUTPUT_DIR/additional_files"
    
    # Create a list of files to download
    ssh -i "$KEY_FILE" ec2-user@$INSTANCE_IP "find ~/emotion_training -name \"*sync_aug*\" -not -path \"*/models/*\" -not -path \"*/scripts/*\"" > /tmp/sync_aug_files.txt
    
    # Download each file
    while IFS= read -r file; do
        TARGET_DIR="$OUTPUT_DIR/additional_files"
        TARGET_FILE="$(basename "$file")"
        echo "Downloading $TARGET_FILE..."
        scp -i "$KEY_FILE" ec2-user@$INSTANCE_IP:"$file" "$TARGET_DIR/$TARGET_FILE"
    done < /tmp/sync_aug_files.txt
    
    rm /tmp/sync_aug_files.txt
fi

# Calculate total download size
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)

echo "================================================================="
echo "DOWNLOAD COMPLETE"
echo "================================================================="
echo "Downloaded files saved to: $OUTPUT_DIR"
echo "Total size: $TOTAL_SIZE"
echo "Best model(s) located in: $OUTPUT_DIR/models/"
echo "Training log saved to: $OUTPUT_DIR/logs/training_log.txt"
echo "================================================================="
echo "To verify model accuracy, check $OUTPUT_DIR/model_info.json"
echo "To inspect training progress, view the training log"
echo "================================================================="
