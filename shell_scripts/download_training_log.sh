#!/bin/bash
# Script to download and examine the training log for the model with 82.9% accuracy

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"
LOG_FILE="/home/ec2-user/emotion_training/training_no_leakage_conda.log"
TARGET_EPOCH="36"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DOWNLOADING TRAINING LOG FOR EPOCH ${TARGET_EPOCH} (82.9% ACCURACY)     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Create local directory for logs if it doesn't exist
mkdir -p target_model/logs

# Download the complete log file
echo -e "${YELLOW}Downloading the complete training log...${NC}"
scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:${LOG_FILE} target_model/logs/training_no_leakage_conda.log

# Check if download was successful
if [ ! -f "target_model/logs/training_no_leakage_conda.log" ]; then
    echo -e "${RED}Failed to download the log file.${NC}"
    
    # Try to get log excerpts directly
    echo -e "${YELLOW}Trying to extract relevant sections directly from the EC2 instance...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/extract_log.sh << 'EOF'
#!/bin/bash
if [ -f ${LOG_FILE} ]; then
  echo \"Extracting epoch ${TARGET_EPOCH} information from ${LOG_FILE}\"
  # Extract lines related to epoch 36 and surrounding context
  grep -A 15 -B 5 \"Epoch ${TARGET_EPOCH}/\" ${LOG_FILE} > ~/epoch_${TARGET_EPOCH}_extract.txt
  
  # Also extract general accuracy information
  grep -E 'val_accuracy:|accuracy:' ${LOG_FILE} > ~/accuracy_summary.txt
  
  echo \"Extracts created successfully\"
else
  echo \"Log file not found: ${LOG_FILE}\"
fi
EOF"

    # Make the script executable and run it
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/extract_log.sh && ~/extract_log.sh"
    
    # Download the extracts
    echo -e "${YELLOW}Downloading log extracts...${NC}"
    scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/epoch_${TARGET_EPOCH}_extract.txt target_model/logs/ 2>/dev/null
    scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/accuracy_summary.txt target_model/logs/ 2>/dev/null
    
    # Display the extracts if available
    if [ -f "target_model/logs/epoch_${TARGET_EPOCH}_extract.txt" ]; then
        echo -e "${BLUE}=================================================================${NC}"
        echo -e "${BLUE}     EPOCH ${TARGET_EPOCH} TRAINING LOG EXTRACT (82.9% ACCURACY)     ${NC}"
        echo -e "${BLUE}=================================================================${NC}"
        cat target_model/logs/epoch_${TARGET_EPOCH}_extract.txt
    else
        echo -e "${RED}Failed to extract log information.${NC}"
    fi
else
    # Log file was downloaded successfully
    echo -e "${GREEN}Log file downloaded successfully.${NC}"
    
    # Extract and display the relevant section from the log file
    echo -e "${YELLOW}Extracting information about epoch ${TARGET_EPOCH}...${NC}"
    
    # Extract 5 lines before and 15 lines after each occurrence of "Epoch 36/"
    grep -A 15 -B 5 "Epoch ${TARGET_EPOCH}/" target_model/logs/training_no_leakage_conda.log > target_model/logs/epoch_${TARGET_EPOCH}_context.txt
    
    # Also create a summary of all accuracy values
    grep -E 'val_accuracy:|accuracy:' target_model/logs/training_no_leakage_conda.log > target_model/logs/accuracy_summary.txt
    
    # Display the context around epoch 36
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}     EPOCH ${TARGET_EPOCH} TRAINING LOG EXTRACT (82.9% ACCURACY)     ${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    if [ -s "target_model/logs/epoch_${TARGET_EPOCH}_context.txt" ]; then
        cat target_model/logs/epoch_${TARGET_EPOCH}_context.txt
    else
        echo -e "${RED}Could not find explicit reference to epoch ${TARGET_EPOCH} in the log.${NC}"
        echo -e "${YELLOW}Checking for any validation accuracy of 82.9%...${NC}"
        
        # Look for any 82.9% accuracy in the log
        grep -E "val_accuracy: 0.829|val_accuracy: 0.829[0-9]" target_model/logs/training_no_leakage_conda.log -A 3 -B 10 > target_model/logs/accuracy_829_context.txt
        
        if [ -s "target_model/logs/accuracy_829_context.txt" ]; then
            cat target_model/logs/accuracy_829_context.txt
        else
            echo -e "${RED}Could not find validation accuracy of 82.9% in the log.${NC}"
            
            # Display a summary of the log
            echo -e "${YELLOW}Displaying summary of validation accuracies in the log:${NC}"
            grep "val_accuracy:" target_model/logs/training_no_leakage_conda.log | head -50
        fi
    fi
fi

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}PROCESS COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
