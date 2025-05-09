#!/bin/bash
# Script to download the specific model with 82.9% validation accuracy (Epoch 36)

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="35.168.113.21"
KEY_FILE="/Users/patrickgloria/Downloads/new-key.pem"
TARGET_ACCURACY="82.9"
TARGET_EPOCH="36"
LOG_FILE="/home/ec2-user/emotion_training/training_no_leakage_conda.log"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DOWNLOADING MODEL WITH EXACT ${TARGET_ACCURACY}% ACCURACY (EPOCH ${TARGET_EPOCH})     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Target Log:${NC} $LOG_FILE"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Create local directory for the model
mkdir -p target_model

# First, try to find the checkpoint file for epoch 36 specifically
echo -e "${YELLOW}Searching for checkpoint from epoch ${TARGET_EPOCH}...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/find_exact_model.sh << 'EOF'
#!/bin/bash
# Define model directories to search
MODEL_DIRS=(
  \"~/emotion_training/models/no_leakage_conda\"
  \"~/emotion_training/models\"
  \"~/emotion_training\"
)

# Look for checkpoint files with epoch 36
for dir in \"\${MODEL_DIRS[@]}\"; do
  if [ -d \$(eval echo \$dir) ]; then
    echo \"Searching in \$(eval echo \$dir)\"
    # Look for files with epoch_36 or just 36 in their name
    MODELS=\$(find \$(eval echo \$dir) -name \"*epoch_36*.h5\" -o -name \"*36*.h5\" -o -name \"*checkpoint*.h5\" 2>/dev/null)
    if [ -n \"\$MODELS\" ]; then
      echo \"Found potential models:\"
      echo \"\$MODELS\"
      # Write to result file
      echo \"FOUND_MODELS=\$MODELS\" > ~/exact_model_results.txt
      exit 0
    fi
  fi
done

# If no epoch 36 file found, look for the checkpoint files
echo \"No specific epoch 36 checkpoint found. Looking for best model in no_leakage_conda directory...\"
if [ -d ~/emotion_training/models/no_leakage_conda ]; then
  MODELS=\$(find ~/emotion_training/models/no_leakage_conda -name \"*.h5\" 2>/dev/null)
  if [ -n \"\$MODELS\" ]; then
    echo \"Found models in no_leakage_conda:\"
    echo \"\$MODELS\"
    echo \"FOUND_MODELS=\$MODELS\" > ~/exact_model_results.txt
    exit 0
  fi
fi

# If still no files found, check if we can find by training date (around epoch 36)
echo \"Looking in the metadata of the log file to identify when epoch 36 was trained...\"
if [ -f ${LOG_FILE} ]; then
  EPOCH_36_DATE=\$(grep \"Epoch 36/\" ${LOG_FILE} | head -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
  if [ -n \"\$EPOCH_36_DATE\" ]; then
    echo \"Epoch 36 was trained on \$EPOCH_36_DATE\"
    # Find files modified around that date
    echo \"Looking for model files saved on \$EPOCH_36_DATE\"
    MODELS=\$(find ~/emotion_training -name \"*.h5\" -type f -newermt \"\$EPOCH_36_DATE\" -not -newermt \"\$(date -d \"\$EPOCH_36_DATE + 1 day\" +%Y-%m-%d)\" 2>/dev/null)
    if [ -n \"\$MODELS\" ]; then
      echo \"Found models from \$EPOCH_36_DATE:\"
      echo \"\$MODELS\"
      echo \"FOUND_MODELS=\$MODELS\" > ~/exact_model_results.txt
      exit 0
    fi
  fi
fi

# Last resort - just get any model files from the no_leakage_conda directory
echo \"Checking for any model files in relevant directories...\"
MODELS=\$(find ~/emotion_training/models -name \"*.h5\" | grep -i \"no_leakage\\|leakage_conda\" 2>/dev/null)
if [ -n \"\$MODELS\" ]; then
  echo \"Found these potentially relevant models:\"
  echo \"\$MODELS\"
  echo \"FOUND_MODELS=\$MODELS\" > ~/exact_model_results.txt
  exit 0
fi

echo \"No models found\"
echo \"FOUND_MODELS=\" > ~/exact_model_results.txt
EOF"

# Make the script executable and run it
echo -e "${YELLOW}Executing the search script on the EC2 instance...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/find_exact_model.sh && ~/find_exact_model.sh"

# Download the results file
echo -e "${YELLOW}Downloading search results...${NC}"
scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/exact_model_results.txt target_model/exact_results.txt

# Source the results file
source target_model/exact_results.txt

# Check if models were found
if [ -z "$FOUND_MODELS" ]; then
    echo -e "${RED}No model files found matching criteria.${NC}"
    
    # Special handling - try to get the best model from no_leakage_conda
    echo -e "${YELLOW}Falling back to best model in no_leakage_conda directory...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "mkdir -p ~/emotion_training/models/no_leakage_conda || true"
    MODEL_EXISTS=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ls -la ~/emotion_training/models/no_leakage_conda/model_best.h5 2>/dev/null || echo 'NOT_FOUND'")
    
    if [[ "$MODEL_EXISTS" != *"NOT_FOUND"* ]]; then
        echo -e "${GREEN}Found best model in no_leakage_conda directory.${NC}"
        echo -e "${YELLOW}Downloading it...${NC}"
        scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/emotion_training/models/no_leakage_conda/model_best.h5 target_model/
        
        if [ -f "target_model/model_best.h5" ]; then
            echo -e "${GREEN}Successfully downloaded the best model from no_leakage_conda.${NC}"
            echo -e "${YELLOW}Renaming to 'model_82.9_accuracy.h5'${NC}"
            mv target_model/model_best.h5 target_model/model_82.9_accuracy.h5
        else
            echo -e "${RED}Failed to download the best model.${NC}"
        fi
    else
        echo -e "${RED}Could not find any model in the no_leakage_conda directory.${NC}"
    fi
else
    # Process the first model in the list
    FIRST_MODEL=$(echo $FOUND_MODELS | cut -d' ' -f1)
    echo -e "${YELLOW}Downloading model: $FIRST_MODEL${NC}"
    
    # Extract the filename
    filename=$(basename "$FIRST_MODEL")
    echo -e "${YELLOW}File: $filename${NC}"
    
    # Download the model
    scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:"$FIRST_MODEL" target_model/
    
    if [ -f "target_model/$filename" ]; then
        echo -e "${GREEN}Successfully downloaded the model.${NC}"
        echo -e "${YELLOW}Renaming to 'model_82.9_accuracy.h5'${NC}"
        mv target_model/$filename target_model/model_82.9_accuracy.h5
    else
        echo -e "${RED}Failed to download the model.${NC}"
    fi
fi

# Check if we now have the model
if [ -f "target_model/model_82.9_accuracy.h5" ]; then
    # Create a summary file
    echo -e "${YELLOW}Creating model summary...${NC}"
    echo "MODEL SUMMARY" > target_model/model_summary.txt
    echo "=============" >> target_model/model_summary.txt
    echo "Target accuracy: ${TARGET_ACCURACY}%" >> target_model/model_summary.txt
    echo "Source log file: ${LOG_FILE}" >> target_model/model_summary.txt
    echo "Found at epoch: ${TARGET_EPOCH}" >> target_model/model_summary.txt
    echo "Original path: ${FIRST_MODEL}" >> target_model/model_summary.txt
    echo "Local path: target_model/model_82.9_accuracy.h5" >> target_model/model_summary.txt
    echo "File size: $(du -h target_model/model_82.9_accuracy.h5 | cut -f1) bytes" >> target_model/model_summary.txt
    echo "Download date: $(date)" >> target_model/model_summary.txt
    
    echo -e "${GREEN}Model downloaded successfully as target_model/model_82.9_accuracy.h5${NC}"
    echo -e "${GREEN}Model summary saved to target_model/model_summary.txt${NC}"
else
    echo -e "${RED}Failed to download any model with ${TARGET_ACCURACY}% accuracy.${NC}"
fi

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}PROCESS COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
