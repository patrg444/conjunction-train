#!/bin/bash
# Script to find and download a model with validation accuracy of 84.1%

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="3.235.76.0"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
TARGET_ACCURACY="84.1"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     FINDING MODEL WITH VALIDATION ACCURACY AROUND ${TARGET_ACCURACY}%     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
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
mkdir -p models/branched_no_leakage_84_1

# Create a temporary script on the remote machine to find models with accuracy close to target
echo -e "${YELLOW}Creating script to find models with accuracy close to ${TARGET_ACCURACY}%...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/find_target_model.sh << 'EOF'
#!/bin/bash
# Find model checkpoints and their corresponding validation accuracies

# Define directories to search
MODEL_DIRS=(
  \"~/emotion_training/models/no_leakage\"
  \"~/emotion_training/models/no_leakage_conda\"
  \"~/emotion_training/models/dynamic_padding_no_leakage\"
  \"~/emotion_training/models\"
  \"~/emotion_training\"
)

TARGET_ACCURACY=84.1
BEST_DIFF=100
BEST_MODEL=\"\"

# Function to extract accuracies from log files
extract_accuracies() {
  local log_file=\$1
  
  if [ -f \"\$log_file\" ]; then
    # Extract lines with val_accuracy and epoch information
    grep -E 'val_accuracy:|Epoch [0-9]+/' \"\$log_file\" | grep -v ETA
  else
    echo \"Log file \$log_file not found\"
  fi
}

# Find all log files
LOG_FILES=\$(find ~/emotion_training -name \"*.log\" | sort)

echo \"Found \$(echo \$LOG_FILES | wc -w) log files\"

# Process each log file
for log_file in \$LOG_FILES; do
  echo \"Analyzing \$log_file...\"
  
  # Extract accuracies and epoch info
  accuracy_data=\$(extract_accuracies \"\$log_file\")
  
  # Parse the accuracy data to find checkpoints close to target
  while read -r line; do
    if [[ \$line =~ Epoch\ ([0-9]+)/([0-9]+) ]]; then
      current_epoch=\${BASH_REMATCH[1]}
    elif [[ \$line =~ val_accuracy:\ ([0-9]+\.[0-9]+) ]]; then
      accuracy=\${BASH_REMATCH[1]}
      percent_accuracy=\$(echo \"\$accuracy * 100\" | bc)
      diff=\$(echo \"(\$percent_accuracy - \$TARGET_ACCURACY)^2\" | bc)
      
      echo \"Epoch \$current_epoch: \$percent_accuracy% (diff: \$diff)\"
      
      # If this is closer to our target
      if (( \$(echo \"\$diff < \$BEST_DIFF\" | bc -l) )); then
        BEST_DIFF=\$diff
        BEST_MODEL=\"epoch_\$current_epoch\"
        BEST_ACCURACY=\$percent_accuracy
        BEST_LOG_FILE=\$log_file
        
        echo \"New best match: Epoch \$current_epoch with \$percent_accuracy% (diff: \$diff)\"
      fi
    fi
  done <<< \"\$accuracy_data\"
done

# Now find the actual model file for the best match
echo \"\"
echo \"Best match: Epoch with accuracy \$BEST_ACCURACY% (target: \$TARGET_ACCURACY%)\"
echo \"Looking for model checkpoints...\"

# Extract the directory name from the log file path
LOG_DIR=\$(dirname \"\$BEST_LOG_FILE\")
POTENTIAL_MODEL_DIRS=(
  \"\$LOG_DIR\"
  \"\$LOG_DIR/models\"
  \"\$LOG_DIR/../models\"
)

# Add standard model directories
for dir in \"\${MODEL_DIRS[@]}\"; do
  POTENTIAL_MODEL_DIRS+=(\$(eval echo \$dir))
done

FOUND_MODELS=()
for dir in \"\${POTENTIAL_MODEL_DIRS[@]}\"; do
  if [ -d \"\$dir\" ]; then
    # Look for files containing the epoch number
    MODELS=\$(find \"\$dir\" -name \"*\$BEST_MODEL*.h5\" 2>/dev/null)
    if [ -n \"\$MODELS\" ]; then
      FOUND_MODELS+=(\$MODELS)
      echo \"Found models in \$dir:\"
      echo \"\$MODELS\"
    fi
    
    # Also look for checkpoints with number only
    if [[ \$BEST_MODEL =~ epoch_([0-9]+) ]]; then
      EPOCH_NUM=\${BASH_REMATCH[1]}
      MODELS=\$(find \"\$dir\" -name \"*\$EPOCH_NUM*.h5\" 2>/dev/null)
      if [ -n \"\$MODELS\" ]; then
        FOUND_MODELS+=(\$MODELS)
        echo \"Found models matching epoch \$EPOCH_NUM in \$dir:\"
        echo \"\$MODELS\"
      fi
    fi
    
    # Also look for the best model in relevant directories
    MODELS=\$(find \"\$dir\" -name \"*best*.h5\" 2>/dev/null)
    if [ -n \"\$MODELS\" ]; then
      FOUND_MODELS+=(\$MODELS)
      echo \"Found best models in \$dir:\"
      echo \"\$MODELS\"
    fi
  fi
done

# If we couldn't find exact epoch models, try to find closest checkpoints
if [ \${#FOUND_MODELS[@]} -eq 0 ]; then
  echo \"No exact epoch match found, looking for closest checkpoints...\"
  
  for dir in \"\${POTENTIAL_MODEL_DIRS[@]}\"; do
    if [ -d \"\$dir\" ]; then
      # Prioritize no_leakage models
      MODELS=\$(find \"\$dir\" -name \"*.h5\" | grep -i \"no_leakage\" 2>/dev/null)
      if [ -n \"\$MODELS\" ]; then
        echo \"Found no_leakage models in \$dir:\"
        echo \"\$MODELS\"
        FOUND_MODELS+=(\$MODELS)
      else
        # Otherwise just get any model files
        MODELS=\$(find \"\$dir\" -name \"*.h5\" 2>/dev/null)
        if [ -n \"\$MODELS\" ]; then
          echo \"Found models in \$dir:\"
          echo \"\$MODELS\"
          FOUND_MODELS+=(\$MODELS)
        fi
      fi
    fi
  done
fi

# Print final results
echo \"\"
echo \"=======================================\"
echo \"RESULTS SUMMARY\"
echo \"=======================================\"
echo \"Target accuracy: \$TARGET_ACCURACY%\"
echo \"Closest match found: \$BEST_ACCURACY%\"
echo \"Difference: \$(echo \"sqrt(\$BEST_DIFF)\" | bc -l)%\"
echo \"Log file: \$BEST_LOG_FILE\"
echo \"\"
echo \"Potential model files:\"
for model in \"\${FOUND_MODELS[@]}\"; do
  echo \"\$model\"
done
echo \"=======================================\"

# Create a file with the results
echo \"TARGET_ACCURACY=\$TARGET_ACCURACY\" > ~/target_model_results.txt
echo \"BEST_ACCURACY=\$BEST_ACCURACY\" >> ~/target_model_results.txt
echo \"BEST_LOG_FILE=\$BEST_LOG_FILE\" >> ~/target_model_results.txt
echo \"FOUND_MODELS=\${FOUND_MODELS[*]}\" >> ~/target_model_results.txt
EOF"

# Make the script executable and run it
echo -e "${YELLOW}Executing the script on the EC2 instance...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/find_target_model.sh && ~/find_target_model.sh"

# Download the results file
echo -e "${YELLOW}Downloading results...${NC}"
scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/target_model_results.txt models/branched_no_leakage_84_1/results.txt

# Source the results file
source models/branched_no_leakage_84_1/results.txt

# Download the model automatically
echo -e "${YELLOW}Downloading model file(s)...${NC}"
for model in $FOUND_MODELS; do
    # Extract the filename
    filename=$(basename "$model")
    echo -e "${YELLOW}Downloading $filename...${NC}"
    scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:"$model" models/branched_no_leakage_84_1/
    
    # Check if download was successful
    if [ -f "models/branched_no_leakage_84_1/$filename" ]; then
        echo -e "${GREEN}Successfully downloaded $filename${NC}"
        # Create a symbolic link for best_model.h5
        if [[ "$filename" == *"best"* ]]; then
            echo -e "${YELLOW}Creating link to best_model.h5${NC}"
            ln -sf "$filename" models/branched_no_leakage_84_1/best_model.h5
        fi
    else
        echo -e "${RED}Failed to download $filename${NC}"
    fi
done

# Create model info JSON file
MODEL_INFO_FILE="models/branched_no_leakage_84_1/model_info.json"
echo -e "${YELLOW}Creating model info file...${NC}"
cat > "$MODEL_INFO_FILE" << EOF
{
  "model_description": "Branched network for emotion classification with no data leakage",
  "target_accuracy": ${BEST_ACCURACY},
  "architecture": "Branched network with dynamic padding and no data leakage",
  "input_shape": [null, 100],
  "output_shape": [6],
  "classes": ["anger", "disgust", "fear", "happiness", "sadness", "neutral"],
  "created_date": "$(date +%Y-%m-%d)",
  "training_script": "train_branched_no_leakage.py",
  "notes": "Model trained with 15fps downsampled video and regular audio with OpenSMILE and FaceNet features."
}
EOF

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}PROCESS COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Model(s) downloaded to models/branched_no_leakage_84_1/ directory${NC}"
echo -e "${YELLOW}Model info saved to ${MODEL_INFO_FILE}${NC}"
