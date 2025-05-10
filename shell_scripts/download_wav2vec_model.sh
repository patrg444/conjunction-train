#!/bin/bash
# Script to download trained wav2vec model and history files from EC2

# Settings
SSH_KEY="$1"
AWS_IP="$2"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"
EC2_PROJECT_PATH="/home/$SSH_USER/audio_emotion"
LOCAL_DIR="wav2vec_models"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
  echo "Usage: $0 <path-to-key.pem> <ec2-ip-address>"
  echo "Example: $0 ~/Downloads/gpu-key.pem 54.162.134.77"
  exit 1
}

# Check parameters
if [[ -z "$SSH_KEY" || -z "$AWS_IP" ]]; then
  echo -e "${RED}Error: Missing required parameters${NC}"
  show_usage
fi

# Check if key file exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo -e "${RED}Error: SSH key file does not exist: $SSH_KEY${NC}"
  exit 1
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}          DOWNLOAD WAV2VEC EMOTION RECOGNITION MODEL        ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}$AWS_IP${NC}"
echo -e "Remote path: ${CYAN}$EC2_PROJECT_PATH/checkpoints/${NC}"
echo -e "Local download directory: ${CYAN}$LOCAL_DIR${NC}"
echo -e "${BLUE}============================================================${NC}"

# Create local directory if it doesn't exist
echo -e "\n${YELLOW}Creating local directory for downloads...${NC}"
mkdir -p "$LOCAL_DIR"

# Check if training is completed or still in progress
echo -e "\n${YELLOW}Checking if training is still in progress...${NC}"
is_running=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f 'train_wav2vec_audio_only.py' || echo 'not_running'")

if [[ "$is_running" != "not_running" ]]; then
  echo -e "${YELLOW}Warning: Training is still running (PID: $is_running)${NC}"
  echo -e "You can download checkpoints while training is in progress,"
  echo -e "but the final model might not be available yet."
  echo -e "Consider waiting for training to complete, or use early-stopping checkpoints."
else
  echo -e "${GREEN}✅ Training is not running. Proceeding with download.${NC}"
fi

# List available model files
echo -e "\n${YELLOW}Checking available model files...${NC}"
available_files=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ls -la $EC2_PROJECT_PATH/checkpoints/wav2vec_audio_only_*")

if [[ -z "$available_files" ]]; then
  echo -e "${RED}No model files found! Training may not have generated checkpoints yet.${NC}"
  echo -e "${YELLOW}Checking logs to verify training status...${NC}"
  
  # Check if log file exists
  log_file="$EC2_PROJECT_PATH/train_wav2vec_audio_only.log"
  log_exists=$(ssh -i "$SSH_KEY" "$SSH_HOST" "test -f $log_file && echo 'exists' || echo 'missing'")
  
  if [[ "$log_exists" == "exists" ]]; then
    # Check for completion message
    completed=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -q 'Training completed' $log_file && echo 'completed' || echo 'in_progress'")
    
    if [[ "$completed" == "completed" ]]; then
      echo -e "${RED}Training completed but no model files found. Something went wrong.${NC}"
    else
      echo -e "${YELLOW}Training may still be in progress or failed. Check the logs:${NC}"
      echo -e "${CYAN}ssh -i $SSH_KEY $SSH_HOST \"tail -n 50 $log_file\"${NC}"
    fi
  else
    echo -e "${RED}Log file not found. Training may not have started.${NC}"
  fi
  
  exit 1
fi

echo -e "${GREEN}Found model files:${NC}"
echo -e "$available_files"

# Download best model and history files
echo -e "\n${YELLOW}Downloading best model and history files...${NC}"

# Find the best model file (highest validation accuracy)
best_models=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/checkpoints -name 'wav2vec_audio_only_*_best.h5'")

if [[ -z "$best_models" ]]; then
  echo -e "${YELLOW}No '*_best.h5' models found, trying to find final model...${NC}"
  final_models=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/checkpoints -name 'wav2vec_audio_only_*_final.h5'")
  
  if [[ -z "$final_models" ]]; then
    echo -e "${RED}No final models found either. Training may not have completed successfully.${NC}"
    exit 1
  fi
  
  model_files="$final_models"
else
  model_files="$best_models"
fi

# Find history files
history_files=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/checkpoints -name 'wav2vec_audio_only_*_history.json'")

if [[ -z "$history_files" ]]; then
  echo -e "${YELLOW}No training history files found.${NC}"
else
  echo -e "${GREEN}Found training history files:${NC}"
  echo -e "$history_files"
fi

# Also find architecture files
arch_files=$(ssh -i "$SSH_KEY" "$SSH_HOST" "find $EC2_PROJECT_PATH/checkpoints -name 'wav2vec_audio_only_*_architecture.json'")

# Download all files
echo -e "\n${YELLOW}Starting download...${NC}"

# Download model files
for file in $model_files; do
  filename=$(basename "$file")
  echo -e "${CYAN}Downloading $filename...${NC}"
  scp -i "$SSH_KEY" "$SSH_HOST:$file" "$LOCAL_DIR/"
done

# Download history files
for file in $history_files; do
  filename=$(basename "$file")
  echo -e "${CYAN}Downloading $filename...${NC}"
  scp -i "$SSH_KEY" "$SSH_HOST:$file" "$LOCAL_DIR/"
done

# Download architecture files if available
for file in $arch_files; do
  filename=$(basename "$file")
  echo -e "${CYAN}Downloading $filename...${NC}"
  scp -i "$SSH_KEY" "$SSH_HOST:$file" "$LOCAL_DIR/"
done

# Check if download was successful
if ls "$LOCAL_DIR"/*.h5 >/dev/null 2>&1; then
  echo -e "\n${GREEN}✅ Models downloaded successfully to $LOCAL_DIR${NC}"
  
  # Suggest next steps for evaluation
  echo -e "\n${YELLOW}Suggested next steps:${NC}"
  
  # Check if we have history files to plot
  if ls "$LOCAL_DIR"/*_history.json >/dev/null 2>&1; then
    history_file=$(ls -t "$LOCAL_DIR"/*_history.json | head -1)
    echo -e "1. Plot training curves with:"
    echo -e "${CYAN}   python scripts/plot_training_curve.py --history_file $history_file --metric both${NC}"
  fi
  
  echo -e "2. For model inference, you can load the model with:"
  echo -e "${CYAN}   from tensorflow.keras.models import load_model${NC}"
  echo -e "${CYAN}   model = load_model('$LOCAL_DIR/$(basename $(ls -t "$LOCAL_DIR"/*.h5 | head -1))')${NC}"
  
  echo -e "3. Consider exporting to SavedModel format for deployment:"
  echo -e "${CYAN}   model.save('wav2vec_audio_only_saved', include_optimizer=False)${NC}"
else
  echo -e "\n${RED}⚠️ Download failed or no models were found.${NC}"
fi
