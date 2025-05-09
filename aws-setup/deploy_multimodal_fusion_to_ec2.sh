#!/bin/bash
# Deploy and run multimodal fusion model training on EC2
# This script transfers the necessary files to EC2 and starts the training

set -e

# Configuration
EC2_USER="ubuntu"
EC2_IP=$(cat aws_instance_ip.txt)
KEY_FILE="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_CODE_DIR="/home/ubuntu/humor_detection"
LOCAL_CONFIG_DIR="configs"
LOCAL_MODELS_DIR="models"
LOCAL_DATALOADERS_DIR="dataloaders"
LOCAL_SCRIPTS_DIR="scripts"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${GREEN}===================================================="
echo -e "      Deploying Multimodal Fusion to EC2"
echo -e "====================================================${NC}"

# Check if IP address file exists
if [ ! -f aws_instance_ip.txt ]; then
  echo -e "${RED}Error: aws_instance_ip.txt not found. Please create it with your EC2 instance IP.${NC}"
  exit 1
fi

# Check SSH connection
echo -e "${YELLOW}Testing SSH connection to EC2...${NC}"
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "echo Connected successfully" || { echo -e "${RED}SSH connection failed!${NC}"; exit 1; }

# Create directory structure on EC2
echo -e "${YELLOW}Creating directory structure on EC2...${NC}"
ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "mkdir -p ${EC2_CODE_DIR}/{configs,models,dataloaders,scripts,embeddings/{text,audio,video},training_logs_humor}"

# Sync code to EC2
echo -e "${YELLOW}Syncing code to EC2...${NC}"

# Sync configs
echo -e "Syncing configs..."
rsync -avz -e "ssh -i $KEY_FILE" ${LOCAL_CONFIG_DIR}/model_checkpoint_paths.yaml ${EC2_USER}@${EC2_IP}:${EC2_CODE_DIR}/configs/

# Sync models
echo -e "Syncing model code..."
rsync -avz -e "ssh -i $KEY_FILE" ${LOCAL_MODELS_DIR}/fusion_model.py ${EC2_USER}@${EC2_IP}:${EC2_CODE_DIR}/models/

# Sync dataloaders
echo -e "Syncing dataloader code..."
rsync -avz -e "ssh -i $KEY_FILE" ${LOCAL_DATALOADERS_DIR}/fusion_dataset.py ${EC2_USER}@${EC2_IP}:${EC2_CODE_DIR}/dataloaders/

# Sync scripts
echo -e "Syncing scripts..."
rsync -avz -e "ssh -i $KEY_FILE" \
  ${LOCAL_SCRIPTS_DIR}/generate_multimodal_manifest.py \
  ${LOCAL_SCRIPTS_DIR}/extract_multimodal_embeddings.py \
  ${LOCAL_SCRIPTS_DIR}/train_multimodal_fusion.py \
  ${EC2_USER}@${EC2_IP}:${EC2_CODE_DIR}/scripts/

# Create launcher script on EC2
echo -e "${YELLOW}Creating launcher script on EC2...${NC}"
cat << EOL | ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "cat > ${EC2_CODE_DIR}/run_multimodal_fusion.sh"
#!/bin/bash
# Script to run multimodal fusion on EC2
set -e

cd ${EC2_CODE_DIR}

# Activate conda environment (if using conda)
# source activate pytorch_p38

# Step 1: Create manifest
echo "Step 1: Generating multimodal manifest..."
python scripts/generate_multimodal_manifest.py --ur_funny_dir /home/ubuntu/datasets/ur_funny --output_manifest datasets/manifests/humor/multimodal_humor.csv

# Step 2: Extract embeddings
echo "Step 2: Extracting embeddings..."
python scripts/extract_multimodal_embeddings.py --manifest datasets/manifests/humor/multimodal_humor.csv --config configs/model_checkpoint_paths.yaml

# Step 3: Train fusion model
echo "Step 3: Training fusion model..."
python scripts/train_multimodal_fusion.py \
  --manifest datasets/manifests/humor/multimodal_humor.csv \
  --config configs/model_checkpoint_paths.yaml \
  --fusion_strategy attention \
  --hidden_dim 512 \
  --output_dim 128 \
  --batch_size 32 \
  --epochs 20 \
  --early_stopping \
  --class_weights \
  --experiment_name multimodal_attention_fusion \
  --version v1 \
  --precision 16 \
  --gpus 1

echo "Training complete!"
EOL

# Make the launcher script executable
ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "chmod +x ${EC2_CODE_DIR}/run_multimodal_fusion.sh"

# Create a monitoring script on EC2
echo -e "${YELLOW}Creating monitoring script on EC2...${NC}"
cat << EOL | ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "cat > ${EC2_CODE_DIR}/monitor_fusion_training.sh"
#!/bin/bash
# Script to monitor fusion training on EC2
set -e

cd ${EC2_CODE_DIR}

# Start training in background
nohup ./run_multimodal_fusion.sh > fusion_training.log 2>&1 &
PID=$!

echo "Training started with PID: \$PID"
echo "Monitoring log file. Press Ctrl+C to stop monitoring (training will continue)."
echo "---------------------------------------------"

# Keep monitoring the log file
tail -f fusion_training.log
EOL

# Make the monitoring script executable
ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} "chmod +x ${EC2_CODE_DIR}/monitor_fusion_training.sh"

# Create a download script locally
echo -e "${YELLOW}Creating model download script locally...${NC}"
cat > download_fusion_model.sh << EOL
#!/bin/bash
# Script to download trained multimodal fusion model from EC2
set -e

EC2_USER="ubuntu"
EC2_IP=\$(cat aws_instance_ip.txt)
KEY_FILE="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_MODEL_DIR="/home/ubuntu/humor_detection/training_logs_humor/multimodal_attention_fusion_v1/final_model"
LOCAL_MODEL_DIR="trained_models/multimodal_fusion"

mkdir -p \${LOCAL_MODEL_DIR}

echo "Downloading trained model from EC2..."
rsync -avz -e "ssh -i \$KEY_FILE" \${EC2_USER}@\${EC2_IP}:\${EC2_MODEL_DIR}/* \${LOCAL_MODEL_DIR}/

echo "Model downloaded to \${LOCAL_MODEL_DIR}"
EOL

# Make the download script executable
chmod +x download_fusion_model.sh

echo -e "${GREEN}Deployment setup complete!${NC}"
echo -e "To start training on EC2, run: ${YELLOW}ssh -i \"$KEY_FILE\" ${EC2_USER}@${EC2_IP} \"cd ${EC2_CODE_DIR} && ./monitor_fusion_training.sh\"${NC}"
echo -e "After training, download the model with: ${YELLOW}./download_fusion_model.sh${NC}"
