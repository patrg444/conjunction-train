#!/bin/bash
set -e

# Get EC2 instance IP from file or environment variable
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
elif [ -n "$EC2_INSTANCE_IP" ]; then
    EC2_IP=$EC2_INSTANCE_IP
else
    echo "Error: EC2 instance IP not found. Please set EC2_INSTANCE_IP or create aws_instance_ip.txt."
    exit 1
fi

# Define EC2 username and directories
EC2_USER="ubuntu"
EC2_PROJECT_DIR="~/humor_detection"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
REMOTE_MODEL_PATH="$EC2_PROJECT_DIR/training_logs_humor/xlm-roberta-large_v3_optimized/final_model"
LOCAL_MODEL_DIR="models/xlm-roberta-v3"

# Ensure local directory exists
mkdir -p $LOCAL_MODEL_DIR

echo "Checking if XLM-RoBERTa V3 model training is complete..."

# Check if the model directory exists on the remote server
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -d $REMOTE_MODEL_PATH ]; then
    echo 'Model directory found. Proceeding with download.'
    exit 0
else
    echo 'Model directory not found. Training may not be complete yet.'
    exit 1
fi" || { echo "Training is not yet complete. Try again later."; exit 1; }

# Check if the model files exist
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -f $REMOTE_MODEL_PATH/pytorch_model.bin ]; then
    echo 'Model files found. Ready for download.'
    exit 0
else
    echo 'Model files not found. Training may not be complete yet.'
    exit 1
fi" || { echo "Model files not found. Try again later."; exit 1; }

echo "Downloading XLM-RoBERTa V3 model from EC2 instance..."
echo "This may take a while depending on your connection speed."

# Tar the model directory for faster transfer
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "cd $EC2_PROJECT_DIR && tar -czf xlm-roberta-v3-model.tar.gz -C training_logs_humor/xlm-roberta-large_v3_optimized/final_model ."

# Download the tar file
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP:$EC2_PROJECT_DIR/xlm-roberta-v3-model.tar.gz $LOCAL_MODEL_DIR/

# Remove the remote tar file
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "rm $EC2_PROJECT_DIR/xlm-roberta-v3-model.tar.gz"

# Extract the model
echo "Extracting model files..."
tar -xzf $LOCAL_MODEL_DIR/xlm-roberta-v3-model.tar.gz -C $LOCAL_MODEL_DIR
rm $LOCAL_MODEL_DIR/xlm-roberta-v3-model.tar.gz

# Download validation metrics
echo "Downloading training metrics and logs..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -d $EC2_PROJECT_DIR/training_logs_humor/xlm-roberta-large_v3_optimized ]; then
    cd $EC2_PROJECT_DIR
    find training_logs_humor/xlm-roberta-large_v3_optimized -name \"metrics_epoch_*.json\" -o -name \"confusion_matrix_epoch_*.png\" | tar -czf xlm-roberta-v3-metrics.tar.gz -T -
fi"

# Download the metrics tar file if it exists
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -f $EC2_PROJECT_DIR/xlm-roberta-v3-metrics.tar.gz ]; then
    exit 0
else
    exit 1
fi" && {
    mkdir -p $LOCAL_MODEL_DIR/metrics
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP:$EC2_PROJECT_DIR/xlm-roberta-v3-metrics.tar.gz $LOCAL_MODEL_DIR/metrics/
    tar -xzf $LOCAL_MODEL_DIR/metrics/xlm-roberta-v3-metrics.tar.gz -C $LOCAL_MODEL_DIR/metrics/ --strip-components=3
    rm $LOCAL_MODEL_DIR/metrics/xlm-roberta-v3-metrics.tar.gz
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "rm $EC2_PROJECT_DIR/xlm-roberta-v3-metrics.tar.gz"
    echo "Training metrics downloaded to $LOCAL_MODEL_DIR/metrics/"
} || echo "No metrics files found to download."

echo "Checking best validation performance..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "cd $EC2_PROJECT_DIR && grep -A 5 '\"val_f1\":' training_logs_humor/xlm-roberta-large_v3_optimized/metrics_epoch_*.json | sort -n -k3 | tail -n 1"

# Download the training log
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP:$EC2_PROJECT_DIR/xlm_roberta_v3_training.log $LOCAL_MODEL_DIR/training_log.txt

echo "Download complete! Model files and metrics saved to $LOCAL_MODEL_DIR/"
echo "To use the model:"
echo "from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification"
echo "tokenizer = XLMRobertaTokenizer.from_pretrained('$LOCAL_MODEL_DIR')"
echo "model = XLMRobertaForSequenceClassification.from_pretrained('$LOCAL_MODEL_DIR')"
