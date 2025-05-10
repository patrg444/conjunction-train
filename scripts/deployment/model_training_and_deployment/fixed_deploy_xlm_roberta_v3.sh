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
REMOTE_SCRIPT_PATH="$EC2_PROJECT_DIR/fixed_train_xlm_roberta_script_v2.py"
LOCAL_SCRIPT_PATH="fixed_train_xlm_roberta_script_v2.py"

echo "Deploying XLM-RoBERTa V3 advanced training to EC2 instance at $EC2_IP..."

# Ensure the remote project directory exists
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "mkdir -p $EC2_PROJECT_DIR"

# Copy the training script to the EC2 instance
echo "Copying training script to EC2 instance..."
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" $LOCAL_SCRIPT_PATH $EC2_USER@$EC2_IP:$REMOTE_SCRIPT_PATH

# Create the training launch script on the EC2 instance
echo "Creating training launch script on EC2 instance..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "cat > $EC2_PROJECT_DIR/launch_xlm_roberta_v3.sh" << 'EOF'
#!/bin/bash
set -e

# Check if conda environments are available
conda_envs=$(conda env list 2>/dev/null || echo "No conda")
if [[ $conda_envs == *"humor"* ]]; then
    # Humor environment exists
    if [ -d "$HOME/miniconda3/bin" ]; then
        source "$HOME/miniconda3/bin/activate" humor
    elif [ -d "$HOME/anaconda3/bin" ]; then
        source "$HOME/anaconda3/bin/activate" humor
    fi
    echo "Using conda humor environment"
else
    echo "Conda humor environment not found, using system Python"
fi

cd ~/humor_detection

# Install required packages if not already installed - use latest versions compatible with the system
pip install --quiet transformers pandas scikit-learn matplotlib seaborn tensorboard tqdm
pip install --quiet torch --no-cache-dir
pip install --quiet pytorch-lightning --no-cache-dir

# Define paths for datasets
TRAIN_MANIFEST="datasets/manifests/humor/train_humor_with_text.csv"
VAL_MANIFEST="datasets/manifests/humor/val_humor_with_text.csv"

# Define training parameters 
MODEL_NAME="xlm-roberta-large"
LOG_DIR="training_logs_humor"
EXP_NAME="xlm-roberta-large_v3_optimized"
BATCH_SIZE=8  
ACCUMULATION_STEPS=4  # Effective batch size = 32
EPOCHS=50
LR=3e-5
LABEL_SMOOTHING=0.1
LAYER_DECAY=0.95
NUM_WORKERS=4

echo "Starting XLM-RoBERTa V3 advanced training at $(date)"
echo "Model: $MODEL_NAME"
echo "Effective batch size: $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "Learning rate: $LR"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Layer decay: $LAYER_DECAY"
echo "Epochs: $EPOCHS"

# Launch training with nohup to keep it running after logout
nohup python fixed_train_xlm_roberta_script_v2.py \
    --train_manifest $TRAIN_MANIFEST \
    --val_manifest $VAL_MANIFEST \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --log_dir $LOG_DIR \
    --exp_name $EXP_NAME \
    --fp16 \
    --class_balancing \
    --label_smoothing $LABEL_SMOOTHING \
    --layer_decay $LAYER_DECAY \
    --accumulation_steps $ACCUMULATION_STEPS \
    --scheduler cosine \
    > xlm_roberta_v3_training.log 2>&1 &

echo "Training started in background. Check xlm_roberta_v3_training.log for progress."
echo "Training PID: $!"
echo $! > xlm_roberta_v3_training.pid
EOF

# Make the launch script executable
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "chmod +x $EC2_PROJECT_DIR/launch_xlm_roberta_v3.sh"

# Run the launch script
echo "Launching XLM-RoBERTa V3 advanced training on EC2 instance..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "$EC2_PROJECT_DIR/launch_xlm_roberta_v3.sh"

echo "Deployment complete! XLM-RoBERTa V3 training is now running on the EC2 instance."
echo "Use 'monitor_xlm_roberta_v3.sh' to check training progress."

# Create the monitoring script locally
cat > monitor_xlm_roberta_v3.sh << 'EOF'
#!/bin/bash

# Get EC2 instance IP from file or environment variable
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
elif [ -n "$EC2_INSTANCE_IP" ]; then
    EC2_IP=$EC2_INSTANCE_IP
else
    echo "Error: EC2 instance IP not found. Please set EC2_INSTANCE_IP or create aws_instance_ip.txt."
    exit 1
fi

EC2_USER="ubuntu"
EC2_PROJECT_DIR="~/humor_detection"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "Checking XLM-RoBERTa V3 training status..."
echo "-----------------------------------------"

# Check if training process is still running
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -f $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid ]; then 
    PID=\$(cat $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid)
    if ps -p \$PID > /dev/null; then 
        echo \"Training is running (PID: \$PID)\"
    else 
        echo \"Training process (PID: \$PID) is not running. Check logs for completion or errors.\"
    fi
else 
    echo \"No PID file found. Training may not have been started.\"
fi"

# Show the last 50 lines of the log file
echo "-----------------------------------------"
echo "Recent training log:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "tail -n 50 $EC2_PROJECT_DIR/xlm_roberta_v3_training.log"

# Check GPU usage
echo "-----------------------------------------"
echo "GPU Usage:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "nvidia-smi"

# Check disk space
echo "-----------------------------------------"
echo "Disk Space:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "df -h | grep -E '/$|/home'"
EOF

# Make the monitoring script executable
chmod +x monitor_xlm_roberta_v3.sh

echo "Created monitoring script: monitor_xlm_roberta_v3.sh"
echo "Run './monitor_xlm_roberta_v3.sh' to check training progress."
