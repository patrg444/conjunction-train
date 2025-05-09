#!/bin/bash
# Script to deploy and train the model with dynamic sequence padding

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Preparing to deploy dynamic sequence padding model..."

# First, copy the necessary files to the AWS instance
echo "Uploading required files to AWS..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no \
    /Users/patrickgloria/conjunction-train/scripts/train_branched_with_dynamic_padding.py \
    /Users/patrickgloria/conjunction-train/scripts/sequence_data_generator.py \
    /Users/patrickgloria/conjunction-train/scripts/train_branched_dynamic_funcs.py \
    ${USERNAME}@${INSTANCE_IP}:~/emotion_training/scripts/

echo "Setting up the training environment on AWS..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training

# Kill any previous training processes
echo "Stopping any currently running training processes..."
pkill -f "python" || echo "No python processes found"

# Make the scripts executable
chmod +x scripts/train_branched_with_dynamic_padding.py
chmod +x scripts/sequence_data_generator.py
chmod +x scripts/train_branched_dynamic_funcs.py

# Create a simple launcher script
cat > run_dynamic_padding.sh << 'LAUNCHER'
#!/bin/bash

# Clear the current training log
> training_dynamic.log

# Run the dynamic padding training script using the conda environment
echo "Starting training with dynamic sequence padding..." > training_dynamic.log
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310
echo "Conda environment activated, running script..." >> training_dynamic.log
cd ~/emotion_training
nohup python scripts/train_branched_with_dynamic_padding.py >> training_dynamic.log 2>&1 &
echo "Training started with dynamic sequence padding. Check training_dynamic.log for progress."
ps aux | grep python >> training_dynamic.log
LAUNCHER

# Make the launcher executable
chmod +x run_dynamic_padding.sh

# Execute the launcher
./run_dynamic_padding.sh

echo "Dynamic sequence padding training has been started!"
EOF

echo "Deployment completed!"
echo "The model with dynamic sequence padding is now training on AWS."
echo "To monitor the training progress, use:"
echo "cd aws-setup && ./monitor_dynamic_padding.sh"

# Create a monitoring script for the dynamic padding training
cat > monitor_dynamic_padding.sh << 'MONITOR'
#!/bin/bash
# Script to monitor the dynamic padding training progress

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m  DYNAMIC SEQUENCE PADDING TRAINING MONITOR\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# Check if training is still running
running=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_with_dynamic_padding | grep -v grep | wc -l")

if [ "$running" -eq "0" ]; then
    echo -e "\033[1;31mDynamic padding training process is not running!\033[0m"
    echo "Check if training completed or encountered an error."
    echo ""
fi

# Display the log content with better formatting
echo -e "\033[1;33m=== TRAINING LOG (LATEST ENTRIES) ===\033[0m"
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "\
cat ~/emotion_training/training_dynamic.log | grep -E 'DYNAMIC|Starting|TensorFlow|Processing|Added|Combined|Data augmentation|Enhanced dataset|Class|Epoch|val_accuracy|loss|accuracy|sequence length|PADDING'"

echo ""
echo -e "\033[1;36m========================================================\033[0m"
echo "For more detailed logs, run:"
echo "cd aws-setup && ssh -i \"${KEY_FILE}\" ec2-user@${INSTANCE_IP} \"tail -50 ~/emotion_training/training_dynamic.log\""
MONITOR

# Make the monitoring script executable
chmod +x monitor_dynamic_padding.sh
