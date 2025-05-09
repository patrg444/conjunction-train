#!/bin/bash
# Script to deploy and train the model without data augmentation

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Preparing to deploy model without data augmentation..."

# First, copy the necessary files to the AWS instance
echo "Uploading required files to AWS..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no \
    /Users/patrickgloria/conjunction-train/scripts/train_branched_no_augmentation.py \
    ${USERNAME}@${INSTANCE_IP}:~/emotion_training/scripts/

echo "Setting up the training environment on AWS..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training

# Kill any previous training processes
echo "Stopping any currently running training processes..."
pkill -f "python" || echo "No python processes found"

# Make the script executable
chmod +x scripts/train_branched_no_augmentation.py

# Create a simple launcher script
cat > run_no_augmentation.sh << 'LAUNCHER'
#!/bin/bash

# Clear the current training log
> training_no_aug.log

# Run the training script without augmentation using the conda environment
echo "Starting training without data augmentation..." > training_no_aug.log
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310
echo "Conda environment activated, running script..." >> training_no_aug.log
cd ~/emotion_training
nohup python scripts/train_branched_no_augmentation.py >> training_no_aug.log 2>&1 &
echo "Training started without data augmentation. Check training_no_aug.log for progress."
ps aux | grep python >> training_no_aug.log
LAUNCHER

# Make the launcher executable
chmod +x run_no_augmentation.sh

# Execute the launcher
./run_no_augmentation.sh

echo "Training without data augmentation has been started!"
EOF

echo "Deployment completed!"
echo "The model without data augmentation is now training on AWS."
echo "To monitor the training progress, use:"
echo "cd aws-setup && ./monitor_no_augmentation.sh"

# Create a monitoring script
cat > aws-setup/monitor_no_augmentation.sh << 'MONITOR'
#!/bin/bash
# Script to monitor the training process without data augmentation

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m  TRAINING WITHOUT DATA AUGMENTATION MONITOR\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# Check if training is still running
running=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_no_augmentation | grep -v grep | wc -l")

if [ "$running" -eq "0" ]; then
    echo -e "\033[1;31mTraining process is not running!\033[0m"
    echo "Check if training completed or encountered an error."
    echo ""
fi

# Display the log content with better formatting
echo -e "\033[1;33m=== TRAINING LOG (LATEST ENTRIES) ===\033[0m"
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "\
cat ~/emotion_training/training_no_aug.log | grep -E 'AUGMENTATION|DYNAMIC|Starting|TensorFlow|Processing|Added|Combined|Data augmentation|Enhanced dataset|Class|Epoch|val_accuracy|loss|accuracy|sequence length'"

echo ""
echo -e "\033[1;36m========================================================\033[0m"
echo "For more detailed logs, run:"
echo "cd aws-setup && ssh -i \"${KEY_FILE}\" ec2-user@${INSTANCE_IP} \"tail -50 ~/emotion_training/training_no_aug.log\""
MONITOR

# Make the monitoring script executable
chmod +x aws-setup/monitor_no_augmentation.sh
