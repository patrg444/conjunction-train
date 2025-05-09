#!/bin/bash
# This script will run on the EC2 instance to set up the environment

# Determine the correct user account
if id -u ec2-user >/dev/null 2>&1; then
  USER_ACCOUNT="ec2-user"
elif id -u ubuntu >/dev/null 2>&1; then
  USER_ACCOUNT="ubuntu"
else
  USER_ACCOUNT=$(whoami)
fi

# Upload our setup package
echo "Uploading training package..."
scp -i "emotion-recognition-key-20250322082227.pem" -o StrictHostKeyChecking=no aws-emotion-recognition-training.tar.gz $USER_ACCOUNT@98.82.121.48:~/

# Create an optimized setup script for the instance based on the instance type
cat > cpu_setup.sh << 'CPUEOF'
#!/bin/bash
# CPU-specific optimizations for c5.24xlarge

# Configure TensorFlow for optimal CPU performance
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# For Intel CPUs (c5 instances use Intel CPUs)
export TF_ENABLE_ONEDNN_OPTS=1

# Disable CUDA to ensure CPU usage
export CUDA_VISIBLE_DEVICES=""
CPUEOF

chmod +x cpu_setup.sh

# SSH into the instance and run setup commands
echo "Setting up the instance environment..."
scp -i "emotion-recognition-key-20250322082227.pem" -o StrictHostKeyChecking=no cpu_setup.sh $USER_ACCOUNT@98.82.121.48:~/

ssh -i "emotion-recognition-key-20250322082227.pem" -o StrictHostKeyChecking=no $USER_ACCOUNT@98.82.121.48 << 'ENDSSH'
# Create directories
mkdir -p ~/emotion_training
tar -xzf aws-emotion-recognition-training.tar.gz -C ~/emotion_training
cd ~/emotion_training

# Make scripts executable
chmod +x aws-instance-setup.sh train-on-aws.sh

# Run setup script
echo "Running environment setup script..."
./aws-instance-setup.sh

# Modify train-on-aws.sh to source the CPU optimizations
sed -i '1a source ~/cpu_setup.sh' train-on-aws.sh

# Start training with default parameters
echo "Starting training..."
nohup ./train-on-aws.sh > training.log 2>&1 &
echo "Training started in background. You can check progress with: tail -f training.log"
ENDSSH

# Create a script to check training progress
cat > check_progress.sh << EOF2
#!/bin/bash
ssh -i "emotion-recognition-key-20250322082227.pem" -o StrictHostKeyChecking=no $USER_ACCOUNT@98.82.121.48 "tail -f ~/emotion_training/training.log"
EOF2
chmod +x check_progress.sh

# Create a script to monitor CPU usage
cat > monitor_cpu.sh << EOF3
#!/bin/bash
ssh -i "emotion-recognition-key-20250322082227.pem" -o StrictHostKeyChecking=no $USER_ACCOUNT@98.82.121.48 "top -b -n 1"
EOF3
chmod +x monitor_cpu.sh

# Create a script to download results
cat > download_results.sh << EOF4
#!/bin/bash
mkdir -p results
scp -i "emotion-recognition-key-20250322082227.pem" -r $USER_ACCOUNT@98.82.121.48:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
EOF4
chmod +x download_results.sh

# Create a script to stop/terminate the instance
cat > stop_instance.sh << EOF5
#!/bin/bash
# This script stops or terminates the EC2 instance

echo "Do you want to stop or terminate the instance?"
echo "1) Stop instance (can be restarted later, storage charges apply)"
echo "2) Terminate instance (permanent deletion, no further charges)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo "Stopping instance i-0dd2f787db00b205f..."
        aws ec2 stop-instances --instance-ids "i-0dd2f787db00b205f"
        echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids i-0dd2f787db00b205f"
        ;;
    2)
        echo "Terminating instance i-0dd2f787db00b205f..."
        aws ec2 terminate-instances --instance-ids "i-0dd2f787db00b205f"
        echo "Instance terminated."
        ;;
    *)
        echo "Invalid choice, no action taken."
        ;;
esac
EOF5
chmod +x stop_instance.sh
