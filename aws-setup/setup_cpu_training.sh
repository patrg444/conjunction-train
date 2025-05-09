#!/bin/bash
# Script to set up and run CPU-optimized training on the EC2 instance

# Set variables for the instance
INSTANCE_IP="98.82.121.48"
KEY_FILE="emotion-recognition-key-20250322082227.pem"
CURRENT_DIR=$(pwd)

echo "Preparing to set up CPU-optimized training on instance at ${INSTANCE_IP}..."

# Prepare the CPU optimization script
cat > cpu_optimized_train.sh << 'SCRIPT'
#!/bin/bash
# CPU-optimized training script

# CPU optimization environment variables
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

# Run the training
cd ~/emotion_training
python scripts/train_branched_6class.py \
  --ravdess-dir ravdess_features_facenet \
  --cremad-dir crema_d_features_facenet \
  --learning-rate 1e-4 \
  --epochs 50 \
  --batch-size 128 \
  --model-dir models/branched_6class \
  --use-class-weights
SCRIPT
chmod +x cpu_optimized_train.sh

# First create a tar archive of the project (excluding large directories we don't need)
echo "Creating tarball of project..."
cd "${CURRENT_DIR}/.."
tar --exclude="node_modules" \
    --exclude=".git" \
    --exclude="temp_*" \
    --exclude="__pycache__" \
    -czvf aws-setup/project.tar.gz \
    scripts/ \
    ravdess_features_facenet/ \
    crema_d_features_facenet/ \
    requirements.txt \
    config/

cd "${CURRENT_DIR}"

# Transfer the necessary files to the instance
echo "Transferring project files to instance..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no cpu_optimized_train.sh ec2-user@${INSTANCE_IP}:~/
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no project.tar.gz ec2-user@${INSTANCE_IP}:~/

# SSH into the instance and set up the environment
echo "Setting up the instance environment..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ec2-user@${INSTANCE_IP} << 'ENDSSH'
# Create project directory
mkdir -p ~/emotion_training

# Extract the project files
tar -xzvf project.tar.gz -C ~/emotion_training

# Create model directory
mkdir -p ~/emotion_training/models/branched_6class

# Install dependencies
echo "Installing Python dependencies..."
pip install -r ~/emotion_training/requirements.txt

# Set up TensorFlow optimizations for CPU training
echo "Setting up TensorFlow optimizations..."
pip install tensorflow==2.9.0  # Specify version for stability

# Start the training
echo "Starting training. This will run in the background..."
nohup ~/cpu_optimized_train.sh > training.log 2>&1 &

echo "Training started. You can check the progress with: tail -f training.log"
ENDSSH

echo "Setup complete! The training is now running on the EC2 instance."
echo "Use the following commands to check progress and manage the instance:"
echo "  1. Check training progress: aws-setup/check_progress.sh"
echo "  2. Monitor CPU usage: aws-setup/monitor_cpu.sh"
echo "  3. Download results when done: aws-setup/download_results.sh"
echo "  4. Terminate the instance when done: aws-setup/stop_instance.sh"
