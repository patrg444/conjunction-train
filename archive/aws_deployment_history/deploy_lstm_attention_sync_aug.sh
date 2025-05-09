#!/bin/bash
# Deploy the LSTM attention model with synchronized augmentation to AWS EC2

# Set variables
INSTANCE_TYPE="c5.24xlarge"
AMI_ID="ami-0440d3b780d96b29d"  # Amazon Linux
TIMESTAMP=$(date +%Y%m%d%H%M%S)
KEY_NAME="emotion-recognition-key-lstm-attention-sync-aug-${TIMESTAMP}"
INSTANCE_NAME="lstm-attention-sync-aug-${TIMESTAMP}"
SCRIPT_DIR=$(dirname "$0")
WORKING_DIR="${SCRIPT_DIR}/.."
UPLOAD_FILE="ec2-upload-${TIMESTAMP}.tar.gz"
CONNECTION_INFO_FILE="${SCRIPT_DIR}/lstm_attention_model_connection.txt"

echo "================================================================="
echo "     DEPLOYMENT OF LSTM ATTENTION MODEL (SYNCHRONIZED AUGMENTATION)"
echo "================================================================="

# Step 1: Create the EC2 instance
echo "Creating EC2 instance..."
echo "Instance type: ${INSTANCE_TYPE}"
echo "AMI: ${AMI_ID}"
echo "Key pair: ${KEY_NAME}"

# Create key pair
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text > ${SCRIPT_DIR}/${KEY_NAME}.pem
chmod 400 ${SCRIPT_DIR}/${KEY_NAME}.pem

# Create the instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ${AMI_ID} \
    --instance-type ${INSTANCE_TYPE} \
    --key-name ${KEY_NAME} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":30,\"VolumeType\":\"gp2\"}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance created: ${INSTANCE_ID}"

# Save connection details for other scripts to use
echo "INSTANCE_ID=\"${INSTANCE_ID}\"" > ${CONNECTION_INFO_FILE}
echo "KEY_FILE=\"${SCRIPT_DIR}/${KEY_NAME}.pem\"" >> ${CONNECTION_INFO_FILE}

# Wait for the instance to initialize
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids ${INSTANCE_ID}

# Get the public IP address of the instance
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance is running at ${INSTANCE_IP}"
echo "INSTANCE_IP=\"${INSTANCE_IP}\"" >> ${CONNECTION_INFO_FILE}

# Allow time for SSH to be ready
echo "Waiting for SSH to be available..."
while ! nc -z ${INSTANCE_IP} 22 2>/dev/null; do
    sleep 5
    echo -n "."
done
echo "SSH is up"

# Double check SSH accessibility
sleep 5
ssh -o StrictHostKeyChecking=no -i ${SCRIPT_DIR}/${KEY_NAME}.pem ec2-user@${INSTANCE_IP} "echo 'SSH is up and running'"

# Step 2: Prepare the files for upload
echo "Preparing files for upload..."

# Create a temporary directory structure
TMP_DIR=$(mktemp -d)
mkdir -p ${TMP_DIR}/scripts
mkdir -p ${TMP_DIR}/ravdess_features_facenet
mkdir -p ${TMP_DIR}/crema_d_features_facenet
mkdir -p ${TMP_DIR}/models

# Copy necessary files
cp -r ${WORKING_DIR}/scripts/train_branched_attention_sync_aug.py ${TMP_DIR}/scripts/
cp -r ${WORKING_DIR}/scripts/synchronized_data_generator.py ${TMP_DIR}/scripts/
cp -r ${WORKING_DIR}/scripts/sequence_data_generator.py ${TMP_DIR}/scripts/
cp -r ${WORKING_DIR}/requirements.txt ${TMP_DIR}/

# Package RAVDESS and CREMA-D features
cp -r ${WORKING_DIR}/ravdess_features_facenet/* ${TMP_DIR}/ravdess_features_facenet/
cp -r ${WORKING_DIR}/crema_d_features_facenet/* ${TMP_DIR}/crema_d_features_facenet/

# Create tar.gz archive
tar -czf ${UPLOAD_FILE} -C ${TMP_DIR} .

# Step 3: Upload the files to the EC2 instance
echo "Uploading files to EC2 instance..."
echo "This may take a while depending on your internet connection..."
scp -i ${SCRIPT_DIR}/${KEY_NAME}.pem ${UPLOAD_FILE} ec2-user@${INSTANCE_IP}:~

# Step 4: Set up the environment and start training
echo "Setting up the EC2 environment..."
ssh -i ${SCRIPT_DIR}/${KEY_NAME}.pem ec2-user@${INSTANCE_IP} << 'EOF'
echo "==============================================================="
echo "     SETTING UP EC2 ENVIRONMENT FOR LSTM ATTENTION MODEL"
echo "==============================================================="

# Extract the uploaded files
mkdir -p ~/emotion_training
tar -xzf ~/${UPLOAD_FILE} -C ~/emotion_training
cd ~/emotion_training

# Set up directory structure
mkdir -p models/attention_focal_loss_sync_aug

# Install system dependencies
echo "Installing system dependencies..."
sudo dnf -y update
sudo dnf -y install python3 python3-devel git

# Set up Python environment
echo "Setting up Python environment..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.19.5 tensorflow==2.6.0 keras==2.6.0 scipy==1.7.0 scikit-learn==0.24.2 matplotlib==3.4.2 pandas==1.3.0 protobuf==3.19.4 urllib3==1.26.6

# Configure TensorFlow for optimal CPU performance
echo "Configuring TensorFlow for optimal CPU performance..."
export TF_NUM_INTEROP_THREADS=96
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96

# Add these to .bashrc for future sessions
echo "export TF_NUM_INTEROP_THREADS=96" >> ~/.bashrc
echo "export TF_NUM_INTRAOP_THREADS=96" >> ~/.bashrc
echo "export OMP_NUM_THREADS=96" >> ~/.bashrc

# Test the Python environment
echo "Testing Python environment..."
python3 -c "
import numpy as np
import tensorflow as tf
import keras
import scipy
import sklearn
import matplotlib
import pandas as pd

print('Testing Python environment setup...')
print('✓ NumPy imported successfully:', np.__version__)
print('✓ TensorFlow imported successfully:', tf.__version__)
print('✓ Keras imported successfully:', keras.__version__)
print('✓ SciPy imported successfully:', scipy.__version__)
print('✓ Scikit-learn imported successfully:', sklearn.__version__)
print('✓ Matplotlib imported successfully:', matplotlib.__version__)
print('✓ Pandas imported successfully:', pd.__version__)
print('✓ TensorFlow/Keras layers imported successfully')

print('\nEnvironment verification complete. Setup is correct!')
"

echo "Environment setup successful!"
echo "Setup complete!"
EOF

# Step 5: Start the training process
echo "==============================================================="
echo "     STARTING LSTM ATTENTION MODEL TRAINING WITH SYNCHRONIZED AUGMENTATION"
echo "==============================================================="

# Start training in background and redirect output to log file
ssh -i ${SCRIPT_DIR}/${KEY_NAME}.pem ec2-user@${INSTANCE_IP} "cd ~/emotion_training && export TF_NUM_INTEROP_THREADS=96 && export TF_NUM_INTRAOP_THREADS=96 && export OMP_NUM_THREADS=96 && nohup python3 -u scripts/train_branched_attention_sync_aug.py > training_lstm_attention_sync_aug.log 2>&1 &"

# Check if the process is running
sleep 5
TRAINING_PID=$(ssh -i ${SCRIPT_DIR}/${KEY_NAME}.pem ec2-user@${INSTANCE_IP} "pgrep -f train_branched_attention_sync_aug.py")
if [ -z "$TRAINING_PID" ]; then
    echo "ERROR: Training process failed to start."
    echo "Check the log file for details."
    exit 1
else
    echo "Training started in background with process ID: ${TRAINING_PID}"
    echo "You can monitor the logs with: tail -f training_lstm_attention_sync_aug.log"
    echo "Training process is running successfully!"
fi

echo "EC2 setup complete and training started!"
echo "==============================================================="
echo "LSTM Attention Model (Synchronized Augmentation) deployment complete!"
echo "==============================================================="
echo "Instance ID: ${INSTANCE_ID}"
echo "Public IP: ${INSTANCE_IP}"
echo "SSH Key: ${SCRIPT_DIR}/${KEY_NAME}.pem"
echo "To connect: ssh -i ${SCRIPT_DIR}/${KEY_NAME}.pem ec2-user@${INSTANCE_IP}"
echo "To monitor training: ./aws-setup/live_stream_training.sh"
echo "==============================================================="
