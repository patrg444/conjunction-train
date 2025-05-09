#!/bin/bash
# Script to deploy and run cross-validation on the EC2 server

# Check if the AWS instance IP is available
if [ ! -f "aws_instance_ip.txt" ]; then
  echo "Error: EC2 host IP not found. Please create aws_instance_ip.txt file."
  exit 1
fi

EC2_USER="ubuntu"
EC2_HOST=$(cat aws_instance_ip.txt)

echo "===== Deploying Cross-Validation Evaluation to EC2 ====="

# Create remote directory for evaluation
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "mkdir -p ~/audio_emotion/evaluation"

# Copy the evaluation scripts to the server
echo "Uploading evaluation scripts..."
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no evaluate_wav2vec_full_dataset.py $EC2_USER@$EC2_HOST:~/audio_emotion/evaluation/
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no verify_emotion_labels.py $EC2_USER@$EC2_HOST:~/audio_emotion/evaluation/

# Make scripts executable
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "chmod +x ~/audio_emotion/evaluation/evaluate_wav2vec_full_dataset.py ~/audio_emotion/evaluation/verify_emotion_labels.py"

# Create results directory on the server
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "mkdir -p ~/audio_emotion/evaluation/results"

# Run the evaluation with cross-validation
echo "Running cross-validation on server..."
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "cd ~/audio_emotion/evaluation && python3 evaluate_wav2vec_full_dataset.py --model-path ~/audio_emotion/checkpoints/best_model_v9.h5 --label-classes ~/audio_emotion/checkpoints/label_classes_v9.npy --max-length 221 --output-dir ~/audio_emotion/evaluation/results --cross-validation --n-folds 5"

# Create local results directory
mkdir -p results/server_cross_validation

# Download the results
echo "Downloading results..."
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:~/audio_emotion/evaluation/results/cross_validation_results.json results/server_cross_validation/
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:~/audio_emotion/evaluation/results/cv_confusion_matrix.png results/server_cross_validation/
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:~/audio_emotion/evaluation/results/label_classes.npy results/server_cross_validation/

# Check if results were downloaded successfully
if [ -f "results/server_cross_validation/cross_validation_results.json" ]; then
  echo "Cross-validation results downloaded successfully."
  echo "Results saved to results/server_cross_validation/"
else
  echo "Warning: Could not download cross-validation results."
fi

echo "===== Cross-Validation Deployment Complete ====="
