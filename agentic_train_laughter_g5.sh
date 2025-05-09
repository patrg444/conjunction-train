#!/usr/bin/env bash
# Agentic Playbook: Full Audio-Video + Laughter Training Run 
# This script automates the entire workflow for deploying, training,
# monitoring, and evaluating the laughter detection model on AWS g5.2xlarge.

set -e   # Exit on error
set -o pipefail

# ====================================================================
# 0. PRE-FLIGHT CONSTANTS
# ====================================================================
AWS_IP="18.208.166.91"                # g5.2xlarge public IP
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # path to PEM / private key (matches EC2 key-pair name "gpu-key")
SSH_USER="ubuntu"                     # default user
SSH_HOST="$SSH_USER@$AWS_IP"
LOCAL_REPO="$PWD"                     # project root
EPOCHS=100 
BATCH=256 
MAXLEN=45 
LW=0.3                               # laugh-loss weight
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOCAL_REPO}/agentic_train_${TIMESTAMP}.log"

# Setup logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== STARTING AGENTIC WORKFLOW AT $(date) ==="
echo "Log file: $LOG_FILE"

# ====================================================================
# 1. VERIFY SSH ACCESS
# ====================================================================
echo "Step 1: Verifying SSH access to AWS instance..."
ssh_success=false

# Verify key file exists
if [ ! -f "$SSH_KEY" ]; then
  echo "ERROR: SSH key file not found at $SSH_KEY"
  echo "Please ensure the correct key file path is specified in this script."
  exit 1
fi

# Verify key file permissions
current_perms=$(stat -f "%p" "$SSH_KEY")
if [[ "$current_perms" != "100600" ]]; then
  echo "WARNING: SSH key file has incorrect permissions (should be 600)."
  echo "Attempting to fix permissions..."
  chmod 600 "$SSH_KEY"
  if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to set correct permissions on key file."
    exit 1
  fi
  echo "Key file permissions fixed."
fi

for i in {1..3}; do
  echo "SSH access attempt $i of 3..."
  if ssh -v -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_HOST" "echo ok"; then
    ssh_success=true
    echo "SSH access successful!"
    break
  else
    echo "SSH access attempt failed. Retrying in 10 seconds..."
    sleep 10
  fi
done

if [ "$ssh_success" = false ]; then
  echo "ERROR: Failed to establish SSH access after 3 attempts. Aborting."
  echo ""
  echo "Please check the following:"
  echo "1. Verify the key pair name used to launch the EC2 instance (currently 'new-key')"
  echo "2. Confirm the SSH key file ($SSH_KEY) is the correct private key for this instance"
  echo "3. Ensure security groups allow SSH access from your current IP address"
  echo "4. Check EC2 instance console output for any SSH service issues"
  echo ""
  echo "For more detailed troubleshooting, refer to EC2_CONNECTION_ISSUE.md"
  exit 1
fi

# ====================================================================
# 2. CHECK DATA PRESENCE ON EC2
# ====================================================================
echo "Step 2: Checking for data presence on EC2 instance..."
data_present=false

check_data_status=$(ssh -i "$SSH_KEY" "$SSH_HOST" <<'REMOTE'
set -e
BASE=~/emotion-recognition
test -d "$BASE" || { echo "NEWDIR"; exit 42; }
ravdess_size=$(du -sm $BASE/ravdess_features_facenet 2>/dev/null | cut -f1 || echo "0")
echo "RAVDESS size: ${ravdess_size}MB"
[ "$ravdess_size" -gt 1500 ] || { echo "NEWDIR"; exit 42; }
crema_size=$(du -sm $BASE/crema_d_features_facenet 2>/dev/null | cut -f1 || echo "0")
echo "CREMA-D size: ${crema_size}MB"
[ "$crema_size" -gt 900 ] || { echo "NEWDIR"; exit 42; }
stat_files=$(find $BASE/models/dynamic_padding_no_leakage -name "*normalization_stats.pkl" 2>/dev/null | wc -l || echo "0")
echo "Normalization stats files: $stat_files"
[ "$stat_files" -ge 2 ] || { echo "NEWDIR"; exit 42; }
echo "DATA_PRESENT"
exit 0
REMOTE
)
echo "$check_data_status"

if echo "$check_data_status" | grep -q "DATA_PRESENT"; then
  data_present=true
  echo "Data already present on EC2 instance. Skipping deployment."
else
  echo "Data not present or incomplete on EC2 instance. Proceeding with deployment."
fi

# ====================================================================
# 3. DEPLOY CODE + DATA (if needed)
# ====================================================================
if [ "$data_present" = false ]; then
  echo "Step 3: Deploying code and data to EC2 instance..."
  cd "$LOCAL_REPO"
  ./deploy_to_aws_g5.sh
  
  # Check for successful deployment
  if ! grep -q "Deployment complete!" "$LOG_FILE"; then
    echo "ERROR: Deployment failed. Check the log file for details."
    exit 1
  fi
  echo "Deployment completed successfully."
  
  # ====================================================================
  # 4. POST-DEPLOY VERIFY
  # ====================================================================
  echo "Step 4: Performing post-deployment verification..."
  post_check=$(ssh -i "$SSH_KEY" "$SSH_HOST" <<'REMOTE'
  set -e
  BASE=~/emotion-recognition
  ravdess_size=$(du -sm $BASE/ravdess_features_facenet 2>/dev/null | cut -f1 || echo "0")
  echo "RAVDESS size: ${ravdess_size}MB"
  [ "$ravdess_size" -gt 1500 ] || exit 42
  crema_size=$(du -sm $BASE/crema_d_features_facenet 2>/dev/null | cut -f1 || echo "0")
  echo "CREMA-D size: ${crema_size}MB"
  [ "$crema_size" -gt 900 ] || exit 42
  stat_files=$(find $BASE/models/dynamic_padding_no_leakage -name "*normalization_stats.pkl" 2>/dev/null | wc -l || echo "0")
  echo "Normalization stats files: $stat_files"
  [ "$stat_files" -ge 2 ] || exit 42
  echo "DATA_VERIFIED"
  exit 0
REMOTE
  )
  echo "$post_check"
  
  if ! echo "$post_check" | grep -q "DATA_VERIFIED"; then
    echo "ERROR: Post-deployment verification failed. Data is still missing or incomplete."
    exit 1
  fi
  echo "Post-deployment verification successful."
fi

# ====================================================================
# 5. START TRAINING
# ====================================================================
echo "Step 5: Starting training on EC2 instance..."
training_start=$(ssh -i "$SSH_KEY" "$SSH_HOST" <<'REMOTE'
cd ~/emotion-recognition
shopt -s nullglob
launcher=(train_g5_*.sh)
if [ ${#launcher[@]} -eq 0 ]; then
  echo "No training launcher script found!"
  exit 1
fi
latest_launcher="${launcher[-1]}"
chmod +x "$latest_launcher"
echo "Starting training with $latest_launcher..."
nohup "./$latest_launcher" > logs/training_stdout.log 2>&1 &
echo $! > training.pid
pid=$(cat training.pid)
ps -p $pid -o command= || { echo "Failed to start training job"; exit 1; }
echo "Training job started with PID: $pid"
exit 0
REMOTE
)
echo "$training_start"

if echo "$training_start" | grep -q "Failed to start training job"; then
  echo "ERROR: Failed to start the training job on EC2 instance."
  exit 1
fi
echo "Training job started successfully on EC2 instance."

# ====================================================================
# 6. LIVE MONITORING (daemon)
# ====================================================================
echo "Step 6: Setting up monitoring daemon..."
monitor_setup=$(ssh -i "$SSH_KEY" "$SSH_HOST" <<'REMOTE'
cd ~/emotion-recognition
mkdir -p logs/monitoring
echo "Setting up monitoring daemon..."
nohup bash -c 'while true; do date >> logs/monitoring/monitor.log; find logs -name "train_laugh_*.log" -exec tail -n 10 {} \; >> logs/monitoring/monitor.log 2>&1; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv >> logs/monitoring/monitor.log 2>&1; echo "--------------------------" >> logs/monitoring/monitor.log; sleep 60; done' > /dev/null 2>&1 &
echo $! > monitor.pid
echo "Monitoring daemon started with PID: $(cat monitor.pid)"
exit 0
REMOTE
)
echo "$monitor_setup"
echo "Monitoring daemon setup complete."

# ====================================================================
# 7-8. TRAINING PROGRESS MONITOR (foreground)
# ====================================================================
echo "Step 7-8: Monitoring training progress..."
cat << EOF
Training is now running on the AWS instance.
- Local monitoring script created: agentic_monitor_${TIMESTAMP}.sh
- To download the trained model when complete: agentic_download_${TIMESTAMP}.sh
EOF

# Create local monitoring script
cat > agentic_monitor_${TIMESTAMP}.sh << EOF
#!/usr/bin/env bash
# Monitor script for training progress
ssh -i "$SSH_KEY" "$SSH_HOST" "tail -f ~/emotion-recognition/logs/monitoring/monitor.log"
EOF
chmod +x agentic_monitor_${TIMESTAMP}.sh

# Create download script
cat > agentic_download_${TIMESTAMP}.sh << EOF
#!/usr/bin/env bash
# Download artifacts after training completion

# Check if training is still running
if ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f train_audio_pooling_lstm_with_laughter.py > /dev/null"; then
  echo "Training is still running on EC2. Wait for completion or check with monitoring script."
  exit 1
fi

# Look for download script locally
cd "$LOCAL_REPO"
shopt -s nullglob
dl=(download_g5_model_*.sh)
if [ \${#dl[@]} -eq 0 ]; then
  echo "No download script found. Run deploy_to_aws_g5.sh first."
  exit 1
fi

# Execute download script
chmod +x "\${dl[-1]}"
./"\${dl[-1]}"

# Verify downloaded model
if ls models/audio_pooling_with_laughter_*/model_best.h5 &>/dev/null; then
  echo "Model downloaded successfully!"
  echo "To run local demo:"
  echo "python scripts/demo_emotion_with_laughter.py --model \$(ls -d models/audio_pooling_with_laughter_*)/model_best.h5"
else
  echo "Model download failed or incomplete."
  exit 1
fi

# Extract validation accuracy
echo "Running validation accuracy comparison..."
python extract_all_models_val_accuracy.py
python compare_model_accuracies.py --latest

echo "Download and validation complete."
EOF
chmod +x agentic_download_${TIMESTAMP}.sh

# ====================================================================
# COMPLETION SUMMARY
# ====================================================================
echo "=== AGENTIC WORKFLOW SETUP COMPLETE ==="
echo "What's running:"
echo "1. Training job on AWS g5.2xlarge ($AWS_IP)"
echo "2. Monitoring daemon on AWS (logs/monitoring/monitor.log)"
echo ""
echo "Available scripts:"
echo "- ./agentic_monitor_${TIMESTAMP}.sh  : View real-time training logs and GPU utilization"
echo "- ./agentic_download_${TIMESTAMP}.sh : Download and evaluate the trained model (after completion)"
echo ""
echo "Expected training duration: ~10 hours for 100 epochs"
echo "Log file: $LOG_FILE"
echo "==============================================="
