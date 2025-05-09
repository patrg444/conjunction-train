#!/bin/bash

# Set default values
CHECK_GPU=false
CHECK_PROCESS=false
VIEW_LOGS=false
CHECK_TMUX=false
COUNT_FILES=false
DOWNLOAD_MODEL=false

# Parse command line options
while getopts "gplmcd" opt; do
  case $opt in
    g) CHECK_GPU=true ;;
    p) CHECK_PROCESS=true ;;
    l) VIEW_LOGS=true ;;
    m) CHECK_TMUX=true ;;
    c) COUNT_FILES=true ;;
    d) DOWNLOAD_MODEL=true ;;
    *) echo "Usage: $0 [-g] [-p] [-l] [-m] [-c] [-d]"
       echo "  -g  Check GPU usage"
       echo "  -p  Check if training process is running"
       echo "  -l  View complete logs"
       echo "  -m  View tmux session output"
       echo "  -c  Count loaded files and show label distribution"
       echo "  -d  Download model when complete"
       exit 1 ;;
  esac
done

# EC2 instance details
EC2_USER="ubuntu"
EC2_IP=$(cat aws_instance_ip.txt 2>/dev/null || echo "54.162.134.77")
KEY_PATH="$HOME/Downloads/gpu-key.pem"
LOG_PATH="/home/ubuntu/emotion_project/train_combined_attn_crnn.log"
TMUX_SESSION="attn_crnn_training"

# If downloading model is requested
if $DOWNLOAD_MODEL; then
  echo "Preparing to download the latest model..."
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "cd ~/emotion_project && ls -t models/attn_crnn_wav2vec_*.h5 | head -1" | while read model; do
    if [ ! -z "$model" ]; then
      echo "Found model: $model"
      echo "Downloading model..."
      scp -i "$KEY_PATH" "$EC2_USER@$EC2_IP:~/emotion_project/$model" .
      echo "Model downloaded successfully."
    else
      echo "No model files found yet. Training may still be in progress."
    fi
  done
  exit 0
fi

# If GPU check is requested
if $CHECK_GPU; then
  echo "Checking GPU usage..."
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "nvidia-smi"
  exit 0
fi

# If process check is requested
if $CHECK_PROCESS; then
  echo "Checking process status:"
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "ps -ef | grep fixed_attn_crnn.py | grep -v grep"
  if [ $? -eq 0 ]; then
    echo "PROCESS IS RUNNING"
  else
    echo "PROCESS NOT RUNNING!"
  fi
  exit 0
fi

# If tmux output check is requested
if $CHECK_TMUX; then
  echo "Checking tmux session output:"
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "tmux capture-pane -pt $TMUX_SESSION -S -50"
  exit 0
fi

# If full log view is requested
if $VIEW_LOGS; then
  echo "Displaying complete logs:"
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "cat $LOG_PATH"
  exit 0
fi

# If file count is requested
if $COUNT_FILES; then
  echo "Count of loaded files and label distribution:"
  ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "cat $LOG_PATH | grep -A 5 \"Label distribution\""
  exit 0
fi

# Default behavior: show training status summary
echo "Checking if training process is running..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "ps -ef | grep fixed_attn_crnn.py | grep -v grep" > /dev/null
if [ $? -eq 0 ]; then
  echo "PROCESS IS RUNNING"
else
  echo "PROCESS NOT RUNNING!"
fi

echo -e "\nLatest log entries:"
echo "==============================================================="
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "tail -n 50 $LOG_PATH"

echo -e "\nCheck for data loading:"
echo "==============================================================="
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "grep -A 10 \"Loaded.*valid samples\" $LOG_PATH | tail -10"

echo -e "\nCheck for training progress (batches):"
echo "==============================================================="
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "grep -E \"(Epoch|loss:|val_loss:)\" $LOG_PATH | tail -20"

echo -e "\nCheck for any errors:"
echo "==============================================================="
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "grep -i error $LOG_PATH | tail -5"

echo -e "\nMonitor complete. Run this script again to see updated progress."
