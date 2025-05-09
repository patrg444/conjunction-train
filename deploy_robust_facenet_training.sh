#!/bin/bash
# Deploy and run the robust video-only Facenet LSTM training script
# This version handles both EC2 deployment and local execution

# Configuration
EC2_HOST="ubuntu@18.208.166.91"
REMOTE_PATH="/home/ubuntu/emotion-recognition"
LOCAL_SCRIPT="scripts/train_video_only_facenet_lstm_robust.py"
LOCAL_GENERATOR="scripts/video_only_facenet_generator.py"
REMOTE_SCRIPT="$REMOTE_PATH/scripts/train_video_only_facenet_lstm_robust.py"
LOCAL_RUN=false  # Set to true to run locally instead of on EC2

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_RUN=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --local    Run the training script locally instead of on EC2"
      echo "  --help     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "=== Robust Facenet LSTM Training Deployment ==="
echo "Script: $LOCAL_SCRIPT"

if [ "$LOCAL_RUN" = true ]; then
  echo "Mode: Local Execution"
  
  # Ensure the script is executable
  chmod +x "$LOCAL_SCRIPT"
  
  # Run the script locally
  echo "Starting local training..."
  python3 "$LOCAL_SCRIPT"
else
  echo "Mode: EC2 Deployment"
  echo "Target: $EC2_HOST:$REMOTE_PATH"
  
  # Test SSH connection first
  echo "Testing SSH connection..."
  if ! ssh -o ConnectTimeout=5 "$EC2_HOST" "echo Connection successful"; then
    echo "ERROR: Cannot connect to $EC2_HOST"
    echo "Check your SSH keys and connection settings."
    echo "You can try running locally with --local instead."
    exit 1
  fi
  
  # Ensure remote directory exists
  echo "Ensuring remote directories exist..."
  ssh "$EC2_HOST" "mkdir -p $REMOTE_PATH/scripts"
  
  # Upload the scripts
  echo "Uploading training scripts..."
  scp "$LOCAL_SCRIPT" "$EC2_HOST:$REMOTE_SCRIPT"
  scp "$LOCAL_GENERATOR" "$EC2_HOST:$REMOTE_PATH/scripts/"
  
  # Make scripts executable
  echo "Setting executable permissions..."
  ssh "$EC2_HOST" "chmod +x $REMOTE_SCRIPT"
  
  # Launch the training script
  echo "Launching robust Facenet training on $EC2_HOST..."
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="training_robust_facenet_$TIMESTAMP.log"
  
  # We'll use nohup to keep the process running after we disconnect
  ssh "$EC2_HOST" "cd $REMOTE_PATH && nohup python3 $REMOTE_SCRIPT > $LOG_FILE 2>&1 &"
  
  # Create a monitoring script
  MONITOR_SCRIPT="monitor_robust_facenet_$TIMESTAMP.sh"
  cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Monitor script for robust Facenet training
EC2_HOST="$EC2_HOST"
REMOTE_PATH="$REMOTE_PATH"
LOG_FILE="$LOG_FILE"

while true; do
  echo ""
  date
  echo "--- Latest Training Log ---"
  ssh "\$EC2_HOST" "cd \$REMOTE_PATH && tail -n 30 \$LOG_FILE"
  
  echo ""
  echo "--- Checking for Checkpoints ---"
  ssh "\$EC2_HOST" "cd \$REMOTE_PATH && find models/robust_video_only_facenet_lstm_* -type f -name 'best_model.keras' -exec ls -l {} \;"
  
  echo ""
  echo "Press Ctrl+C to exit monitoring..."
  sleep 60
done
EOF
  
  chmod +x "$MONITOR_SCRIPT"
  echo "Created monitoring script: ./$MONITOR_SCRIPT"
  
  echo "Training started in the background on $EC2_HOST"
  echo "You can monitor progress with: ./$MONITOR_SCRIPT"
  
  # Offer to start monitoring
  read -p "Start monitoring now? (y/n) " START_MONITOR
  if [[ $START_MONITOR == "y" || $START_MONITOR == "Y" ]]; then
    ./$MONITOR_SCRIPT
  else
    echo "You can start monitoring later with: ./$MONITOR_SCRIPT"
  fi
fi

echo "Deployment complete!"
