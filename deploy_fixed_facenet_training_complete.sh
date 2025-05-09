#!/bin/bash
# Deploy and run the fixed video-only Facenet LSTM training script
# This script addresses the core issue where features and labels were not properly aligned,
# and fixes a model saving issue with Keras

# Configuration
EC2_HOST="ubuntu@18.208.166.91"
REMOTE_PATH="/home/ubuntu/emotion-recognition"
LOCAL_TRAINING_SCRIPT="scripts/train_video_only_facenet_lstm_fixed.py"
LOCAL_GENERATOR_SCRIPT="scripts/fixed_video_facenet_generator.py"
LOCAL_FIX_SCRIPT="scripts/fix_keras_save_model.py"
REMOTE_TRAINING_SCRIPT="$REMOTE_PATH/scripts/train_video_only_facenet_lstm_fixed.py"
REMOTE_GENERATOR_SCRIPT="$REMOTE_PATH/scripts/fixed_video_facenet_generator.py"
REMOTE_FIX_SCRIPT="$REMOTE_PATH/scripts/fix_keras_save_model.py"
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

echo "=== Fixed Facenet Video-Only LSTM Training (Complete Version) ==="
echo "Training script: $LOCAL_TRAINING_SCRIPT"
echo "Generator script: $LOCAL_GENERATOR_SCRIPT"
echo "Fix script: $LOCAL_FIX_SCRIPT"

if [ "$LOCAL_RUN" = true ]; then
  echo "Mode: Local Execution"
  
  # Check if the data directories exist locally
  if [ ! -d "./ravdess_features_facenet" ] || [ ! -d "./crema_d_features_facenet" ]; then
    echo "WARNING: Local data directories not found. Ensure you have the Facenet features locally."
    echo "Continue anyway? (y/n)"
    read response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
      echo "Exiting."
      exit 1
    fi
  fi
  
  # Ensure scripts are executable
  chmod +x "$LOCAL_TRAINING_SCRIPT"
  chmod +x "$LOCAL_FIX_SCRIPT"
  
  # Apply the model save fix to the training script
  echo "Applying Keras model save fix..."
  python3 "$LOCAL_FIX_SCRIPT" "$LOCAL_TRAINING_SCRIPT"
  
  # Run the script locally
  echo "Starting local training..."
  python3 "$LOCAL_TRAINING_SCRIPT"
else
  echo "Mode: EC2 Deployment"
  echo "Target: $EC2_HOST:$REMOTE_PATH"
  
  # Test SSH connection first
  echo "Testing SSH connection..."
  if ! ssh -o ConnectTimeout=5 "$EC2_HOST" "echo Connection successful"; then
    echo "ERROR: Cannot connect to $EC2_HOST"
    echo "Check your SSH keys and connection settings."
    echo "You can run locally with --local instead."
    exit 1
  fi
  
  # Ensure remote directories exist
  echo "Ensuring remote directories exist..."
  ssh "$EC2_HOST" "mkdir -p $REMOTE_PATH/scripts"
  
  # Upload the scripts
  echo "Uploading training scripts..."
  scp "$LOCAL_TRAINING_SCRIPT" "$EC2_HOST:$REMOTE_TRAINING_SCRIPT"
  scp "$LOCAL_GENERATOR_SCRIPT" "$EC2_HOST:$REMOTE_GENERATOR_SCRIPT"
  scp "$LOCAL_FIX_SCRIPT" "$EC2_HOST:$REMOTE_FIX_SCRIPT"
  
  # Make scripts executable
  echo "Setting executable permissions..."
  ssh "$EC2_HOST" "chmod +x $REMOTE_TRAINING_SCRIPT $REMOTE_FIX_SCRIPT"
  
  # Apply the model save fix on the remote system
  echo "Applying Keras model save fix on remote system..."
  ssh "$EC2_HOST" "cd $REMOTE_PATH && python3 $REMOTE_FIX_SCRIPT $REMOTE_TRAINING_SCRIPT"
  
  # Check if remote data paths exist
  echo "Verifying data directories on remote host..."
  if ! ssh "$EC2_HOST" "[ -d /home/ubuntu/emotion-recognition/ravdess_features_facenet ] && [ -d /home/ubuntu/emotion-recognition/crema_d_features_facenet ]"; then
    echo "WARNING: Remote data directories not found. Ensure preprocessing has been completed."
    echo "Continue anyway? (y/n)"
    read response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
      echo "Exiting."
      exit 1
    fi
  fi
  
  # Launch the training script
  echo "Launching fixed Facenet training on $EC2_HOST..."
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="training_fixed_facenet_$TIMESTAMP.log"
  
  # Use nohup to keep the process running even if the connection drops
  ssh "$EC2_HOST" "cd $REMOTE_PATH && nohup python3 $REMOTE_TRAINING_SCRIPT > $LOG_FILE 2>&1 &"
  
  # Create a monitoring script
  MONITOR_SCRIPT="monitor_fixed_facenet_$TIMESTAMP.sh"
  cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Monitor script for fixed Facenet training
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
  ssh "\$EC2_HOST" "cd \$REMOTE_PATH && find models/facenet_lstm_fixed_* -type f -name 'best_model.h5' -exec ls -l {} \;"
  
  echo ""
  echo "--- Checking Validation Accuracy ---"
  ssh "\$EC2_HOST" "cd \$REMOTE_PATH && find models/facenet_lstm_fixed_* -type f -name 'metrics.txt' -exec cat {} \;"
  
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
