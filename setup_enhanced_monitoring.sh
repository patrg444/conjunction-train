#!/usr/bin/env bash
# Setup enhanced monitoring and callbacks for G5 training
# This script uploads the necessary files to set up TensorBoard and improve monitoring

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================"
echo "     SETTING UP ENHANCED MONITORING   "
echo "======================================"

# Create remote script that will set up TensorBoard and monitoring improvements
cat > /tmp/remote_setup_monitoring.sh << 'EOF'
#!/usr/bin/env bash
set -e

# Set up directories
cd ~/emotion-recognition
mkdir -p logs/tensorboard
mkdir -p datasets_raw/manifests

# Install TensorBoard if not already installed
pip install --quiet tensorboard

# Check if TensorBoard is already running
TB_RUNNING=$(pgrep -f tensorboard || echo 'not_running')
if [[ "$TB_RUNNING" == "not_running" ]]; then
    echo "Starting TensorBoard server..."
    nohup tensorboard --logdir=logs/tensorboard --port=6006 --host=localhost > ~/tensorboard.log 2>&1 &
    echo "TensorBoard started."
else
    echo "TensorBoard is already running."
fi

# Create a callback modification script
cat > add_tensorboard_callback.py << 'PYTHONEOF'
#!/usr/bin/env python3
"""
Modify the currently running training script to add TensorBoard callback.
This script adds TensorBoard callback by patching the train_audio_pooling_lstm_fixed.py file.
"""
import os
import sys
import fileinput
import re
from datetime import datetime

# Get the training script path
script_path = "scripts/train_audio_pooling_lstm_fixed.py"

if not os.path.exists(script_path):
    print(f"Error: Training script not found at {script_path}")
    sys.exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/tensorboard/run_{timestamp}"

# Check if TensorBoard callback is already present
with open(script_path, 'r') as f:
    content = f.read()
    if "TensorBoard" in content:
        print("TensorBoard callback already exists in the script.")
        sys.exit(0)

# Find the setup_callbacks function and add TensorBoard callback
callback_pattern = r"def setup_callbacks\(model_dir\):"
tensorboard_code = """
def setup_callbacks(model_dir):
    # Create tensorboard callback
    from tensorflow.keras.callbacks import TensorBoard
    import os
    log_dir = os.path.join('logs/tensorboard', 'run_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

    # Previous callbacks
"""

# Also add ModelCheckpoint for every 5 epochs
checkpoint_pattern = r"checkpoint = ModelCheckpoint\("
checkpoint_addition = """    # Save checkpoint every 5 epochs
    regular_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'model_epoch_{epoch:03d}.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
        save_freq=5*steps_per_epoch
    )

    # Best checkpoint (existing)
    checkpoint = ModelCheckpoint("""

# Add the callbacks to the list
callbacks_return_pattern = r"return \[checkpoint, early_stopping, reduce_lr, cos_annealing\]"
callbacks_return_replacement = "return [checkpoint, regular_checkpoint, early_stopping, reduce_lr, cos_annealing, tensorboard_callback]"

# Import datetime if not present
import_datetime_pattern = r"import time"
import_datetime_replacement = "import time\nfrom datetime import datetime"

# Collect modifications
modifications = [
    (callback_pattern, tensorboard_code),
    (checkpoint_pattern, checkpoint_addition),
    (callbacks_return_pattern, callbacks_return_replacement),
    (import_datetime_pattern, import_datetime_replacement)
]

# Apply modifications
modified = False
for pattern, replacement in modifications:
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        modified = True

if modified:
    # Write the modified content back to the file
    with open(script_path, 'w') as f:
        f.write(content)
    print(f"Added TensorBoard and regular checkpoint callbacks to {script_path}")
else:
    print("Could not find the expected patterns in the script. No modifications made.")

# Now try to fix the variable reference for steps_per_epoch
if "steps_per_epoch" in content and "NameError: name 'steps_per_epoch' is not defined" not in content:
    steps_pattern = r"steps_per_epoch = .*?\n"
    if not re.search(steps_pattern, content):
        # Add steps_per_epoch calculation before callbacks setup
        steps_code = """
    # Calculate steps per epoch for checkpointing
    steps_per_epoch = len(train_gen)
"""
        function_pattern = r"def train_model\(.*?\):"
        content = re.sub(function_pattern, lambda m: m.group(0) + steps_code, content)
        with open(script_path, 'w') as f:
            f.write(content)
        print("Added steps_per_epoch calculation for the checkpointing logic")

print("Script modifications completed.")
PYTHONEOF

# Make the script executable
chmod +x add_tensorboard_callback.py

# Check for laughter manifest
if [[ ! -f datasets_raw/manifests/laughter_v1.csv ]]; then
    echo "Creating placeholder laughter manifest..."
    mkdir -p datasets_raw/manifests
    # Create a basic placeholder until the real one is available
    echo "filename,laughter" > datasets_raw/manifests/laughter_v1.csv
    echo "sample1.wav,0" >> datasets_raw/manifests/laughter_v1.csv
    echo "sample2.wav,1" >> datasets_raw/manifests/laughter_v1.csv
    echo "Created placeholder laughter manifest at datasets_raw/manifests/laughter_v1.csv"
fi

# Copy normalization files to the expected location
mkdir -p models/dynamic_padding_no_leakage
if [[ -f models/audio_normalization_stats.pkl ]]; then
    cp models/audio_normalization_stats.pkl models/dynamic_padding_no_leakage/
    echo "Copied audio normalization stats to expected location."
fi
if [[ -f models/video_normalization_stats.pkl ]]; then
    cp models/video_normalization_stats.pkl models/dynamic_padding_no_leakage/
    echo "Copied video normalization stats to expected location."
fi

# Run the script to add TensorBoard callback
echo "Adding TensorBoard callback to training script..."
python add_tensorboard_callback.py

echo "Setup complete!"
echo "To monitor with TensorBoard, run:"
echo "ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@18.208.166.91"
echo "Then open http://localhost:6006 in your browser"
EOF

# Upload the setup script
echo "Uploading setup script to EC2 instance..."
scp -i "$SSH_KEY" /tmp/remote_setup_monitoring.sh "$SSH_HOST":~/setup_enhanced_monitoring.sh

# Make the script executable and run it
echo "Making script executable and running it..."
ssh -i "$SSH_KEY" "$SSH_HOST" "chmod +x ~/setup_enhanced_monitoring.sh && ~/setup_enhanced_monitoring.sh"

echo "Enhanced monitoring setup has been configured on the EC2 instance."
echo "You can now monitor the training with:"
echo "1. ./enhanced_monitor_g5.sh - For overall status and log monitoring"
echo "2. For TensorBoard visualization:"
echo "   ssh -i $SSH_KEY -L 6006:localhost:6006 $SSH_HOST"
echo "   Then open http://localhost:6006 in your browser"
