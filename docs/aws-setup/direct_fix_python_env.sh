#!/bin/bash
# Script to directly fix Python environment issues on EC2 instance

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Existing instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DIRECT FIX FOR PYTHON ENVIRONMENT ON EC2 INSTANCE           ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Create direct fix script to run on the instance
echo -e "${YELLOW}Creating direct fix script...${NC}"
cat > aws-setup/direct_fix.py << 'EOF'
#!/usr/bin/env python3
"""
Script to directly fix Python environment and install required packages
"""
import sys
import os
import subprocess
import time

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0

def check_import(module):
    """Check if a module can be imported"""
    try:
        exec(f"import {module}")
        print(f"✓ Successfully imported {module}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module}: {e}")
        return False

# Print Python info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Ensure pip is up to date
run_command(f"{sys.executable} -m pip install --upgrade pip")

# List of required packages
required_packages = [
    "numpy",
    "tensorflow",
    "pandas", 
    "scikit-learn",
    "matplotlib",
    "h5py"
]

# Install required packages using the current Python
for package in required_packages:
    print(f"\nInstalling {package}...")
    run_command(f"{sys.executable} -m pip install {package}")
    
    # Verify installation
    check_import(package)

# Create a simple test script to verify TensorFlow
test_script = """
import numpy as np
import tensorflow as tf

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Create a small test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
print("Test model compiled successfully")

# Create some dummy data
x = np.random.random((10, 5))
y = np.random.randint(0, 2, (10, 1))

# Train for just one step
model.fit(x, y, epochs=1)
print("Test complete!")
"""

# Write test script
with open("test_tensorflow.py", "w") as f:
    f.write(test_script)

# Run the test script
print("\nRunning TensorFlow test...")
success = run_command(f"{sys.executable} test_tensorflow.py")

if success:
    print("\n✓ TensorFlow environment is correctly set up!")
else:
    print("\n✗ TensorFlow environment setup failed")

# Create a wrapper script for running the LSTM attention model
wrapper_script = """#!/bin/bash
cd ~/emotion_training
export PYTHONPATH=$PYTHONPATH:~/emotion_training
echo "Starting LSTM attention model training at $(date)"
python3 scripts/train_branched_attention.py 2>&1 | tee training_lstm_attention_model.log
echo "Training completed with exit code $? at $(date)"
"""

# Write wrapper script
with open("run_lstm_attention.sh", "w") as f:
    f.write(wrapper_script)

# Make it executable
run_command("chmod +x run_lstm_attention.sh")

print("\nEnvironment fix completed. Ready to restart LSTM attention model training.")
EOF

# Upload the script to the EC2 instance
echo -e "${YELLOW}Uploading fix script to EC2 instance...${NC}"
scp -i $KEY_FILE aws-setup/direct_fix.py ec2-user@$INSTANCE_IP:~/direct_fix.py

# Execute the script on the EC2 instance
echo -e "${YELLOW}Running fix script on EC2 instance...${NC}"
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP "chmod +x ~/direct_fix.py && python3 ~/direct_fix.py"

# Restart the LSTM attention model training
echo -e "${YELLOW}Restarting LSTM attention model training...${NC}"
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP << EOF
    # Kill any existing sessions
    pkill -f train_branched_attention.py || true
    tmux kill-session -t lstm_attention 2>/dev/null || true
    
    # Start in tmux
    cd ~/emotion_training
    tmux new-session -d -s lstm_attention "./run_lstm_attention.sh"
    
    echo "LSTM attention model training restarted in tmux session."
    
    # Check if it's running
    sleep 5
    if pgrep -f train_branched_attention.py > /dev/null; then
        echo "Confirmed: Training process is now running."
    else
        echo "Warning: Training process may not have started successfully."
        
        # Check log for errors
        if [ -f training_lstm_attention_model.log ]; then
            echo "Last 10 lines of log:"
            tail -10 training_lstm_attention_model.log
        fi
    fi
EOF

echo -e "${GREEN}Python environment fixed and LSTM attention model training restarted.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      MONITORING OPTIONS${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "bash aws-setup/stream_logs.sh"
echo ""
echo -e "${YELLOW}To check CPU usage:${NC}"
echo -e "bash aws-setup/monitor_cpu_usage.sh"
echo ""
echo -e "${YELLOW}To check training files:${NC}"
echo -e "bash aws-setup/check_training_files.sh"
echo -e "${BLUE}=================================================================${NC}"
