#!/bin/bash

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_IP=$(cat aws_instance_ip.txt)

echo "Increasing max epochs for XLM-RoBERTa-large training..."

# Create a Python script to update the max_epochs parameter
cat << 'EOF' > /tmp/increase_epochs.py
import os
import sys
import signal
import subprocess
import json
import glob
import time

def find_newest_checkpoint():
    checkpoint_dir = "/home/ubuntu/training_logs_humor/xlm-roberta-large/checkpoints"
    checkpoints = glob.glob(f"{checkpoint_dir}/*.ckpt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def get_current_max_epochs():
    # Try to determine current max_epochs from process command line
    try:
        cmd = "ps aux | grep python | grep train_xlm_roberta_large.py | grep -v grep"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.stdout:
            cmd_line = process.stdout
            if "--max_epochs" in cmd_line:
                idx = cmd_line.find("--max_epochs")
                parts = cmd_line[idx:].split()
                if len(parts) > 1:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error determining current max_epochs: {e}")
    
    # Default if we can't determine
    return 5

def stop_current_process():
    try:
        cmd = "ps aux | grep python | grep train_xlm_roberta_large.py | grep -v grep | awk '{print $2}'"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.stdout.strip():
            pid = process.stdout.strip()
            print(f"Stopping current training process (PID: {pid})...")
            subprocess.run(f"kill {pid}", shell=True)
            # Give it a moment to clean up
            time.sleep(5)
            return True
        else:
            print("No running training process found.")
            return False
    except Exception as e:
        print(f"Error stopping process: {e}")
        return False

def restart_with_increased_epochs():
    checkpoint_path = find_newest_checkpoint()
    if not checkpoint_path:
        print("No checkpoint found. Cannot restart training.")
        return False
    
    current_epochs = get_current_max_epochs()
    new_epochs = current_epochs + 10  # Increase by 10 epochs
    
    print(f"Found checkpoint: {checkpoint_path}")
    print(f"Current max_epochs: {current_epochs}")
    print(f"New max_epochs: {new_epochs}")
    
    if stop_current_process():
        # Start the training again with the new max_epochs value
        cmd = f"cd /home/ubuntu && nohup python -m scripts.train_xlm_roberta_large --resume_from_checkpoint {checkpoint_path} --max_epochs {new_epochs} > /home/ubuntu/training_logs_humor/xlm-roberta-large/training_continued.log 2>&1 &"
        
        print(f"Restarting training with command: {cmd}")
        subprocess.run(cmd, shell=True)
        
        # Append the output of the continued training to the main log
        append_cmd = f"nohup bash -c 'tail -f /home/ubuntu/training_logs_humor/xlm-roberta-large/training_continued.log >> /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log' > /dev/null 2>&1 &"
        subprocess.run(append_cmd, shell=True)
        
        print("Training restarted with increased max_epochs.")
        return True
    else:
        print("Failed to stop current process or no process was running.")
        return False

if __name__ == "__main__":
    print("Starting epoch increase process...")
    restart_with_increased_epochs()
EOF

# Copy the Python script to the EC2 instance
scp -i "$SSH_KEY" /tmp/increase_epochs.py ubuntu@$EC2_IP:/home/ubuntu/increase_epochs.py

# Execute the script on the EC2 instance
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "python /home/ubuntu/increase_epochs.py"

echo "Max epochs increase process completed."
echo "Monitor the continued training with:"
echo "./monitor_xlm_roberta_training.sh"
