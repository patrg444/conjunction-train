#!/bin/bash

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_IP=$(cat aws_instance_ip.txt)

echo "Improving XLM-RoBERTa-large training configuration..."
echo "- Increasing max epochs to 15"
echo "- Configuring to only save checkpoints with improved validation F1 scores"

# Create a Python script to implement both improvements
cat << 'EOF' > /tmp/improve_training.py
import os
import sys
import subprocess
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

def create_custom_checkpoint_callback():
    """Create a custom checkpoint callback that only saves when F1 improves"""
    
    callback_dir = "/home/ubuntu/callbacks"
    os.makedirs(callback_dir, exist_ok=True)
    
    callback_code = """
import pytorch_lightning as pl

class BestF1ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = 0.0
        
    def _update_best_and_save(self, current, trainer, pl_module):
        # Only save if current val_f1 is better than the best so far
        if current > self.best_f1:
            self.best_f1 = current
            return super()._update_best_and_save(current, trainer, pl_module)
        return False
"""
    
    with open(os.path.join(callback_dir, "checkpoint_callback.py"), "w") as f:
        f.write(callback_code)
    
    print("Created custom checkpoint callback that only saves models with improved F1 scores.")
    return True

def restart_with_improved_settings():
    checkpoint_path = find_newest_checkpoint()
    if not checkpoint_path:
        print("No checkpoint found. Cannot restart training.")
        return False
    
    current_epochs = get_current_max_epochs()
    new_epochs = 15  # Set to fixed 15 epochs
    
    print(f"Found checkpoint: {checkpoint_path}")
    print(f"Current max_epochs: {current_epochs}")
    print(f"New max_epochs: {new_epochs}")
    
    if stop_current_process():
        # Create custom checkpoint callback
        create_custom_checkpoint_callback()
        
        # Add the callbacks directory to Python path
        os.environ["PYTHONPATH"] = "/home/ubuntu:" + os.environ.get("PYTHONPATH", "")
        
        # Start the training again with the new max_epochs value and custom checkpoint callback
        cmd = f"cd /home/ubuntu && nohup python -m scripts.train_xlm_roberta_large --resume_from_checkpoint {checkpoint_path} --max_epochs {new_epochs} --checkpoint_callback=callbacks.checkpoint_callback.BestF1ModelCheckpoint > /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log 2>&1 &"
        
        print(f"Restarting training with command: {cmd}")
        subprocess.run(cmd, shell=True)
        
        # Append the output of the improved training to the main log
        append_cmd = f"nohup bash -c 'tail -f /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log >> /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log' > /dev/null 2>&1 &"
        subprocess.run(append_cmd, shell=True)
        
        print("Training restarted with improved configuration.")
        return True
    else:
        print("No training process was running. Starting with the latest checkpoint.")
        # Create custom checkpoint callback
        create_custom_checkpoint_callback()
        
        # Add the callbacks directory to Python path
        os.environ["PYTHONPATH"] = "/home/ubuntu:" + os.environ.get("PYTHONPATH", "")
        
        # Start the training with the new max_epochs value and custom checkpoint callback
        cmd = f"cd /home/ubuntu && nohup python -m scripts.train_xlm_roberta_large --resume_from_checkpoint {checkpoint_path} --max_epochs {new_epochs} --checkpoint_callback=callbacks.checkpoint_callback.BestF1ModelCheckpoint > /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log 2>&1 &"
        
        print(f"Starting training with command: {cmd}")
        subprocess.run(cmd, shell=True)
        
        # Append the output of the improved training to the main log
        append_cmd = f"nohup bash -c 'tail -f /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log >> /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log' > /dev/null 2>&1 &"
        subprocess.run(append_cmd, shell=True)
        
        print("Training started with improved configuration.")
        return True

if __name__ == "__main__":
    print("Starting improved training process...")
    restart_with_improved_settings()
EOF

# Copy the Python script to the EC2 instance
scp -i "$SSH_KEY" /tmp/improve_training.py ubuntu@$EC2_IP:/home/ubuntu/improve_training.py

# Execute the script on the EC2 instance
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "python /home/ubuntu/improve_training.py"

echo ""
echo "XLM-RoBERTa training configuration improved:"
echo "- Max epochs set to 15"
echo "- Checkpoint saving modified to only save models with better validation F1 scores"
echo ""
echo "Monitor the continued training with:"
echo "./monitor_xlm_roberta_training.sh"
