#!/bin/bash

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_IP=$(cat aws_instance_ip.txt)

echo "Starting improved XLM-RoBERTa-large training with checkpoint saving optimization..."

# Create a Python script to restart training with improved settings
cat << 'EOF' > /tmp/start_improved_training.py
import os
import sys
import subprocess
import glob

def find_newest_checkpoint():
    checkpoint_dir = "/home/ubuntu/training_logs_humor/xlm-roberta-large/checkpoints"
    checkpoints = glob.glob(f"{checkpoint_dir}/*.ckpt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def create_checkpoint_script():
    # Create a script to modify the checkpoint callback
    script_content = """
import os
import sys
import glob
import pytorch_lightning as pl

# Custom ModelCheckpoint that saves only when validation F1 improves
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

# Path where the original script would look for the default callback
checkpoint_dir = os.path.join(os.getcwd(), 'callbacks')
os.makedirs(checkpoint_dir, exist_ok=True)

# Write the custom callback to a file that will be imported
with open(os.path.join(checkpoint_dir, 'checkpoint_callback.py'), 'w') as f:
    f.write('''
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
''')

print("Created custom checkpoint callback that only saves when validation F1 improves.")
"""
    
    # Write the script to a file
    with open('/home/ubuntu/create_custom_checkpoint.py', 'w') as f:
        f.write(script_content)
    
    print("Created script to implement custom checkpoint callback.")

def start_training():
    checkpoint_path = find_newest_checkpoint()
    if not checkpoint_path:
        print("No checkpoint found. Cannot restart training.")
        return False
    
    max_epochs = 15  # Set to 15 epochs
    
    print(f"Found checkpoint: {checkpoint_path}")
    print(f"Setting max_epochs to: {max_epochs}")
    
    # Create custom checkpoint callback
    create_checkpoint_script()
    
    # Start the training with the improved configuration
    cmd = f"cd /home/ubuntu && python create_custom_checkpoint.py && PYTHONPATH=/home/ubuntu:$PYTHONPATH nohup python -m scripts.train_xlm_roberta_large --resume_from_checkpoint {checkpoint_path} --max_epochs {max_epochs} --checkpoint_callback=callbacks.checkpoint_callback.BestF1ModelCheckpoint > /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log 2>&1 &"
    
    print(f"Starting training with command: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # Append the output of the improved training to the main log
    append_cmd = f"nohup bash -c 'tail -f /home/ubuntu/training_logs_humor/xlm-roberta-large/training_improved.log >> /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log' > /dev/null 2>&1 &"
    subprocess.run(append_cmd, shell=True)
    
    print("Training started with improved configuration.")
    return True

if __name__ == "__main__":
    print("Starting improved training process...")
    start_training()
EOF

# Copy the Python script to the EC2 instance
scp -i "$SSH_KEY" /tmp/start_improved_training.py ubuntu@$EC2_IP:/home/ubuntu/start_improved_training.py

# Execute the script on the EC2 instance
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "python /home/ubuntu/start_improved_training.py"

echo "Improved training configuration started."
echo "- Max epochs set to 15"
echo "- Checkpoint saving modified to only save models with better validation F1 scores"
echo ""
echo "Monitor the training with:"
echo "./monitor_xlm_roberta_training.sh"
