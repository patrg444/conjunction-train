#!/usr/bin/env python3
"""
This script fixes the missing comma in the WarmUpReduceLROnPlateau class
by directly modifying the file on the server.
"""

import os
import sys
import subprocess
import tempfile

# Server details
EC2_HOST = "ubuntu@54.162.134.77"
KEY_PATH = "~/Downloads/gpu-key.pem"
REMOTE_FILE = "/home/ubuntu/audio_emotion/fixed_v6_script_final.py"
LOCAL_TEMP = "temp_fixed_script.py"

def run_command(cmd):
    """Run a shell command and return its output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        return None
    return result.stdout

def main():
    # Download the file from the server
    print(f"Downloading the script from the server...")
    download_cmd = f"scp -i {KEY_PATH} {EC2_HOST}:{REMOTE_FILE} {LOCAL_TEMP}"
    run_command(download_cmd)
    
    if not os.path.exists(LOCAL_TEMP):
        print(f"Failed to download the file. Please check the connection and paths.")
        return
    
    # Read the file content
    with open(LOCAL_TEMP, 'r') as f:
        content = f.read()
    
    # Check if the comma is missing
    if 'set_value(self.model.optimizer.learning_rate warmup_lr)' in content:
        print("Found missing comma in the script.")
        
        # Replace the missing comma in two places
        content = content.replace(
            'set_value(self.model.optimizer.learning_rate warmup_lr)',
            'set_value(self.model.optimizer.learning_rate, warmup_lr)')
        
        content = content.replace(
            'set_value(self.model.optimizer.learning_rate new_lr)',
            'set_value(self.model.optimizer.learning_rate, new_lr)')
        
        # Write the fixed content back to the file
        with open(LOCAL_TEMP, 'w') as f:
            f.write(content)
        
        print("File fixed successfully! Uploading back to the server...")
        
        # Upload the fixed file back to the server
        upload_cmd = f"scp -i {KEY_PATH} {LOCAL_TEMP} {EC2_HOST}:{REMOTE_FILE}"
        run_command(upload_cmd)
        
        # Restart the training
        print("Restarting the training with the fixed script...")
        restart_cmd = f"ssh -i {KEY_PATH} {EC2_HOST} 'cd /home/ubuntu/audio_emotion && python3 {REMOTE_FILE} > wav2vec_fixed_comma_direct_fix_$(date +%Y%m%d_%H%M%S).log 2>&1 &'"
        run_command(restart_cmd)
        
        print("Fix completed and training restarted!")
    else:
        print("No comma issue found in the script. It may have been fixed already.")
    
    # Clean up the temporary file
    os.remove(LOCAL_TEMP)

if __name__ == "__main__":
    main()
