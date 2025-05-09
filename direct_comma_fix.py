#!/usr/bin/env python3
"""
This is a direct fix script that will SSH into the server, 
edit the file to add the missing commas, and restart the training.
"""

import subprocess
import tempfile
import os

# Server information
SERVER = "ubuntu@54.162.134.77"
KEY_PATH = "~/Downloads/gpu-key.pem"
SCRIPT_PATH = "/home/ubuntu/audio_emotion/fixed_v6_script_final.py"
TEMP_FILE = "temp_script.py"

def run_command(cmd, show_output=True):
    """Run a shell command and return its output"""
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if show_output:
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(f"ERROR: {process.stderr}")
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    
    return process.returncode == 0, process.stdout

def main():
    # Step 1: Download the script from the server
    print("\n=== Downloading the script from server ===")
    download_cmd = f"scp -i {KEY_PATH} {SERVER}:{SCRIPT_PATH} {TEMP_FILE}"
    success, _ = run_command(download_cmd)
    
    if not success:
        print("Failed to download the script. Aborting.")
        return
    
    # Step 2: Fix the missing commas in the file
    print("\n=== Fixing the missing commas ===")
    try:
        with open(TEMP_FILE, 'r') as f:
            content = f.read()
        
        # Make the replacements - we'll make them very specific to avoid changing anything else
        old_text1 = "tf.keras.backend.set_value(self.model.optimizer.learning_rate warmup_lr)"
        new_text1 = "tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)"
        
        old_text2 = "tf.keras.backend.set_value(self.model.optimizer.learning_rate new_lr)"
        new_text2 = "tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)"
        
        # Count occurrences before replacement
        count1 = content.count(old_text1)
        count2 = content.count(old_text2)
        
        # Make the replacements
        content = content.replace(old_text1, new_text1)
        content = content.replace(old_text2, new_text2)
        
        # Write the fixed content back to the file
        with open(TEMP_FILE, 'w') as f:
            f.write(content)
        
        print(f"Fixed {count1} occurrences of the first missing comma")
        print(f"Fixed {count2} occurrences of the second missing comma")
        
        if count1 == 0 and count2 == 0:
            print("No missing commas found in the file. It might have been fixed already or the line format doesn't match.")
    except Exception as e:
        print(f"Error while fixing the file: {str(e)}")
        return
    
    # Step 3: Upload the fixed file back to the server
    print("\n=== Uploading the fixed script to server ===")
    upload_cmd = f"scp -i {KEY_PATH} {TEMP_FILE} {SERVER}:{SCRIPT_PATH}"
    success, _ = run_command(upload_cmd)
    
    if not success:
        print("Failed to upload the fixed script. Aborting.")
        return
    
    # Step 4: Run the fixed script on the server
    print("\n=== Restarting the training with the fixed script ===")
    # Stop any existing processes first
    stop_cmd = f"ssh -i {KEY_PATH} {SERVER} 'pkill -f fixed_v6_script_final.py || true'"
    run_command(stop_cmd, show_output=False)
    
    # Run the fixed script
    timestamp = subprocess.check_output("date +%Y%m%d_%H%M%S", shell=True).decode().strip()
    log_file = f"/home/ubuntu/audio_emotion/wav2vec_fixed_comma_direct_fix_{timestamp}.log"
    run_cmd = f"ssh -i {KEY_PATH} {SERVER} 'cd /home/ubuntu/audio_emotion && nohup python {SCRIPT_PATH} > {log_file} 2>&1 &'"
    success, _ = run_command(run_cmd)
    
    if not success:
        print("Failed to restart the training. Please check the server manually.")
        return
    
    # Step 5: Clean up the local temporary file
    os.remove(TEMP_FILE)
    
    print("\n=== Fix completed ===")
    print(f"The script has been fixed and training has been restarted.")
    print(f"You can monitor the training progress with:")
    print(f"./monitor_direct_fix.sh")

if __name__ == "__main__":
    main()
