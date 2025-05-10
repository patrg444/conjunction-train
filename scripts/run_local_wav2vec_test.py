#!/usr/bin/env python3
# Local test script for wav2vec emotion mapping
# This creates a modified copy of fixed_v6_script_final.py for local testing

import os
import sys
import tempfile
import shutil
import subprocess

# Create a modified version of the script for local testing
def create_local_test_script():
    input_script = "fixed_v6_script_final.py"
    output_script = "local_test_wav2vec.py"
    
    # Check if the input script exists
    if not os.path.exists(input_script):
        print(f"Error: {input_script} not found")
        sys.exit(1)
    
    # Read the original script
    with open(input_script, 'r') as f:
        script_content = f.read()
    
    # Path to the local directory containing wav2vec features
    local_data_dir = "./data/wav2vec"
    
    # Replace the paths and parameters for local testing
    replacements = [
        ('"/home/ubuntu/audio_emotion"', '"."'),
        ('epochs=100', 'epochs=2'),
        ('batch_size = 32', 'batch_size = 4'),
        ('os.path.join(data_dir, "models/wav2vec")', f'"{local_data_dir}"'),
        ('sample_size = min(500, len(valid_files))', 'sample_size = min(5, len(valid_files))'),
        ('test_size=0.1', 'test_size=0.2'),  # Larger test size for small dataset
    ]
    
    # Apply the replacements
    for old, new in replacements:
        script_content = script_content.replace(old, new)
    
    # Write the modified script
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(output_script, 0o755)
    
    print(f"Created local test script: {output_script}")
    return output_script

def main():
    # Create data directory if needed
    os.makedirs("./data/wav2vec", exist_ok=True)
    
    # Check for wav2vec data
    data_dir = "./data/wav2vec"
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')] if os.path.exists(data_dir) else []
    
    if not files:
        print("Warning: No .npz files found in ./data/wav2vec")
        print("Please provide the path to your wav2vec features:")
        external_data_path = input("Path to directory with wav2vec .npz files: ")
        
        if os.path.exists(external_data_path):
            print(f"Copying a few files from {external_data_path} to {data_dir} for testing...")
            # Copy a small subset of files for testing
            input_files = [f for f in os.listdir(external_data_path) if f.endswith('.npz')]
            # Prefer files with 'neutral' and 'calm' if available
            neutral_calm_files = [f for f in input_files if 'neutral' in f.lower() or 'calm' in f.lower()]
            # Take a mix of files, prioritizing neutral/calm but including others
            files_to_copy = neutral_calm_files[:5]
            # Add some other emotion files if needed
            other_files = [f for f in input_files if f not in neutral_calm_files]
            files_to_copy.extend(other_files[:10-len(files_to_copy)])
            
            for file in files_to_copy:
                src = os.path.join(external_data_path, file)
                dst = os.path.join(data_dir, file)
                shutil.copy2(src, dst)
            
            print(f"Copied {len(files_to_copy)} files for testing")
        else:
            print(f"Error: Path {external_data_path} does not exist")
            sys.exit(1)
    
    # Create the local test script
    test_script = create_local_test_script()
    
    # Run the test script
    print("\nRunning local test with a few epochs...")
    result = subprocess.run(['python', test_script], capture_output=True, text=True)
    
    # Display the results
    print("\n==== TEST OUTPUT ====")
    
    # Focus on the critical parts of output we want to verify
    output_lines = result.stdout.split('\n')
    
    # Print emotion mapping
    emotion_mapping_lines = []
    in_mapping_section = False
    for line in output_lines:
        if "Using emotion mapping:" in line:
            in_mapping_section = True
            emotion_mapping_lines.append(line)
        elif in_mapping_section and "->" in line:
            emotion_mapping_lines.append(line)
        elif in_mapping_section and line.strip() == "":
            in_mapping_section = False
    
    print("\nEmotion Mapping:")
    for line in emotion_mapping_lines:
        print(line)
    
    # Print unique labels before encoding
    unique_labels_line = [line for line in output_lines if "Original unique label values:" in line]
    if unique_labels_line:
        print("\n" + unique_labels_line[0])
    
    # Print a few lines of epoch results if training started
    epoch_lines = [line for line in output_lines if "Epoch " in line and "/" in line]
    if epoch_lines:
        print("\nTraining Progress:")
        for line in epoch_lines[:5]:  # Just the first few epoch lines
            print(line)
    
    # Check for errors
    if result.returncode != 0 or "Error:" in result.stdout or "error:" in result.stdout:
        print("\nWARNING: Test encountered errors. Check the following error messages:")
        for line in output_lines:
            if "Error:" in line or "error:" in line or "Exception" in line:
                print(line)
        print("\nFull error output:")
        print(result.stderr)
    else:
        print("\nSUCCESS: Local test completed without obvious errors")
        print("- Ensure 'neutral' and 'calm' are both mapped to index 0")
        print("- Verify training started and completed at least one epoch")
    
    print("\nIf local test is successful, you can now deploy the fixed script to EC2 with:")
    print("./deploy_fixed_existing_script.sh")

if __name__ == "__main__":
    main()
