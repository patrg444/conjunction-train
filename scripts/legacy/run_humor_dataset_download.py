#!/usr/bin/env python3
"""
Master script to download and prepare humor classification datasets.
This script orchestrates the download and preparation of multiple humor detection datasets,
then creates suitable manifests for training DistilBERT models.
"""

import os
import subprocess
import argparse
import glob
import pandas as pd
from tqdm import tqdm

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("datasets/humor_datasets", exist_ok=True)
    os.makedirs("datasets/manifests/humor", exist_ok=True)

def download_all_datasets():
    """Download all humor datasets"""
    print("Step 1: Downloading datasets...")
    
    # Download FunLines dataset
    funlines_script_path = "download_funlines.py"
    if os.path.exists(funlines_script_path):
        print(f"\nDownloading FunLines dataset...")
        subprocess.run(["python", funlines_script_path], check=True)
    else:
        print(f"Warning: {funlines_script_path} not found. Skipping FunLines.")
    
    # Download Short Humor dataset
    short_humor_script_path = "download_short_humor.py" 
    if os.path.exists(short_humor_script_path):
        print(f"\nDownloading Short Humor dataset...")
        subprocess.run(["python", short_humor_script_path], check=True)
    else:
        print(f"Warning: {short_humor_script_path} not found. Skipping Short Humor.")

def create_combined_manifests():
    """Combine all humor datasets into unified manifests"""
    print("\nStep 2: Creating combined manifests...")
    
    # Define paths
    manifest_dir = "datasets/manifests/humor"
    combined_train_path = os.path.join(manifest_dir, "combined_train_humor.csv")
    combined_val_path = os.path.join(manifest_dir, "combined_val_humor.csv")
    
    # Find all train and val manifests
    train_manifests = glob.glob(os.path.join(manifest_dir, "*train*humor.csv"))
    val_manifests = glob.glob(os.path.join(manifest_dir, "*val*humor.csv"))
    
    # Remove any previous combined manifests from the lists
    train_manifests = [m for m in train_manifests if "combined" not in m]
    val_manifests = [m for m in val_manifests if "combined" not in m]
    
    print(f"Found {len(train_manifests)} training manifests and {len(val_manifests)} validation manifests.")
    
    # Combine training manifests
    combined_train_df = pd.DataFrame()
    for manifest in train_manifests:
        print(f"  Reading {os.path.basename(manifest)}...")
        try:
            df = pd.read_csv(manifest)
            # Add dataset source column based on filename
            source = os.path.basename(manifest).split('_')[0]
            df['source'] = source
            combined_train_df = pd.concat([combined_train_df, df], ignore_index=True)
            print(f"    Added {len(df)} samples.")
        except Exception as e:
            print(f"    Error reading {manifest}: {e}")
    
    # Combine validation manifests
    combined_val_df = pd.DataFrame()
    for manifest in val_manifests:
        print(f"  Reading {os.path.basename(manifest)}...")
        try:
            df = pd.read_csv(manifest)
            # Add dataset source column based on filename
            source = os.path.basename(manifest).split('_')[0]
            df['source'] = source
            combined_val_df = pd.concat([combined_val_df, df], ignore_index=True)
            print(f"    Added {len(df)} samples.")
        except Exception as e:
            print(f"    Error reading {manifest}: {e}")
    
    # Save combined manifests
    combined_train_df.to_csv(combined_train_path, index=False)
    combined_val_df.to_csv(combined_val_path, index=False)
    
    # Print statistics
    print("\nCombined dataset statistics:")
    print("Training set:")
    print(f"  Total samples: {len(combined_train_df)}")
    print(f"  Label 0 (non-humorous): {sum(combined_train_df['label'] == 0)}")
    print(f"  Label 1 (humorous): {sum(combined_train_df['label'] == 1)}")
    
    print("Validation set:")
    print(f"  Total samples: {len(combined_val_df)}")
    print(f"  Label 0 (non-humorous): {sum(combined_val_df['label'] == 0)}")
    print(f"  Label 1 (humorous): {sum(combined_val_df['label'] == 1)}")
    
    return combined_train_path, combined_val_path

def generate_train_script(train_manifest, val_manifest):
    """Generate a training script to use with the combined manifests"""
    print("\nStep 3: Creating training script...")
    
    train_script_path = "train_humor_classifier.sh"
    
    # Build the training command
    train_command = (
        "python enhanced_train_distil_humor.py "
        f"--train_manifest {train_manifest} "
        f"--val_manifest {val_manifest} "
        "--batch_size 32 "
        "--epochs 5 "
        "--learning_rate 5e-5 "
        "--model_name 'distilbert-base-uncased' "
        "--output_dir './checkpoints/humor_classifier/' "
        "--max_length 128"
    )
    
    # Create the training script
    with open(train_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to train a humor classifier on combined datasets\n\n")
        f.write(f"{train_command}\n")
    
    # Make it executable
    os.chmod(train_script_path, 0o755)
    
    print(f"Created training script: {train_script_path}")
    print("You can execute it with:")
    print(f"  ./{train_script_path}")
    
    return train_script_path

def push_to_ec2(train_manifest, val_manifest):
    """Generate an EC2 deployment script to train on AWS"""
    print("\nStep 4: Creating EC2 deployment script...")
    
    ec2_script_path = "train_humor_classifier_ec2.sh"
    
    with open(ec2_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to deploy humor classifier training to EC2\n\n")
        f.write("# Get the instance IP from file\n")
        f.write("EC2_IP=$(cat aws_instance_ip.txt)\n\n")
        
        f.write("# Copy manifest files to EC2\n")
        f.write(f"echo \"Copying manifest files to EC2...\"\n")
        f.write(f"ssh -i \"/Users/patrickgloria/Downloads/gpu-key.pem\" ubuntu@$EC2_IP \"mkdir -p ~/conjunction-train/datasets/manifests/humor/\"\n")
        f.write(f"scp -i \"/Users/patrickgloria/Downloads/gpu-key.pem\" {train_manifest} ubuntu@$EC2_IP:~/conjunction-train/{train_manifest}\n")
        f.write(f"scp -i \"/Users/patrickgloria/Downloads/gpu-key.pem\" {val_manifest} ubuntu@$EC2_IP:~/conjunction-train/{val_manifest}\n\n")
        
        f.write("# Run training on EC2\n")
        f.write("echo \"Starting training on EC2...\"\n")
        f.write("ssh -i \"/Users/patrickgloria/Downloads/gpu-key.pem\" ubuntu@$EC2_IP \"cd ~/conjunction-train && \\\n")
        f.write("  python enhanced_train_distil_humor.py \\\n")
        f.write(f"    --train_manifest {train_manifest} \\\n")
        f.write(f"    --val_manifest {val_manifest} \\\n")
        f.write("    --batch_size 64 \\\n")
        f.write("    --epochs 5 \\\n")
        f.write("    --learning_rate 5e-5 \\\n")
        f.write("    --model_name 'distilbert-base-uncased' \\\n")
        f.write("    --output_dir './checkpoints/humor_classifier/' \\\n")
        f.write("    --max_length 128\"\n")
    
    # Make it executable
    os.chmod(ec2_script_path, 0o755)
    
    print(f"Created EC2 deployment script: {ec2_script_path}")
    print("You can execute it with:")
    print(f"  ./{ec2_script_path}")
    
    return ec2_script_path

def main():
    """Main function to download and process humor datasets"""
    parser = argparse.ArgumentParser(description="Download and process humor datasets")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading datasets")
    parser.add_argument("--skip_combine", action="store_true", help="Skip combining manifests")
    args = parser.parse_args()
    
    setup_directories()
    
    if not args.skip_download:
        download_all_datasets()
    else:
        print("Skipping dataset download.")
    
    if not args.skip_combine:
        train_manifest, val_manifest = create_combined_manifests()
    else:
        print("Skipping manifest combination.")
        # Use default paths
        train_manifest = "datasets/manifests/humor/combined_train_humor.csv"
        val_manifest = "datasets/manifests/humor/combined_val_humor.csv"
    
    generate_train_script(train_manifest, val_manifest)
    push_to_ec2(train_manifest, val_manifest)
    
    print("\nWorkflow complete! You now have:")
    print("1. Downloaded humor datasets")
    print("2. Created combined manifests")
    print("3. Generated local training script")
    print("4. Generated EC2 training script")
    print("\nYou can now train a humor classifier using the generated scripts.")

if __name__ == "__main__":
    main()
