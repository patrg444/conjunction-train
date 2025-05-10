# Emotion Recognition Model: AWS Training Master Guide

This guide brings together all the components we've prepared for training your emotion recognition model on AWS GPU instances.

## Setup Overview

We've prepared a complete AWS training setup that includes:

1. **Data and code package**: 
   - All required Python scripts
   - Feature data archives (RAVDESS and CREMA-D)
   - Existing model checkpoints

2. **AWS deployment scripts**:
   - Environment setup scripts
   - Training scripts with GPU optimization
   - Multi-GPU support for high-performance instances

3. **Documentation and guides**:
   - Step-by-step instructions for AWS setup
   - Detailed performance configuration options

## Files and Components

| Component | Description | Location |
|-----------|-------------|----------|
| Deployment archive | Complete package (2.8GB) | `aws-emotion-recognition-training.tar.gz` |
| Deployment script | Interactive script to generate deployment scripts | `deploy-to-aws.sh` |
| EC2 Launch Guide | Step-by-step guide to launch an AWS instance | `EC2_LAUNCH_GUIDE.md` |
| Setup instructions | Quick reference instructions | `aws-temp/SETUP_INSTRUCTIONS.txt` |

## Complete Deployment Workflow

1. **Launch an AWS EC2 Instance**
   - Follow `EC2_LAUNCH_GUIDE.md` to launch a suitable instance
   - Recommended: g4dn.xlarge for balanced cost/performance (~$0.52/hour)
   - For maximum performance: p3.8xlarge with 4 V100 GPUs (~$12.24/hour)

2. **Generate Deployment Scripts**
   - Run `./deploy-to-aws.sh` from the aws-setup directory
   - Provide your AWS key file (.pem) path, instance IP, and training parameters
   - This will create a deployment directory with all necessary scripts

3. **Execute the Deployment**
   - Navigate to the generated deployment directory
   - Run `./run_all_steps.sh` to execute all deployment steps automatically
   - Or run individual scripts in sequence for more control:
     - `./01_upload_to_aws.sh` - Uploads the deployment archive
     - `./02_setup_on_aws.sh` - Sets up the AWS environment
     - `./03_run_training.sh` - Starts the training process

4. **Retrieve Results**
   - After training completes, run `./04_download_results.sh` to download the model

## Expected Performance

Based on the instance type selected, you can expect these approximate training speeds:

- Local machine: ~5 minutes per epoch
- g4dn.xlarge: ~30-60 seconds per epoch (5-10x speedup)
- p3.8xlarge: ~5-10 seconds per epoch (30x speedup)

A full 50-epoch training run should complete in:
- Local machine: ~4-5 hours
- g4dn.xlarge: ~25-50 minutes
- p3.8xlarge: ~5-10 minutes

## Important Reminders

1. Remember to stop or terminate your AWS instance when training is complete to avoid unnecessary charges.
2. For security purposes, it's recommended to limit SSH access to your IP address only.
3. Download your training results before terminating the instance.
