# AWS Training Setup for Emotion Recognition Model

This guide walks you through the process of training your emotion recognition model on AWS GPU instances for significantly faster performance.

## Available Setup Scripts

We provide several scripts for setting up AWS training:

1. **`aws_gpu_fallback.sh`** (Recommended) - Tries multiple GPU instance types in order, automatically falling back to alternatives if high-performance options aren't available
   
2. **`aws_cli_setup_nopager.sh`** - Standard setup script for a single instance type without pagination issues
   
3. **`deploy-to-aws.sh`** - Interactive setup with generation of deployment scripts

## Prerequisites

- AWS account with access to EC2 services
- AWS CLI installed and configured on your local machine
- SSH key pair for EC2 access
- Basic familiarity with Linux commands

## Step 1: Prepare Your Files

Run the `prepare-data.sh` script to package your code and data:

```bash
chmod +x prepare-data.sh
./prepare-data.sh
```

This will create a directory `aws-temp` containing:
- Required Python scripts
- Compressed archives of your feature datasets
- Any existing models (if available)

## Step 2: Launch an AWS EC2 Instance

1. **Log in to the AWS Console** and navigate to EC2

2. **Launch a new instance** with these recommended specifications:
   - **AMI**: Deep Learning AMI (Amazon Linux 2) with CUDA 11.x
   - **Instance Type** options:
     - Standard: g4dn.xlarge (~$0.52/hour) or p3.2xlarge (~$3.06/hour)
     - **High RAM** (recommended): r5.2xlarge (8 vCPUs, 64GB RAM, ~$0.50/hour) with GPU
     - For maximum performance: p3.8xlarge (4 V100 GPUs, 32 vCPUs, 244GB RAM, ~$12.24/hour)
   - **Storage**: 100GB (ensure enough space for datasets and model checkpoints)
   - **Security Group**: Allow SSH access (port 22)
   - **Key Pair**: Select your existing key pair or create a new one
   - **AWS Account ID**: Use account ID 3240-3729-1814 (approved for high RAM instances)

3. **Wait for the instance to launch** and note its public IP address

## Step 3: Transfer Files to EC2

Using SCP, transfer your prepared files:

```bash
cd aws-temp
scp -i /path/to/your-key.pem * ec2-user@your-instance-ip:~/
```

## Step 4: Set Up the EC2 Environment

SSH into your instance:

```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-ip
```

Make the setup script executable and run it:

```bash
chmod +x aws-instance-setup.sh
./aws-instance-setup.sh
```

This script will:
- Install necessary dependencies
- Extract your feature datasets
- Set up the required directory structure

## Step 5: Run Training

Make the training script executable:

```bash
chmod +x train-on-aws.sh
```

Start training with default parameters:

```bash
./train-on-aws.sh
```

Or customize with options:

```bash
./train-on-aws.sh --learning-rate 2e-5 --epochs 100 --batch-size 64
```

Optional: Use S3 for result storage:

```bash
./train-on-aws.sh --use-s3 your-s3-bucket-name
```

## Step 6: Monitor Training and Retrieve Results

Training logs are saved to a timestamped file and output to the console in real-time.

To retrieve trained models after completion:

```bash
# From your local machine
scp -i /path/to/your-key.pem -r ec2-user@your-instance-ip:~/models/branched_6class ./
```

## Cost Considerations

- g4dn.xlarge: ~$0.52/hour
- p3.2xlarge: ~$3.06/hour
- r5.2xlarge (high RAM): ~$0.50/hour
- p3.8xlarge (high performance): ~$12.24/hour
- Training time estimate: 
  - Standard instances: 3-5 hours for 50 epochs
  - High-performance instances: <1 hour for 50 epochs
- **Important**: Remember to stop or terminate your instance when not in use!

## Troubleshooting

- If you encounter CUDA/GPU issues, verify the instance has GPU drivers by running `nvidia-smi`
- For memory errors, reduce batch size or consider a larger instance
- For network-related transfer issues, consider using AWS S3 as an intermediate storage

## Performance Comparison

Training on AWS GPU instances vs. local CPU:
- Local CPU: ~5 minutes per epoch (estimated)
- AWS g4dn.xlarge: ~30-60 seconds per epoch (estimated)
- AWS p3.2xlarge: ~15-30 seconds per epoch (estimated)
- AWS r5.2xlarge with GPU: ~20-40 seconds per epoch (estimated)
- AWS p3.8xlarge: ~5-10 seconds per epoch (estimated)

This translates to approximately 8-20x speedup depending on the instance type.
