# AWS EC2 Instance Launch Guide

Follow these step-by-step instructions to launch the appropriate EC2 instance for your emotion recognition model training.

## 1. Login to AWS Console

1. Go to [AWS Management Console](https://aws.amazon.com/console/)
2. Sign in with your account credentials (ID: 3240-3729-1814)

## 2. Navigate to EC2 Service

1. Click on "Services" in the top navigation bar
2. Select "EC2" under Compute services or use the search bar

## 3. Launch a New Instance

1. Click the "Launch instance" button
2. Provide a name for your instance (e.g., "Emotion-Recognition-Training")

## 4. Select an AMI (Amazon Machine Image)

1. Search for "Deep Learning AMI"
2. Select "Deep Learning AMI (Amazon Linux 2) Version xx.x (xxx)" with CUDA 11.x support
   - This AMI comes pre-installed with all necessary deep learning frameworks and CUDA drivers

## 5. Choose an Instance Type

For your high-RAM approved account, select one of these recommended options:

1. **g4dn.xlarge** (~$0.52/hour):
   - 4 vCPUs, 16 GB RAM, 1 NVIDIA T4 GPU (16 GB VRAM)
   - Good balance of performance and cost

2. **r5.2xlarge** with GPU (~$0.50/hour):
   - 8 vCPUs, 64 GB RAM
   - Great for high memory requirements

3. **p3.8xlarge** (~$12.24/hour):
   - 32 vCPUs, 244 GB RAM, 4 NVIDIA V100 GPUs (64 GB VRAM)
   - Maximum performance for fastest training

## 6. Configure Key Pair

1. Under "Key pair (login)", click "Create new key pair"
2. Name your key pair (e.g., "emotion-recognition-key")
3. Select RSA for Key pair type and .pem for Private key file format
4. Click "Create key pair" and save the .pem file to a secure location
   - This file will be required to access your instance

## 7. Configure Network Settings

1. Under "Network settings", ensure:
   - VPC: Default VPC is selected
   - Auto-assign public IP: Enable
   - Security group: Create a new security group
   - Allow SSH traffic from: Your IP (or Anywhere if your IP changes)

## 8. Configure Storage

1. Under "Configure storage", set:
   - Root volume: gp2, 100 GB (or more if needed)
   - This needs to be large enough for your datasets and model checkpoints

## 9. Launch the Instance

1. Review your settings
2. Click "Launch instance"
3. Wait for the instance to initialize (usually takes 2-5 minutes)

## 10. Connect to Your Instance

1. In the EC2 Dashboard, select your instance
2. Copy the "Public IPv4 address" or "Public IPv4 DNS"
3. Use this address with the scripts in the `aws-setup` directory

## 11. After Training Completion

1. Return to EC2 Dashboard
2. Select your instance
3. Click "Instance state" > "Stop" or "Terminate"
   - Use "Stop" if you plan to use it again later
   - Use "Terminate" if you're completely done with it
   - **IMPORTANT**: AWS charges continue to accrue while your instance is running!

---

After completing these steps, return to the `deploy-to-aws.sh` script and provide the requested information when prompted.
