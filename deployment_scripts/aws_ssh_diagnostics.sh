#!/usr/bin/env bash
# AWS SSH Diagnostics Tool
# This script diagnoses SSH connectivity issues with AWS EC2 instances

set -e

# Target instance details
AWS_IP="52.90.38.179"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"

# SSH key options to try
SSH_KEY_OPTIONS=(
  "$HOME/.ssh/id_rsa"
  "$HOME/.ssh/id_ed25519"
  "$HOME/.ssh/aws-key.pem"
  "$HOME/.ssh/aws.pem"
  "$HOME/Downloads/aws-key.pem"
  "$HOME/Downloads/aws_key.pem"
)

echo "=== AWS SSH DIAGNOSTICS ==="
echo "Starting SSH diagnostics at $(date)"
echo "Target: $SSH_HOST"
echo ""

# Check SSH connectivity
echo "1. Testing basic connectivity to port 22..."
if nc -zv -w 5 $AWS_IP 22 2>/dev/null; then
  echo "✅ Port 22 is open and accessible"
else
  echo "❌ Cannot connect to port 22. Check security groups and network ACLs."
  echo "Try command: nc -zv -w 5 $AWS_IP 22"
fi
echo ""

# Check SSH keys
echo "2. Checking for available SSH keys..."
echo "Found SSH keys:"
for key in "${SSH_KEY_OPTIONS[@]}"; do
  if [[ -f "$key" ]]; then
    echo "  ✅ $key exists"
    ls -la "$key" 2>/dev/null || echo "    (cannot access permissions)"
    ssh-keygen -l -f "$key" 2>/dev/null || echo "    (not a valid SSH key or is encrypted)"
  else
    echo "  ❌ $key not found"
  fi
done
echo ""

# Other SSH keys in ~/.ssh
echo "3. Other potential SSH keys in ~/.ssh directory:"
ls -la ~/.ssh/ 2>/dev/null | grep -E '\.(pem|key|pub)$' || echo "No additional keys found"
echo ""

# Try connecting with verbose output
echo "4. Attempting SSH connection with extended diagnostics..."
for key in "${SSH_KEY_OPTIONS[@]}"; do
  if [[ -f "$key" ]]; then
    echo "Trying with key: $key"
    ssh -v -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$key" $SSH_HOST "echo SUCCESS" 2>&1 | grep -E '(debug1|debug2|debug3|WARNING|ERROR|Permission denied|SUCCESS)'
    echo ""
  fi
done

# Check AWS CLI availability and credentials
echo "5. Checking AWS CLI installation and configuration..."
if command -v aws &>/dev/null; then
  echo "✅ AWS CLI is installed"
  aws --version
  
  echo "Checking AWS credentials..."
  if aws sts get-caller-identity 2>/dev/null; then
    echo "✅ AWS credentials are configured"
  else
    echo "❌ AWS credentials not found or invalid"
  fi
else
  echo "❌ AWS CLI not installed"
fi
echo ""

# Create Instructions
echo "=== TROUBLESHOOTING INSTRUCTIONS ==="
echo "1. Verify the EC2 instance is running: aws ec2 describe-instances --instance-ids i-03d733fa99127153a"
echo "2. Check that your local SSH key matches the key pair used when launching the EC2 instance"
echo "3. SSH keys need restrictive permissions: chmod 600 /path/to/key.pem"
echo "4. Try manual connect: ssh -v -i /path/to/key.pem $SSH_HOST"
echo "5. Check security groups for the instance to ensure port 22 is open"
echo ""

echo "=== DIAGNOSTICS COMPLETE ==="
echo "Run time: $(date)"
