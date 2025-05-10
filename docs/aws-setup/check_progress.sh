#!/bin/bash
# Script to check training progress on the EC2 instance

INSTANCE_IP="98.82.121.48"

echo "Connecting to instance to check training progress..."
echo "Please enter your username for the EC2 instance (default: ec2-user):"
read -p "Username: " USERNAME
USERNAME=${USERNAME:-ec2-user}

ssh -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "tail -f ~/training.log"
