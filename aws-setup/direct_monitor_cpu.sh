#!/bin/bash
# Directly monitor system resources without menu

INSTANCE_IP="98.82.121.48"
# Pre-configured username - change this if needed
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Connecting to instance to monitor CPU usage..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "top -b -n 1"
