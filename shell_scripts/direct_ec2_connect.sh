#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem

echo "===========================================" 
echo "Connecting directly to EC2 GPU instance..." 
echo "===========================================" 
echo "EC2 instance: $IP" 
echo "PEM key: $PEM" 
echo "===========================================" 

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Connect directly to the instance
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP
