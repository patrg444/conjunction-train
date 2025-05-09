#!/bin/bash
set -e

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem

echo "=========================================="
echo "Cleaning up files on EC2 instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Get process IDs of rsync or any data transfer processes and kill them
echo "Stopping any ongoing file transfers..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF_REMOTE'
    # Find and kill rsync processes
    pids=$(pgrep -f rsync || echo "")
    if [ -n "$pids" ]; then
        echo "Killing rsync processes: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
    
    # Find and kill other file transfer processes if needed
    # Add more commands if necessary
EOF_REMOTE

# Cleaning up the emotion_project directory
echo "Removing emotion_project directory..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP << 'EOF_REMOTE'
    # Check if the directory exists
    if [ -d "$HOME/emotion_project" ]; then
        # Backup the best_attn_crnn_model.h5 if it exists
        if [ -f "$HOME/emotion_project/best_attn_crnn_model.h5" ]; then
            cp "$HOME/emotion_project/best_attn_crnn_model.h5" "$HOME/"
            echo "Backed up model file to home directory"
        fi
        
        # Remove the entire directory
        rm -rf "$HOME/emotion_project"
        echo "Removed emotion_project directory"
        
        # Create a clean minimal directory structure
        mkdir -p "$HOME/emotion_project/scripts"
        mkdir -p "$HOME/emotion_project/models"
        
        # Restore the model file if it was backed up
        if [ -f "$HOME/best_attn_crnn_model.h5" ]; then
            mv "$HOME/best_attn_crnn_model.h5" "$HOME/emotion_project/models/"
            echo "Restored model file to new directory"
        fi
        
        echo "Created clean directory structure"
    else
        echo "emotion_project directory does not exist"
    fi
    
    # Show disk usage after cleanup
    echo "Current disk usage:"
    df -h /
EOF_REMOTE

echo "=========================================="
echo "Cleanup completed!"
echo "=========================================="
