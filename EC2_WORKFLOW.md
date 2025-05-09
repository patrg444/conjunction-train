# EC2 Workflow for Attention CRNN Training

This document outlines the improved workflow for training the Attention CRNN model directly on the EC2 instance, minimizing file transfers and overhead.

## Setup Overview

We've transitioned to a more efficient workflow by:

1. Cleaning up unnecessary files on the EC2 instance
2. Only transferring essential training scripts
3. Working directly on the instance rather than deploying massive amounts of data
4. Using simple, targeted monitoring commands

## Available Scripts

### Connection and File Transfer

- `direct_ec2_connect.sh` - Direct SSH connection to the EC2 instance
- `transfer_attn_crnn_script.sh` - Transfer only the essential training script

### Monitoring and Management

- `monitor_attn_crnn.sh` - Multi-purpose monitoring script with the following options:
  - No options: Directly connect to the EC2 instance
  - `-l`: View training logs
  - `-s`: Check server/GPU status
  - `-d`: Download the trained model
  - `-h`: Show help information
- `continuous_attn_crnn_monitor.sh` - Real-time continuous monitoring script:
  - No options: Monitor continuously with 3-minute intervals
  - `-i SECONDS`: Set custom check interval
  - `-1`: Run once and exit
  - `-s`: Show GPU status only and exit
  - `-h`: Show help information

## Workflow Steps

1. **Prepare**: Clean up unnecessary files on the EC2 instance (already done)
   ```
   ./cleanup_ec2_files.sh
   ```

2. **Transfer**: Send only the needed training script
   ```
   ./transfer_attn_crnn_script.sh
   ```

3. **Connect and Setup**: Connect to the EC2 instance and set up the environment
   ```
   ./direct_ec2_connect.sh
   ```
   
   Then on the EC2 instance, make sure dependencies are installed:
   ```
   source ~/miniconda3/bin/activate
   conda activate emotion_env || conda create -n emotion_env python=3.9 -y && conda activate emotion_env
   pip install tensorflow==2.15.0 tensorflow-addons seaborn tqdm matplotlib numpy pandas scikit-learn
   ```

4. **Train**: Start the training process in a tmux session
   ```
   cd ~/emotion_project
   mkdir -p models
   tmux new -s train
   CUDA_VISIBLE_DEVICES=0 python scripts/train_attn_crnn.py --data_dirs . --augment
   # Use Ctrl+b, d to detach from tmux
   ```

5. **Monitor**: Choose monitoring approach based on your needs
   ```bash
   # Quick manual checks
   ./monitor_attn_crnn.sh -s  # Check GPU status
   ./monitor_attn_crnn.sh -l  # View logs
   
   # Option 1: Periodic diagnostic monitoring (every 3 min)
   ./continuous_attn_crnn_monitor.sh       # Regular health checks
   ./continuous_attn_crnn_monitor.sh -i 60 # More frequent checks (every minute)
   ./continuous_attn_crnn_monitor.sh -1    # Run a single diagnostic check
   
   # Option 2: Continuous real-time streaming
   ./stream_attn_crnn_monitor.sh           # Stream training output in real-time
   ./stream_attn_crnn_monitor.sh -c        # Check health then stream
   ./stream_attn_crnn_monitor.sh -v        # Focus on validation metrics only
   ```
   
   **Monitoring Features:**
   - Continuous Monitor: Provides comprehensive health checks, error detection, model file tracking
   - Streaming Monitor: Shows colored real-time output with no delay, best for watching training progress
   
   See `ATTN_CRNN_MONITORING_GUIDE.md` for complete details on both monitoring tools.

6. **Download**: Once training completes, download the model
   ```
   ./monitor_attn_crnn.sh -d
   ```

## Benefits of This Approach

- **Efficiency**: Minimal data transfer and reduced disk usage
- **Direct Control**: Working directly on the instance for more immediate feedback
- **Simplicity**: Clean, focused scripts that do exactly what's needed
- **Better Resource Usage**: Avoids filling up EC2 storage with unnecessary files
