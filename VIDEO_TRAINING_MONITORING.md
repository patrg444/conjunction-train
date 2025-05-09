# Video Training Monitoring Guide

This guide explains the different monitoring options available for tracking video emotion recognition training progress.

## Monitoring Options

We've implemented several monitoring approaches, each with different strengths:

### 1. Basic Monitor (`monitor_video_training.sh`)

The basic monitor refreshes every 10 seconds with simple output showing:
- GPU status
- Disk space usage
- Top training processes
- Latest training logs

**Usage:**
```bash
./monitor_video_training.sh
```

### 2. Continuous Monitor (`continuous_video_monitor.sh`)

A minimalist monitor that continuously shows only GPU status.

**Usage:**
```bash
./continuous_video_monitor.sh
```

### 3. Advanced Monitor (`advanced_video_monitor.sh`)

A visually enhanced monitor with color-coded output showing:
- Training progress visualized with a progress bar
- GPU statistics
- Validation results
- Recent warnings

**Usage:**
```bash
./advanced_video_monitor.sh
```

### 4. Background Streaming Monitor (`direct_stream_monitor.sh`)

This is the most robust solution that sets up continuous logging in the background. It creates:
- A continuous GPU monitoring daemon
- A continuous training output capture
- Log files that persist even after disconnection

**Usage:**
```bash
# Start the background monitor
./direct_stream_monitor.sh

# View training logs in real-time
tail -f /home/ubuntu/monitor_logs/video_training_stream.log

# View GPU statistics in real-time
tail -f /home/ubuntu/monitor_logs/gpu_stats.log

# Stop the monitoring daemon
pkill -f "bash.*direct_stream_monitor.sh"; tmux kill-session -t monitor_*
```

## Choosing the Right Monitoring Approach

1. **For Quick Status Checks**:
   - Use `monitor_video_training.sh` for a comprehensive snapshot
   - Use `advanced_video_monitor.sh` for a more visual representation

2. **For Remote Training Sessions**:
   - Use `direct_stream_monitor.sh` to set up persistent logging
   - Reconnect later and use `tail -f` to view the logs

3. **For GPU Utilization Focus**:
   - Use `continuous_video_monitor.sh` for minimal GPU monitoring

## Troubleshooting

If you encounter issues with the monitors:

1. Ensure the training is running in a tmux session named `video_training`
2. Check if tmux is installed (`sudo apt install tmux` if not)
3. Verify permissions on the monitor scripts (`chmod +x *.sh`)
4. For background monitor, ensure the log directory exists (`mkdir -p ~/monitor_logs`)

## Tips for Effective Monitoring

- Use SSH with the `-Y` flag for better terminal support when viewing colorized output
- Consider setting up automatic monitoring on training start
- For long-running training sessions, the background monitor is recommended
- Monitor both GPU stats and training logs to spot potential issues early
