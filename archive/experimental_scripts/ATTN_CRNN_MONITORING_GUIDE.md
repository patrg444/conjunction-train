# ATTN-CRNN Training Monitoring Guide

This guide explains how to use the monitoring tools for the ATTN-CRNN emotion recognition model training process.

## Available Monitoring Tools

We've created multiple monitoring tools with different capabilities:

### Quick Checks

- `monitor_attn_crnn_training.sh`: Basic snapshot of GPU status and recent logs
- `monitor_attn_crnn.sh`: Direct attachment to the tmux session for interactive viewing

### Advanced Monitoring Options

- `continuous_attn_crnn_monitor.sh`: Periodic checks with full diagnostics
- `stream_attn_crnn_monitor.sh`: **Continuous real-time streaming** of training output

## Choose the Right Monitoring Tool

### Continuous Monitor

The periodic monitoring tool provides comprehensive health checks:

1. **Process Status Checks**:
   - Verifies tmux session existence
   - Confirms Python training process is active
   - Shows training duration

2. **Training Progress Analysis**:
   - Shows latest epoch information
   - Displays validation loss and accuracy trends
   - Updates automatically at regular intervals

3. **Error Detection**:
   - Scans recent logs for errors or exceptions
   - Specifically checks for data loading issues
   - Reports immediately if problems are found

4. **Checkpoint Tracking**:
   - Monitors for model checkpoint file creation
   - Reports checkpoint file size and timestamp
   - Confirms when models are successfully saved

5. **Dataset Diagnostics**:
   - Counts WAV2VEC feature files
   - Helps diagnose data availability issues

### Command Options for Continuous Monitor

```bash
# Periodic checks with 3-minute intervals
./continuous_attn_crnn_monitor.sh

# Custom interval: Check every minute
./continuous_attn_crnn_monitor.sh -i 60

# Single check: Run once and exit
./continuous_attn_crnn_monitor.sh -1

# Check GPU only: Show nvidia-smi and exit
./continuous_attn_crnn_monitor.sh -s

# Help: Show usage information
./continuous_attn_crnn_monitor.sh -h
```

## Live Streaming Monitor

The streaming monitor provides a **real-time feed** of training output:

1. **Continuous Output**: 
   - Streams tmux session output in real time
   - Shows every line as it's generated
   - No delay between updates

2. **Filtering Options**:
   - Filter for epoch/training lines
   - Focus on validation metrics
   - View all raw output

3. **Color Highlighting**:
   - Epochs highlighted in cyan
   - Val_accuracy in green
   - Val_loss in magenta
   - Errors in red

### Command Options for Streaming Monitor

```bash
# Default: Stream filtered training output (epoch/validation)
./stream_attn_crnn_monitor.sh

# Stream all raw output unfiltered
./stream_attn_crnn_monitor.sh -a

# Focus only on validation metrics
./stream_attn_crnn_monitor.sh -v

# Check session health before streaming
./stream_attn_crnn_monitor.sh -c

# Help: Show usage information
./stream_attn_crnn_monitor.sh -h
```

## Recommended Workflows

### For Regular Monitoring with Diagnostics

1. Start training in a tmux session as described in EC2_WORKFLOW.md
2. Launch the continuous monitor with periodic checks:
   ```bash
   ./continuous_attn_crnn_monitor.sh
   ```
3. Leave it running for regular status updates
4. Press Ctrl+C when you want to stop monitoring

### For Continuous Real-Time Streaming

1. Start training in a tmux session as described in EC2_WORKFLOW.md
2. First check system health:
   ```bash
   ./stream_attn_crnn_monitor.sh -c
   ```
3. Or directly start real-time streaming:
   ```bash
   ./stream_attn_crnn_monitor.sh
   ```
4. Watch training progress in real time, with colored output
5. Press Ctrl+C when you want to stop streaming

## Troubleshooting Training Issues

If the monitor shows errors or problems:

1. Look for "ERROR" or "WARNING" indicators in red text
2. Check if the training process is still running
3. Examine if WAV2VEC feature files are being found
4. Verify if model checkpoints are being created

Common issues detected by the monitor:
- Missing .npz feature files
- Script errors or exceptions
- Training process termination
- Missing tmux session
