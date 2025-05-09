#!/bin/bash
# Script to monitor the CNN audio feature extraction progress

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

clear
echo "======================================"
echo "  CNN FEATURE EXTRACTION MONITOR"
echo "======================================"
echo "Monitoring extraction progress on G5 instance..."
echo ""

# Check SSH connection
if ! ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" "echo" &>/dev/null; then
    echo "❌ SSH connection failed. Check your SSH key and connection."
    exit 1
fi

# Check if extraction process is running
echo "[1/4] Checking extraction processes..."
EXTRACTION_PROCESSES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ps aux | grep 'preprocess_cnn_audio_features.py' | grep -v grep | wc -l")
echo "CNN feature extraction processes running: $EXTRACTION_PROCESSES"

if [ "$EXTRACTION_PROCESSES" -eq 0 ]; then
    echo "⚠️ No extraction processes running! The process may have completed or failed."
else
    echo "✅ Extraction is active. CPU is being used since GPU had initialization errors."
    
    # Show CPU usage of the extraction process
    echo ""
    echo "Process details:"
    ssh -i "$SSH_KEY" "$SSH_HOST" "ps aux | grep -v grep | grep 'preprocess_cnn_audio_features'"
fi

# Check directory sizes and file counts
echo ""
echo "[2/4] Checking extracted feature files..."
ssh -i "$SSH_KEY" "$SSH_HOST" "
    echo 'Directory sizes:'
    du -sh ~/emotion-recognition/data/ravdess_features_cnn_audio/ ~/emotion-recognition/data/crema_d_features_cnn_audio/
    
    echo ''
    echo 'File counts:'
    RAVDESS_COUNT=\$(find ~/emotion-recognition/data/ravdess_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l)
    CREMAD_COUNT=\$(find ~/emotion-recognition/data/crema_d_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l)
    TOTAL_COUNT=\$((RAVDESS_COUNT + CREMAD_COUNT))
    TOTAL_FILES=8882  # Expected total (1440 RAVDESS + 7442 CREMA-D)
    
    echo \"RAVDESS CNN features: \$RAVDESS_COUNT / 1440\"
    echo \"CREMA-D CNN features: \$CREMAD_COUNT / 7442\"
    echo \"Total progress: \$TOTAL_COUNT / \$TOTAL_FILES files (\$(echo \"scale=1; \$TOTAL_COUNT*100/\$TOTAL_FILES\" | bc)%)\"
"

# Check latest log entries
echo ""
echo "[3/4] Latest log entries..."
ssh -i "$SSH_KEY" "$SSH_HOST" "
    LOG_FILE=\$(ls -t ~/emotion-recognition/logs/preprocess_cnn_audio_*.log 2>/dev/null | head -1)
    if [ -n \"\$LOG_FILE\" ]; then
        echo \"Log file: \$LOG_FILE\"
        echo \"Recent entries:\"
        tail -10 \$LOG_FILE
    else
        echo \"No log file found.\"
    fi
"

# Estimate completion time
echo ""
echo "[4/4] Estimated completion..."
ssh -i "$SSH_KEY" "$SSH_HOST" "
    TOTAL_FILES=8882
    RAVDESS_COUNT=\$(find ~/emotion-recognition/data/ravdess_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l)
    CREMAD_COUNT=\$(find ~/emotion-recognition/data/crema_d_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l)
    TOTAL_COUNT=\$((RAVDESS_COUNT + CREMAD_COUNT))
    
    if [ \$TOTAL_COUNT -eq 0 ]; then
        echo \"Cannot estimate completion time yet - no files processed.\"
    else
        # Get process start time
        PROCESS_ID=\$(ps -ef | grep preprocess_cnn_audio_features.py | grep -v grep | head -1 | awk '{print \$2}')
        if [ -n \"\$PROCESS_ID\" ]; then
            START_TIME=\$(ps -p \$PROCESS_ID -o lstart= | xargs -0 date +%s -d)
            CURRENT_TIME=\$(date +%s)
            ELAPSED_SECONDS=\$((CURRENT_TIME - START_TIME))
            
            if [ \$ELAPSED_SECONDS -gt 0 ] && [ \$TOTAL_COUNT -gt 0 ]; then
                # Calculate rate and remaining time
                RATE=\$(echo \"scale=2; \$TOTAL_COUNT / \$ELAPSED_SECONDS\" | bc)
                REMAINING_FILES=\$((TOTAL_FILES - TOTAL_COUNT))
                REMAINING_SECONDS=\$(echo \"scale=0; \$REMAINING_FILES / \$RATE\" | bc)
                
                # Convert to hours:minutes:seconds
                ELAPSED_HMS=\$(date -u -d @\$ELAPSED_SECONDS +\"%H:%M:%S\")
                REMAINING_HMS=\$(date -u -d @\$REMAINING_SECONDS +\"%H:%M:%S\")
                COMPLETION_TIME=\$(date -d \"+\$REMAINING_SECONDS seconds\")
                
                echo \"Processing rate: \$RATE files per second\"
                echo \"Elapsed time: \$ELAPSED_HMS\"
                echo \"Estimated remaining time: \$REMAINING_HMS\"
                echo \"Estimated completion: \$COMPLETION_TIME\"
            else
                echo \"Not enough data to estimate completion time yet.\"
            fi
        else
            echo \"Cannot determine process start time.\"
        fi
    fi
"

echo ""
echo "======================================"
echo "Run this script periodically to check the extraction progress."
echo "When extraction completes, restart training with:"
echo "./run_audio_pooling_with_laughter.sh"
echo "======================================"
