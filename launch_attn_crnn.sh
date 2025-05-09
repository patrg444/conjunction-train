#!/bin/bash

# Kill any existing attn_crnn tmux sessions
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "tmux kill-session -t attn_crnn 2>/dev/null || true"

# Launch the ATTN-CRNN training in a tmux session
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "cd ~/emotion_project && tmux new-session -d -s attn_crnn 'python3 scripts/fixed_attn_crnn.py --data_dirs /home/ubuntu/audio_emotion/models/wav2vec --epochs 30 --batch_size 32 --patience 5'"

echo "ATTN-CRNN training launched in tmux session. Use monitor_attn_crnn.sh to check progress."
