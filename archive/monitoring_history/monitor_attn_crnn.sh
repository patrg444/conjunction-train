#!/bin/bash

# Connect to the running tmux session to monitor progress
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 -t "tmux attach-session -t attn_crnn"
