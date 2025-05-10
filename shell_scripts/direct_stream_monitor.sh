#!/bin/bash

# Direct Stream Monitor - Creates a continuous log of training output
# This script sets up continuous logging on the server that can be monitored through SSH

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configure log file location
LOG_DIR="/home/ubuntu/monitor_logs"
LOG_FILE="${LOG_DIR}/video_training_stream.log"
STATS_FILE="${LOG_DIR}/gpu_stats.log"

# Make sure the log directory exists
mkdir -p ${LOG_DIR}

# Clear existing logs
> ${LOG_FILE}
> ${STATS_FILE}

echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${BOLD}${GREEN}          CONTINUOUS STREAMING TRAINING MONITOR            ${NC}"
echo -e "${BOLD}${GREEN}===========================================================${NC}"
echo -e "${CYAN}Setting up continuous monitoring on server...${NC}\n"

# Start GPU monitoring in the background
echo "Starting GPU monitoring daemon..."
(
    while true; do
        echo "===== $(date) =====" >> ${STATS_FILE}
        echo "GPU Stats:" >> ${STATS_FILE}
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> ${STATS_FILE}
        echo "" >> ${STATS_FILE}
        echo "Process Stats:" >> ${STATS_FILE}
        ps -o pid,%cpu,%mem,cmd -p $(pgrep -f train_video_full.py) | grep -v grep >> ${STATS_FILE}
        echo -e "\n" >> ${STATS_FILE}
        sleep 5
    done
) &
GPU_MONITOR_PID=$!

# Create a timestamp for tmux session name
SESSION_NAME="monitor_$(date +%s)"

# Create a new tmux session that continuously captures training output
tmux new-session -d -s ${SESSION_NAME}
tmux send-keys -t ${SESSION_NAME} "tmux capture-pane -pt video_training -S - > ${LOG_FILE}; tail -f -n 0 ${LOG_FILE}" C-m

echo -e "${GREEN}Monitoring daemons started successfully!${NC}"
echo -e "${YELLOW}To view training progress continuously:${NC}"
echo -e "${BOLD}   tail -f ${LOG_FILE}${NC}"
echo -e "${YELLOW}To view GPU statistics continuously:${NC}"
echo -e "${BOLD}   tail -f ${STATS_FILE}${NC}"
echo -e "\n${CYAN}These monitors will keep running in the background.${NC}"
echo -e "${CYAN}You can now disconnect from SSH and reconnect later to view logs.${NC}"
echo -e "${RED}To stop the monitoring daemons, run:${NC}"
echo -e "${BOLD}   pkill -f \"bash.*direct_stream_monitor.sh\"; tmux kill-session -t ${SESSION_NAME}${NC}"
