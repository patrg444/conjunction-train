#!/bin/bash
# Enhanced script for continuous monitoring of LSTM attention model training on AWS
# This script will maintain a persistent connection and provide real-time updates

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if connection file exists
if [ ! -f aws-setup/lstm_attention_model_connection.txt ]; then
    echo -e "${RED}Connection details not found. Please run deploy_lstm_attention_model.sh first.${NC}"
    exit 1
fi

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# Define monitoring modes
LOG_MODE="log"
GPU_MODE="gpu"
SYSTEM_MODE="system"
ALL_MODE="all"

# Default to log mode if no argument provided
MONITOR_MODE=${1:-$LOG_MODE}

print_banner() {
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BOLD}${BLUE}     CONTINUOUS MONITORING: LSTM ATTENTION MODEL TRAINING     ${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${YELLOW}Instance:${NC} $INSTANCE_IP"
    echo -e "${YELLOW}Mode:${NC} $MONITOR_MODE"
    echo -e "${YELLOW}Press Ctrl+C to exit monitoring${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo ""
}

monitor_logs() {
    echo -e "${CYAN}Establishing continuous log monitoring...${NC}"
    # Use a persistent SSH connection to stream logs with timestamp prefix
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -f ~/emotion_training/$LOG_FILE"
}

monitor_gpu() {
    echo -e "${MAGENTA}Establishing continuous GPU monitoring...${NC}"
    # Create a script on the remote machine to continuously monitor GPU
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/gpu_monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo \"=== GPU STATUS (Updated: \$(date)) ===\"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv
    echo \"\"
    echo \"=== GPU PROCESSES ===\"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep 5
done
EOF"

    # Make the script executable and run it
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/gpu_monitor.sh && ~/gpu_monitor.sh"
}

monitor_system() {
    echo -e "${GREEN}Establishing continuous system monitoring...${NC}"
    # Create a script on the remote machine to continuously monitor system resources
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/system_monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo \"=== SYSTEM STATUS (Updated: \$(date)) ===\"
    echo \"\"
    echo \"--- CPU USAGE ---\"
    top -b -n 1 | head -n 20
    echo \"\"
    echo \"--- MEMORY USAGE ---\"
    free -h
    echo \"\"
    echo \"--- DISK USAGE ---\"
    df -h | grep -v tmp
    echo \"\"
    echo \"--- ACTIVE PYTHON PROCESSES ---\"
    ps aux | grep python | grep -v grep
    sleep 10
done
EOF"

    # Make the script executable and run it
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/system_monitor.sh && ~/system_monitor.sh"
}

monitor_all() {
    echo -e "${BOLD}Establishing comprehensive monitoring (all metrics)...${NC}"
    # Create a specialized script for multi-window monitoring
    echo -e "${YELLOW}Creating comprehensive monitoring setup...${NC}"
    echo -e "${YELLOW}This will use tmux to display multiple monitoring views.${NC}"
    
    # Create the comprehensive monitoring script on the remote machine
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cat > ~/comprehensive_monitor.sh << 'EOF'
#!/bin/bash

# Check if already in a tmux session
if [ -n \"\$TMUX\" ]; then
    echo \"Already in a tmux session. Please exit current session first.\"
    exit 1
fi

# Kill any existing monitoring session
tmux kill-session -t monitoring 2>/dev/null

# Create monitoring scripts
cat > ~/gpu_monitor.sh << 'EOFGPU'
#!/bin/bash
while true; do
    clear
    echo \"=== GPU STATUS (Updated: \$(date)) ===\"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv
    echo \"\"
    echo \"=== GPU PROCESSES ===\"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep 5
done
EOFGPU

cat > ~/system_monitor.sh << 'EOFSYS'
#!/bin/bash
while true; do
    clear
    echo \"=== SYSTEM STATUS (Updated: \$(date)) ===\"
    echo \"\"
    echo \"--- CPU USAGE ---\"
    top -b -n 1 | head -n 20
    echo \"\"
    echo \"--- MEMORY USAGE ---\"
    free -h
    sleep 10
done
EOFSYS

chmod +x ~/gpu_monitor.sh ~/system_monitor.sh

# Create a new tmux session
tmux new-session -d -s monitoring

# Create windows and panes
tmux rename-window -t monitoring:0 'Monitoring Dashboard'
tmux split-window -h -t monitoring:0
tmux split-window -v -t monitoring:0.0
tmux split-window -v -t monitoring:0.1

# Run monitoring in each pane
tmux send-keys -t monitoring:0.0 'echo \"=== TRAINING LOG ===\" && tail -f ~/emotion_training/$LOG_FILE' C-m
tmux send-keys -t monitoring:0.1 'echo \"=== GPU METRICS ===\" && ~/gpu_monitor.sh' C-m
tmux send-keys -t monitoring:0.2 'echo \"=== SYSTEM RESOURCES ===\" && ~/system_monitor.sh' C-m
tmux send-keys -t monitoring:0.3 'echo \"=== MODEL DIRECTORY ===\" && watch -n 10 \"ls -la ~/emotion_training/models/attention_focal_loss/\"' C-m

# Attach to the session
tmux attach-session -t monitoring
EOF"

    # Make the script executable and run it
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "chmod +x ~/comprehensive_monitor.sh && ~/comprehensive_monitor.sh"
}

# Print the banner
print_banner

# Execute the appropriate monitoring based on mode
case $MONITOR_MODE in
    $LOG_MODE)
        monitor_logs
        ;;
    $GPU_MODE)
        monitor_gpu
        ;;
    $SYSTEM_MODE)
        monitor_system
        ;;
    $ALL_MODE)
        monitor_all
        ;;
    *)
        echo -e "${RED}Invalid monitoring mode: $MONITOR_MODE${NC}"
        echo -e "Valid modes are: $LOG_MODE, $GPU_MODE, $SYSTEM_MODE, $ALL_MODE"
        exit 1
        ;;
esac
