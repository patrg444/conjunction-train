#!/bin/bash
# Script to provide continuous monitoring with automatic reconnection for the LSTM attention model training
# Combines log monitoring, CPU monitoring, and connection status checks

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if instance is running
check_instance_status() {
  echo -e "${YELLOW}Checking instance status...${NC}"
  STATUS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  echo -e "${GREEN}Instance status: ${STATUS}${NC}"
  
  if [[ "$STATUS" != "running" ]]; then
    echo -e "${RED}Instance is not running (status: $STATUS)!${NC}"
    return 1
  fi
  return 0
}

# Test SSH connection
test_ssh_connection() {
  echo -e "${YELLOW}Testing SSH connection...${NC}"
  if ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP "echo Connection successful" &>/dev/null; then
    echo -e "${GREEN}SSH connection successful${NC}"
    return 0
  else
    echo -e "${RED}SSH connection failed!${NC}"
    return 1
  fi
}

# Monitor training log
monitor_log() {
  echo -e "${YELLOW}Starting log monitoring...${NC}"
  ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -f ~/emotion_training/$LOG_FILE" || return 1
}

# Monitor CPU
monitor_cpu() {
  echo -e "${YELLOW}Checking CPU usage...${NC}"
  ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "top -b -n 1 | head -20" || return 1
}

# Monitor memory
monitor_memory() {
  echo -e "${YELLOW}Checking memory usage...${NC}"
  ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "free -h" || return 1
}

# Monitor disk space
monitor_disk() {
  echo -e "${YELLOW}Checking disk usage...${NC}"
  ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "df -h" || return 1
}

# Check training process
check_training_process() {
  echo -e "${YELLOW}Checking if training process is running...${NC}"
  PROCESS_COUNT=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ps aux | grep train_branched_attention | grep -v grep | wc -l" || echo "0")
  
  if [[ "$PROCESS_COUNT" -gt "0" ]]; then
    echo -e "${GREEN}Training process is running (found $PROCESS_COUNT matching processes)${NC}"
    return 0
  else
    echo -e "${RED}Warning: No training process detected!${NC}"
    return 1
  fi
}

# Check for model outputs
check_model_outputs() {
  echo -e "${YELLOW}Checking for model checkpoints...${NC}"
  MODEL_COUNT=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training/models -type f | wc -l" || echo "0")
  
  echo -e "${GREEN}Found $MODEL_COUNT model files${NC}"
  if [[ "$MODEL_COUNT" -gt "0" ]]; then
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ls -la ~/emotion_training/models" || return 1
  fi
}

# Check last few lines of log
check_last_log_lines() {
  echo -e "${YELLOW}Last 10 lines of training log:${NC}"
  ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -n 10 ~/emotion_training/$LOG_FILE" || return 1
}

print_separator() {
  echo -e "\n${BLUE}==================================================================${NC}\n"
}

print_banner() {
  echo -e "${BLUE}==================================================================${NC}"
  echo -e "${BLUE}     CONTINUOUS LSTM ATTENTION MODEL TRAINING MONITOR            ${NC}"
  echo -e "${BLUE}==================================================================${NC}"
  echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
  echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
  echo -e "${YELLOW}Monitoring started at:${NC} $(date)"
  echo -e "${BLUE}==================================================================${NC}"
  echo ""
}

# Main monitoring loop with auto-reconnection
continuous_monitor() {
  local mode=$1
  local watch_interval=60  # seconds between full status checks
  local reconnect_attempts=0
  local max_reconnect_attempts=10
  
  print_banner
  
  while true; do
    # Check instance status
    if ! check_instance_status; then
      echo -e "${RED}Instance not running. Exiting monitor.${NC}"
      return 1
    fi
    
    # Try to establish connection
    if ! test_ssh_connection; then
      reconnect_attempts=$((reconnect_attempts + 1))
      if [[ $reconnect_attempts -gt $max_reconnect_attempts ]]; then
        echo -e "${RED}Failed to reconnect after $max_reconnect_attempts attempts. Exiting.${NC}"
        return 1
      fi
      echo -e "${YELLOW}Reconnection attempt $reconnect_attempts of $max_reconnect_attempts...${NC}"
      sleep 10
      continue
    fi
    
    # Connection successful, reset counter
    reconnect_attempts=0
    
    # Check if training is running
    check_training_process
    print_separator
    
    # Check resource usage
    monitor_cpu
    print_separator
    monitor_memory
    print_separator
    monitor_disk
    print_separator
    
    # Check model outputs
    check_model_outputs
    print_separator
    
    # Check log
    check_last_log_lines
    print_separator
    
    # If mode is "log", monitor log continuously for a while
    if [[ "$mode" == "log" ]]; then
      echo -e "${YELLOW}Monitoring log continuously for 2 minutes...${NC}"
      timeout 120 monitor_log
      print_separator
    fi
    
    echo -e "${CYAN}Waiting $watch_interval seconds for next status check... (Press Ctrl+C to exit)${NC}"
    sleep $watch_interval
    clear
    print_banner
  done
}

# Parse command line options
case "$1" in
  --log)
    continuous_monitor "log"
    ;;
  --quick)
    # One-time status report without continuous monitoring
    print_banner
    check_instance_status
    print_separator
    test_ssh_connection
    print_separator
    check_training_process
    print_separator
    monitor_cpu
    print_separator
    check_last_log_lines
    ;;
  *)
    continuous_monitor "status"
    ;;
esac
