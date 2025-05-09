#!/bin/bash
# Script to check the status of the LSTM attention model deployment on AWS
# This is particularly useful when the deployment is taking longer than expected

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}      CHECKING LSTM ATTENTION MODEL DEPLOYMENT STATUS           ${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Check if the deployment process is still running
DEPLOY_PID=$(ps aux | grep "deploy_lstm_attention_model.sh" | grep -v grep | awk '{print $2}')
if [ -n "$DEPLOY_PID" ]; then
    echo -e "${GREEN}Deployment process is still running with PID: $DEPLOY_PID${NC}"
else
    echo -e "${RED}Deployment process is not running!${NC}"
    echo -e "${YELLOW}Check terminal output for any errors.${NC}"
fi

# Check if the connection file has been created
if [ -f aws-setup/lstm_attention_model_connection.txt ]; then
    echo -e "${GREEN}Connection file has been created. Deployment reached final stages.${NC}"
    echo -e "${CYAN}Connection details:${NC}"
    cat aws-setup/lstm_attention_model_connection.txt
else
    echo -e "${YELLOW}Connection file not yet created.${NC}"
    echo -e "${YELLOW}The deployment is likely still in progress at one of these stages:${NC}"
    echo -e "  - Launching EC2 instance"
    echo -e "  - Waiting for instance to initialize"
    echo -e "  - Establishing SSH connection"
    echo -e "  - Copying project files"
    echo -e "  - Setting up environment and dependencies"
    echo -e "  - Starting training"
fi

# Check for SSH keys
SSH_KEYS=$(ls -la aws-setup/emotion-recognition-key-*.pem 2>/dev/null)
if [ -n "$SSH_KEYS" ]; then
    echo -e "${GREEN}SSH keys have been created:${NC}"
    ls -la aws-setup/emotion-recognition-key-*.pem
else
    echo -e "${RED}No SSH keys found!${NC}"
fi

# Check for monitoring scripts
MONITOR_SCRIPTS=$(ls -la aws-setup/monitor_*attention*.sh 2>/dev/null)
if [ -n "$MONITOR_SCRIPTS" ]; then
    echo -e "${GREEN}Monitoring scripts have been created:${NC}"
    ls -la aws-setup/monitor_*attention*.sh
else
    echo -e "${YELLOW}No monitoring scripts created yet.${NC}"
fi

# Check for terminal output to determine current stage
echo -e "${BLUE}=================================================================${NC}"
echo -e "${CYAN}To see the current deployment stage, check the terminal where${NC}"
echo -e "${CYAN}the deployment script is running.${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Wait for deployment to complete"
echo -e "2. Verify the sequence generator implementation:"
echo -e "   ./aws-setup/check_sequence_generator.sh"
echo -e "3. Monitor training with continuous feed:"
echo -e "   ./aws-setup/live_continuous_monitor.sh all"
echo -e "${BLUE}=================================================================${NC}"
