#!/bin/bash
# Script to sequentially redeploy all TCN models on AWS with enhanced logging

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# AWS instance details
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"

# Timestamp for unique identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="redeployment_logs_${TIMESTAMP}"

# Create local log directory
mkdir -p "$LOG_DIR"

# Ensure key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Function to validate SSH connection
validate_ssh_connection() {
    echo -e "${YELLOW}Validating SSH connection to AWS instance...${NC}"
    
    if ssh -i "$KEY_FILE" -o ConnectTimeout=5 "$USERNAME@$INSTANCE_IP" "echo 'Connection successful'" &>/dev/null; then
        echo -e "${GREEN}SSH connection successful!${NC}"
        return 0
    else
        echo -e "${RED}Failed to connect to AWS instance.${NC}"
        echo -e "${RED}Please check your connection and SSH key.${NC}"
        return 1
    fi
}

# Function to stop any existing training process
stop_existing_training() {
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${YELLOW}    STOPPING ANY EXISTING TRAINING PROCESSES    ${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    # Use the existing stop script
    ./stop_tcn_large_training.sh
    
    # Additional cleanup to make sure everything is stopped
    echo -e "${YELLOW}Additional cleanup to ensure all processes are stopped...${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
        killall -9 python3 2>/dev/null || true
        rm -f ${REMOTE_DIR}/*_pid.txt 2>/dev/null || true
    "
    
    echo -e "${GREEN}All training processes terminated.${NC}"
    echo -e "${BLUE}=================================================================${NC}"
}

# Function to deploy a model using a specific deployment script
deploy_model() {
    local script_name="$1"
    local model_name="$2"
    local wait_time="$3"
    
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${GREEN}    DEPLOYING MODEL: ${YELLOW}$model_name${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${YELLOW}Using deployment script:${NC} $script_name"
    
    # Execute the deployment script and log output
    if [ -f "$script_name" ]; then
        chmod +x "$script_name"
        echo "Starting deployment at $(date)" > "${LOG_DIR}/${model_name}_deployment.log"
        ./"$script_name" | tee -a "${LOG_DIR}/${model_name}_deployment.log"
        
        echo -e "${GREEN}Model $model_name deployment initiated!${NC}"
        
        # Check if there's a monitoring script created by the deployment
        if [[ $script_name == *"deploy_"* ]]; then
            monitor_script="monitor_$(echo $script_name | sed 's/deploy_//')"
            monitor_script="${monitor_script/.sh/_model.sh}"
            
            if [ -f "$monitor_script" ]; then
                echo -e "${YELLOW}Found monitoring script:${NC} $monitor_script"
                echo -e "${YELLOW}Copying monitoring script to:${NC} ${LOG_DIR}/${model_name}_monitor.sh"
                cp "$monitor_script" "${LOG_DIR}/${model_name}_monitor.sh"
            fi
        fi
        
        # If wait time is specified, wait before deploying the next model
        if [ -n "$wait_time" ]; then
            echo -e "${YELLOW}Waiting for $wait_time minutes before the next deployment...${NC}"
            echo "Waiting period started at $(date)" >> "${LOG_DIR}/${model_name}_deployment.log"
            
            # Convert minutes to seconds
            sleep_seconds=$((wait_time * 60))
            sleep $sleep_seconds
            
            echo "Waiting period completed at $(date)" >> "${LOG_DIR}/${model_name}_deployment.log"
        fi
    else
        echo -e "${RED}Error: Deployment script not found: $script_name${NC}"
    fi
    
    echo -e "${BLUE}=================================================================${NC}"
}

# Function to monitor a deployed model for a brief period
monitor_model() {
    local model_name="$1"
    local monitor_time="$2"  # in seconds
    
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${GREEN}    MONITORING MODEL: ${YELLOW}$model_name${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    # Check if there's a monitoring script for this model in the log directory
    if [ -f "${LOG_DIR}/${model_name}_monitor.sh" ]; then
        echo -e "${YELLOW}Using monitoring script:${NC} ${LOG_DIR}/${model_name}_monitor.sh"
        
        # Make the script executable
        chmod +x "${LOG_DIR}/${model_name}_monitor.sh"
        
        # Run with timeout
        echo -e "${YELLOW}Monitoring for $monitor_time seconds...${NC}"
        timeout $monitor_time "${LOG_DIR}/${model_name}_monitor.sh}" | tee -a "${LOG_DIR}/${model_name}_monitoring.log"
    else
        echo -e "${YELLOW}No specific monitoring script found for $model_name.${NC}"
        echo -e "${YELLOW}Using general monitoring approach...${NC}"
        
        # Generic monitoring - get process status and log tail
        ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
            echo 'Process status:'
            ps aux | grep 'train_.*\.py' | grep -v grep
            echo ''
            echo 'Recent log lines:'
            find ${REMOTE_DIR} -name '*.log' -type f -mmin -60 -exec ls -ltr {} \; | tail -5
            find ${REMOTE_DIR} -name '*.log' -type f -mmin -60 -exec tail -10 {} \; | tail -30
        " | tee -a "${LOG_DIR}/${model_name}_monitoring.log"
    fi
    
    echo -e "${BLUE}=================================================================${NC}"
}

# Function to get validation accuracy from an actively training model
check_val_accuracy() {
    local model_name="$1"
    
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${GREEN}    CHECKING VALIDATION ACCURACY: ${YELLOW}$model_name${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    # Look for val_accuracy in logs
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "
        echo 'Searching for validation accuracy...'
        find ${REMOTE_DIR} -name '*.log' -type f -mmin -60 -exec grep -l 'val_accuracy' {} \; | xargs -I{} grep -m 3 'val_accuracy' {} | tail -3 || echo 'No validation accuracy found'
    " | tee -a "${LOG_DIR}/${model_name}_accuracy.log"
    
    echo -e "${BLUE}=================================================================${NC}"
}

# Main execution
echo -e "${BLUE}===========================================================================${NC}"
echo -e "${GREEN}${BOLD}REDEPLOYING ALL TCN MODELS ON AWS${NC}"
echo -e "${BLUE}===========================================================================${NC}"
echo -e "${YELLOW}Timestamp:${NC} $TIMESTAMP"
echo -e "${YELLOW}Log directory:${NC} $LOG_DIR"
echo -e "${BLUE}===========================================================================${NC}"

# Step 1: Validate connection
if ! validate_ssh_connection; then
    exit 1
fi

# Step 2: Stop any existing training
stop_existing_training

# Step 3: Deploy and monitor each model with a waiting period between deployments

# Models to deploy - add all deployment scripts to be executed
# Models to deploy - add all deployment scripts to be executed
models=("deploy_fixed_tcn_large_model.sh" "deploy_fixed_tcn_large_model_simple.sh" "deploy_fixed_tcn_model_v2.sh")
model_names=("fixed_tcn_large" "fixed_tcn_large_simple" "fixed_tcn_v2")

# Record start time
echo "Redeployment started at: $(date)" > "${LOG_DIR}/redeployment_summary.log"

# Deploy each model
model_count=1
total_models=${#models[@]}

for i in $(seq 0 $((total_models-1))); do
    script="${models[$i]}"
    model_name="${model_names[$i]}"
    
    echo -e "${YELLOW}Deploying model ${model_count}/${total_models}: ${model_name}${NC}"

    # Determine wait time - last model doesn't need waiting period
    wait_time=0  # default 0 minutes (removed delay)
    if [ $model_count -eq $total_models ]; then
        wait_time=""
    fi
    
    # Deploy the model
    deploy_model "$script" "$model_name" "$wait_time"
    
    # Wait a bit for deployment to initialize
    echo -e "${YELLOW}Waiting for deployment to initialize...${NC}"
    sleep 60
    
    # Monitor briefly to ensure it's running
    monitor_model "$model_name" 120  # 2 minutes of monitoring
    
    # Check initial validation accuracy
    check_val_accuracy "$model_name"
    
    # Increment model counter
    ((model_count++))
done

# Record end time
echo "Redeployment completed at: $(date)" >> "${LOG_DIR}/redeployment_summary.log"

echo -e "${BLUE}===========================================================================${NC}"
echo -e "${GREEN}All models have been redeployed!${NC}"
echo -e "${YELLOW}Logs are available in:${NC} $LOG_DIR"
echo -e "${YELLOW}To monitor specific models, use the monitoring scripts in the log directory:${NC}"
ls -1 ${LOG_DIR}/*_monitor.sh 2>/dev/null

echo -e "${MAGENTA}Note: Models are training in sequence. Each will run to completion before the next one starts.${NC}"
echo -e "${MAGENTA}If you want to check progress later, you can use:${NC}"
echo -e "  ./continuous_tcn_monitoring_crossplatform.sh"
echo -e "${BLUE}===========================================================================${NC}"
