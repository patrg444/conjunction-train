#!/bin/bash
# Script to find the model with exactly 84.77% validation accuracy from AWS logs

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# AWS instance details (same as in monitor_fixed_tcn_model.sh)
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed_v2.log"

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    FINDING MODEL WITH 84.77% VALIDATION ACCURACY    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo "Instance: $USERNAME@$INSTANCE_IP"
echo "Log file: $REMOTE_DIR/$LOG_FILE"
echo -e "${BLUE}===================================================================${NC}"
echo ""

# Search for the exact val_accuracy line
echo -e "${YELLOW}Searching for val_accuracy: 0.8477 in log file...${NC}"
VAL_ACCURACY_LINE=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -n 'val_accuracy: 0.8477' ${REMOTE_DIR}/${LOG_FILE}")

if [ -z "$VAL_ACCURACY_LINE" ]; then
    echo -e "${RED}Could not find any model with exactly 84.77% validation accuracy.${NC}"
    echo -e "${YELLOW}Checking for very close matches...${NC}"
    
    # Try matching with a range for possible floating point variations
    CLOSE_MATCHES=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -n 'val_accuracy: 0.847[5-9]' ${REMOTE_DIR}/${LOG_FILE}")
    
    if [ -n "$CLOSE_MATCHES" ]; then
        echo -e "${GREEN}Found similar matches:${NC}"
        echo "$CLOSE_MATCHES"
        
        # Extract line numbers
        LINE_NUMBERS=$(echo "$CLOSE_MATCHES" | cut -d':' -f1)
        
        # Show context for each match
        echo -e "${YELLOW}Extracting model context for each match...${NC}"
        for LINE in $LINE_NUMBERS; do
            # Get 20 lines before and after for context
            CONTEXT=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "sed -n '$((LINE-20)),$((LINE+20))p' ${REMOTE_DIR}/${LOG_FILE}")
            
            echo -e "${BLUE}-------------------------------------------------------------------${NC}"
            echo -e "${GREEN}Context for match at line $LINE:${NC}"
            echo "$CONTEXT"
            
            # Try to find the model checkpoint information
            EPOCH=$(echo "$CONTEXT" | grep -o "Epoch [0-9]\+/[0-9]\+" | tail -1)
            MODEL_SAVE=$(echo "$CONTEXT" | grep -A 10 "Saving model" | head -3)
            
            if [ -n "$EPOCH" ] || [ -n "$MODEL_SAVE" ]; then
                echo -e "${GREEN}Model details:${NC}"
                [ -n "$EPOCH" ] && echo "- $EPOCH"
                [ -n "$MODEL_SAVE" ] && echo -e "- Save info:\n$MODEL_SAVE"
            fi
            echo -e "${BLUE}-------------------------------------------------------------------${NC}"
        done
    else
        echo -e "${RED}No models found with validation accuracy between 84.75% and 84.79%.${NC}"
        echo -e "${YELLOW}Searching for highest validation accuracy...${NC}"
        
        # Find the top 3 best accuracies
        BEST_ACCURACIES=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -o 'val_accuracy: [0-9\.]\+' ${REMOTE_DIR}/${LOG_FILE} | sort -r | head -3")
        echo -e "${GREEN}Top 3 best validation accuracies found:${NC}"
        echo "$BEST_ACCURACIES"
        
        # Find exact location of the best one
        BEST_ONE=$(echo "$BEST_ACCURACIES" | head -1)
        BEST_LINE=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "grep -n '$BEST_ONE' ${REMOTE_DIR}/${LOG_FILE}" | head -1)
        
        if [ -n "$BEST_LINE" ]; then
            LINE_NUM=$(echo "$BEST_LINE" | cut -d':' -f1)
            echo -e "${YELLOW}Extracting context for best model (line $LINE_NUM)...${NC}"
            
            BEST_CONTEXT=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "sed -n '$((LINE_NUM-20)),$((LINE_NUM+20))p' ${REMOTE_DIR}/${LOG_FILE}")
            echo -e "${BLUE}-------------------------------------------------------------------${NC}"
            echo "$BEST_CONTEXT"
            echo -e "${BLUE}-------------------------------------------------------------------${NC}"
        fi
    fi
    exit 1
fi

# Found the exact match
echo -e "${GREEN}Found model with 84.77% validation accuracy!${NC}"
echo "$VAL_ACCURACY_LINE"

# Extract line number
LINE_NUM=$(echo "$VAL_ACCURACY_LINE" | cut -d':' -f1)
echo "Match found at line: $LINE_NUM"

# Get surrounding context for the model (20 lines before and after)
echo -e "${YELLOW}Extracting model context...${NC}"
MODEL_CONTEXT=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "sed -n '$((LINE_NUM-20)),$((LINE_NUM+20))p' ${REMOTE_DIR}/${LOG_FILE}")

echo -e "${BLUE}-------------------------------------------------------------------${NC}"
echo "$MODEL_CONTEXT"
echo -e "${BLUE}-------------------------------------------------------------------${NC}"

# Try to extract key model information
echo -e "${YELLOW}Extracting model details...${NC}"

# Get epoch information
EPOCH=$(echo "$MODEL_CONTEXT" | grep -o "Epoch [0-9]\+/[0-9]\+" | tail -1)
if [ -n "$EPOCH" ]; then
    echo -e "${GREEN}Epoch: $EPOCH${NC}"
fi

# Look for model checkpoint information
SAVE_INFO=$(echo "$MODEL_CONTEXT" | grep -A 10 "Saving model" | head -3)
if [ -n "$SAVE_INFO" ]; then
    echo -e "${GREEN}Model checkpoint information:${NC}"
    echo "$SAVE_INFO"
fi

# Check if there's a model filename mentioned
MODEL_FILE=$(echo "$MODEL_CONTEXT" | grep -o "model[._-][0-9a-zA-Z_.-]\+\.h5")
if [ -n "$MODEL_FILE" ]; then
    echo -e "${GREEN}Model filename: $MODEL_FILE${NC}"
fi

# Check for custom checkpointing pattern in this codebase
CHECKPOINT_PATH=$(echo "$MODEL_CONTEXT" | grep -o "/emotion_training/[0-9a-zA-Z_.-]\+\.h5")
if [ -n "$CHECKPOINT_PATH" ]; then
    echo -e "${GREEN}Full checkpoint path: $CHECKPOINT_PATH${NC}"
    
    # Extract just the filename
    CHECKPOINT_FILE=$(basename "$CHECKPOINT_PATH")
    echo -e "${GREEN}Checkpoint filename: $CHECKPOINT_FILE${NC}"
    
    # Provide command to download this model
    echo -e "${YELLOW}To download this specific model, you can use:${NC}"
    echo "scp -i \"$KEY_FILE\" \"$USERNAME@$INSTANCE_IP:$CHECKPOINT_PATH\" ."
fi

# Look for steps ahead and behind the match for additional information
echo -e "${YELLOW}Checking 50 lines ahead for additional information...${NC}"
AHEAD_CONTEXT=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "sed -n '$((LINE_NUM+20)),$((LINE_NUM+50))p' ${REMOTE_DIR}/${LOG_FILE}" | grep -E "(Saving|Checkpoint|model|accuracy)")

if [ -n "$AHEAD_CONTEXT" ]; then
    echo -e "${BLUE}Additional model information (ahead):${NC}"
    echo "$AHEAD_CONTEXT"
fi

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Search complete!${NC}"
