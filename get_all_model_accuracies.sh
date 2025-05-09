#!/bin/bash
# Script to extract validation accuracy from all model training logs on AWS

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    EXTRACTING VALIDATION ACCURACY FROM ALL MODEL LOGS    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo "Time: $(date)"
echo ""

# Check if SSH key exists
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Check if Python script exists
PYTHON_SCRIPT="extract_all_models_val_accuracy.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script not found: $PYTHON_SCRIPT${NC}"
    echo "Please make sure the script exists in the current directory."
    exit 1
fi

# Check if matplotlib is installed
python3 -c "import matplotlib" &>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: matplotlib not found. Installing required packages...${NC}"
    pip install matplotlib numpy
fi

# Check if we can connect to the AWS instance
echo -e "${YELLOW}Checking AWS instance connection...${NC}"
ssh -i "$KEY_FILE" -o ConnectTimeout=5 "ec2-user@3.235.76.0" "echo 'Connection successful'" &>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Could not connect to AWS instance.${NC}"
    echo "Please check your AWS instance and SSH key."
    exit 1
fi

echo -e "${GREEN}AWS instance connection successful.${NC}"
echo ""

# Ask if user wants to generate plots
echo -e "${YELLOW}Do you want to generate accuracy plots for each model? [y/N]${NC}"
read -n 1 -r
echo ""

PLOT_OPTION=""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    PLOT_OPTION="--plot"
    echo -e "${GREEN}Will generate accuracy plots for each model.${NC}"
else
    echo -e "${YELLOW}Will not generate plots.${NC}"
fi

# Run the Python script
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Running extraction script...${NC}"
echo -e "${BLUE}===================================================================${NC}"

python3 "$PYTHON_SCRIPT" $PLOT_OPTION

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${GREEN}Extraction completed successfully!${NC}"
    echo -e "${BLUE}===================================================================${NC}"
    
    # Check if results exist
    if [ -f "model_validation_summary.csv" ]; then
        echo -e "${YELLOW}Results are available in:${NC}"
        echo "- model_validation_summary.csv (CSV summary)"
        echo "- model_validation_accuracy.json (Detailed JSON data)"
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "- model_accuracy_plots/ (Accuracy plots)"
        fi
    else
        echo -e "${RED}Warning: Result files were not generated.${NC}"
    fi
else
    echo -e "${RED}Error: Extraction script failed.${NC}"
    echo "Please check the output above for details."
    exit 1
fi
