#!/bin/bash
# Script to generate comparative visualizations of model validation accuracies

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    MODEL VALIDATION ACCURACY COMPARISON    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo "Time: $(date)"
echo ""

# Check if JSON data file exists
JSON_FILE="model_validation_accuracy.json"
if [ ! -f "$JSON_FILE" ]; then
    echo -e "${RED}Error: $JSON_FILE not found.${NC}"
    echo "Please run get_all_model_accuracies.sh first to extract model data."
    exit 1
fi

# Check if Python script exists
PYTHON_SCRIPT="compare_model_accuracies.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script not found: $PYTHON_SCRIPT${NC}"
    echo "Please make sure the script exists in the current directory."
    exit 1
fi

# Check if pandas is installed
python3 -c "import pandas" &>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: pandas not found. Installing required packages...${NC}"
    pip install pandas matplotlib numpy
fi

echo -e "${YELLOW}What type of comparison would you like to generate?${NC}"
echo "1) Top 5 models (default)"
echo "2) Top 10 models"
echo "3) All models with at least 20 epochs"
echo "4) All models (any number of epochs)"
echo "5) Custom settings"
read -p "Enter your choice (1-5): " CHOICE

case $CHOICE in
    1|"")
        # Default: Top 5 models
        TOP=5
        MIN_EPOCHS=10
        echo -e "${GREEN}Comparing top 5 models with at least 10 epochs of training${NC}"
        ;;
    2)
        # Top 10 models
        TOP=10
        MIN_EPOCHS=10
        echo -e "${GREEN}Comparing top 10 models with at least 10 epochs of training${NC}"
        ;;
    3)
        # All models with at least 20 epochs
        TOP=100
        MIN_EPOCHS=20
        echo -e "${GREEN}Comparing all models with at least 20 epochs of training${NC}"
        ;;
    4)
        # All models
        TOP=100
        MIN_EPOCHS=1
        echo -e "${GREEN}Comparing all models (any number of epochs)${NC}"
        ;;
    5)
        # Custom settings
        read -p "Enter number of top models to show: " TOP
        read -p "Enter minimum epochs required: " MIN_EPOCHS
        echo -e "${GREEN}Comparing top $TOP models with at least $MIN_EPOCHS epochs of training${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Using default settings.${NC}"
        TOP=5
        MIN_EPOCHS=10
        echo -e "${GREEN}Comparing top 5 models with at least 10 epochs of training${NC}"
        ;;
esac

# Generate output filename with parameters
OUTPUT_FILE="model_comparison_top${TOP}_min${MIN_EPOCHS}.png"

# Run the Python script with chosen parameters
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Generating comparisons...${NC}"
echo -e "${BLUE}===================================================================${NC}"

python3 "$PYTHON_SCRIPT" --top $TOP --min-epochs $MIN_EPOCHS --output "$OUTPUT_FILE"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${GREEN}Comparison generated successfully!${NC}"
    echo -e "${BLUE}===================================================================${NC}"
    
    # Check if results exist
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${YELLOW}Results are available in:${NC}"
        echo "- $OUTPUT_FILE (Comparison chart)"
        echo "- final_accuracies.png (Bar chart of best accuracies)"
        echo "- model_performance_table.csv (Detailed performance metrics)"
    else
        echo -e "${RED}Warning: Result files were not generated.${NC}"
    fi
else
    echo -e "${RED}Error: Comparison script failed.${NC}"
    echo "Please check the output above for details."
    exit 1
fi
