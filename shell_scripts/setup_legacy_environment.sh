#!/bin/bash
# Script to create a dedicated conda environment with an older TensorFlow version
# that's compatible with the original model (supporting the time_major parameter)

# ANSI colors for better output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}Legacy TensorFlow Environment Setup${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Miniconda or Anaconda first:${NC}"
    echo -e "${YELLOW}https://docs.conda.io/en/latest/miniconda.html${NC}"
    exit 1
fi

# Environment name
ENV_NAME="emotion_recognition_legacy"

# Check if environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n $ENV_NAME
    else
        echo -e "${YELLOW}Using existing environment. Make sure it has the correct packages.${NC}"
        echo -e "${GREEN}You can activate it with:${NC} conda activate $ENV_NAME"
        exit 0
    fi
fi

echo -e "${GREEN}Creating conda environment '$ENV_NAME' with compatible TensorFlow...${NC}"

# Create environment with Python 3.8 (which works with TensorFlow 1.15)
conda create -y -n $ENV_NAME python=3.8

# Activate environment and install packages
echo -e "${GREEN}Installing packages...${NC}"

# Use conda run to execute commands in the new environment
# We use TensorFlow 1.15 which supports the time_major parameter in LSTM layers
conda run -n $ENV_NAME pip install tensorflow==1.15.0
conda run -n $ENV_NAME pip install numpy==1.18.5
conda run -n $ENV_NAME pip install opencv-python
conda run -n $ENV_NAME pip install matplotlib
conda run -n $ENV_NAME pip install opensmile

echo -e "${GREEN}Installing additional dependencies...${NC}"
conda run -n $ENV_NAME pip install scikit-learn==0.24.2
conda run -n $ENV_NAME pip install pandas
conda run -n $ENV_NAME pip install soundfile
conda run -n $ENV_NAME pip install librosa

# Create a wrapper script to run the original pipeline using the legacy environment
echo -e "${GREEN}Creating wrapper script for original pipeline...${NC}"

cat > run_original_pipeline.sh << 'EOF'
#!/bin/bash
# Wrapper script to run the original emotion recognition pipeline
# using the legacy environment with TensorFlow 1.15

# Activate the legacy environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate emotion_recognition_legacy

# Run the original pipeline script
python scripts/realtime_emotion_recognition.py "$@"

# Deactivate the environment when done
conda deactivate
EOF

chmod +x run_original_pipeline.sh

echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}To use the original model with the legacy TensorFlow:${NC}"
echo -e "1. ${YELLOW}Activate the environment:${NC} conda activate $ENV_NAME"
echo -e "2. ${YELLOW}Run the pipeline:${NC} ./run_original_pipeline.sh"
echo -e ""
echo -e "${GREEN}Or simply use the wrapper script:${NC}"
echo -e "${YELLOW}./run_original_pipeline.sh${NC}"
echo -e ""
echo -e "${BLUE}This environment uses TensorFlow 1.15 which supports${NC}"
echo -e "${BLUE}the time_major parameter in LSTM layers.${NC}"
echo -e "${BLUE}=============================================${NC}"
