#!/bin/bash
# Script to force a clean Conda installation by terminating any hanging processes

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     FORCING CLEAN ENVIRONMENT INSTALLATION                      ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Force kill any hanging Conda processes
echo -e "${YELLOW}Terminating any hanging Conda processes...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
    echo "Current Conda processes:"
    ps aux | grep conda
    
    echo "Killing any Conda processes..."
    pkill -9 -f conda || true
    
    echo "Processes after kill:"
    ps aux | grep conda
EOF

# Check if Miniconda is already installed, and if not, install it
echo -e "${YELLOW}Setting up clean Miniconda installation...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
    if [ -d "$HOME/miniconda" ]; then
        echo "Miniconda installation found."
    else
        echo "Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
    fi
    
    # Add to path
    echo "Adding Miniconda to PATH..."
    if ! grep -q "miniconda/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Initialize for bash shell
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # Make sure conda is available in the current shell
    source $HOME/miniconda/bin/activate
    
    # Test Conda installation
    echo "Testing Conda installation..."
    conda --version
    
    # Update Conda
    echo "Updating Conda..."
    conda update -y conda
    
    # Clean any partial environments
    echo "Removing any emotion_model environment if it exists..."
    conda env remove -n emotion_model -y || true
    
    # Create fresh environment
    echo "Creating fresh emotion_model environment..."
    conda create -y -n emotion_model python=3.8
    
    # Activate and install packages
    source activate emotion_model
    
    echo "Installing core packages..."
    conda install -y -n emotion_model numpy pandas scikit-learn
    
    echo "Installing TensorFlow..."
    conda install -y -n emotion_model -c conda-forge tensorflow=2.9
    
    echo "Installing additional packages..."
    conda install -y -n emotion_model matplotlib h5py
    
    echo "Testing imports..."
    conda run -n emotion_model python -c "import numpy; import tensorflow; import pandas; import matplotlib; import sklearn; print('All imports successful!')"
    
    echo "Environment setup complete!"
    conda env list
EOF

echo -e "${GREEN}Force clean installation completed.${NC}"
echo -e "${YELLOW}You can now run verify_and_launch_training.sh to start the training.${NC}"
