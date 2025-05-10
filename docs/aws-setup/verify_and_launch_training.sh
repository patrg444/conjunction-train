#!/bin/bash
# Script to verify environment setup and launch the LSTM attention model training

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
echo -e "${BLUE}     VERIFYING ENVIRONMENT AND LAUNCHING TRAINING                ${NC}"
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

# Check if Conda is still installing packages
echo -e "${YELLOW}Checking if Conda installation is still in progress...${NC}"
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "pgrep -f 'conda install'" &> /dev/null; then
    echo -e "${YELLOW}Conda is still installing packages. Please wait for it to complete.${NC}"
    echo -e "${YELLOW}You can run this script again later to check the status.${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ps aux | grep 'conda install' | grep -v grep"
    exit 0
fi

# Check if miniconda is properly installed
echo -e "${YELLOW}Checking if Conda environment is properly set up...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "command -v conda" &> /dev/null; then
    echo -e "${RED}Conda command not found. Installing Miniconda...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p $HOME/miniconda
        rm ~/miniconda.sh
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
EOF
fi

# Check if emotion_model environment exists and create if needed
echo -e "${YELLOW}Checking if emotion_model environment exists...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "source ~/.bashrc && conda env list | grep -q emotion_model"; then
    echo -e "${YELLOW}Creating emotion_model environment...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        source ~/.bashrc
        conda create -y -n emotion_model python=3.8
        conda activate emotion_model
        conda install -y numpy pandas scipy scikit-learn matplotlib
        conda install -y -c conda-forge tensorflow=2.9.0
        conda install -y h5py
EOF
else
    echo -e "${GREEN}emotion_model environment exists.${NC}"
fi

# Create the sequence data generator if needed
echo -e "${YELLOW}Checking if sequence_data_generator.py exists...${NC}"
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "[ ! -f ~/emotion_training/scripts/sequence_data_generator.py ]"; then
    echo -e "${YELLOW}Creating sequence_data_generator.py...${NC}"
    
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
cat > ~/emotion_training/scripts/sequence_data_generator.py << 'INNEREOFEOF'
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class SequenceDataGenerator(Sequence):
    """Generates data for Keras sequence models"""
    def __init__(self, features, labels, batch_size=32, shuffle=True, 
                 max_seq_length=None, padding='post', truncating='post',
                 attention_weights=False):
        """Initialization"""
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.truncating = truncating
        self.attention_weights = attention_weights
        self.indices = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.features) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(batch_indices)
        
        if self.attention_weights:
            # For models with attention, return dummy attention weights (ones)
            return X, y, np.ones((X.shape[0], X.shape[1])) 
        else:
            return X, y
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.indices = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_indices):
        """Generates data containing batch_size samples"""
        batch_features = [self.features[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        # Process sequences if max_seq_length is specified
        if self.max_seq_length is not None:
            # Find actual max length in this batch
            if self.max_seq_length == 'max':
                max_len = max(len(seq) for seq in batch_features)
            else:
                max_len = self.max_seq_length
            
            # Pad sequences
            padded_sequences = []
            for seq in batch_features:
                if len(seq) > max_len:
                    if self.truncating == 'post':
                        padded_seq = seq[:max_len]
                    else:  # 'pre'
                        padded_seq = seq[-max_len:]
                else:
                    padded_seq = np.zeros((max_len, seq.shape[1]))
                    if self.padding == 'post':
                        padded_seq[:len(seq)] = seq
                    else:  # 'pre'
                        padded_seq[-len(seq):] = seq
                padded_sequences.append(padded_seq)
            
            return np.array(padded_sequences), np.array(batch_labels)
        else:
            return np.array(batch_features), np.array(batch_labels)
INNEREOFEOF
EOF
    
    echo -e "${GREEN}sequence_data_generator.py created.${NC}"
else
    echo -e "${GREEN}sequence_data_generator.py already exists.${NC}"
fi

# Create the training launcher script
echo -e "${YELLOW}Creating training launcher script...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
cat > ~/emotion_training/launch_lstm_attention.sh << 'INNEREOFEOF'
#!/bin/bash
# Script to launch LSTM attention model training in the Conda environment

# Print startup information
echo "===== LSTM Attention Model Training ====="
echo "Starting at: $(date)"
echo "Conda environment: emotion_model"

# Activate Conda environment
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate emotion_model

# Environment details
echo "Python version: $(python --version)"
echo "Conda environment path: $CONDA_PREFIX"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "NumPy version: $(python -c 'import numpy as np; print(np.__version__)')"

# Ensure model directory exists
mkdir -p ~/emotion_training/models/attention_focal_loss

# Set Python path to include emotion_training
export PYTHONPATH="$HOME/emotion_training:$PYTHONPATH"

# Launch the training
cd ~/emotion_training
python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log

echo "Training completed at: $(date)"
INNEREOFEOF

chmod +x ~/emotion_training/launch_lstm_attention.sh
EOF

echo -e "${GREEN}Training launcher script created.${NC}"

# Create model output directory
echo -e "${YELLOW}Ensuring model output directory exists...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "mkdir -p ~/emotion_training/models/attention_focal_loss"

# Check if training is already running
echo -e "${YELLOW}Checking if training is already running...${NC}"
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "pgrep -f train_branched_attention.py" &> /dev/null; then
    echo -e "${GREEN}Training is already running. You can monitor it using the monitoring scripts.${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ps aux | grep train_branched_attention.py | grep -v grep"
else
    # Start the training in a tmux session
    echo -e "${YELLOW}Starting training in a tmux session...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOF'
        # Kill any existing tmux sessions
        tmux kill-session -t lstm_attention 2>/dev/null || true
        
        # Start new tmux session
        tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./launch_lstm_attention.sh"
        
        echo "Training launched in tmux session 'lstm_attention'"
        
        # Check if it started
        sleep 5
        if pgrep -f train_branched_attention.py > /dev/null; then
            echo "Training process is running."
        else
            echo "Warning: Training process may not have started properly."
        fi
EOF
fi

echo -e "${GREEN}Verification and launch completed.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      MONITORING OPTIONS${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "bash aws-setup/stream_logs.sh"
echo ""
echo -e "${YELLOW}To check CPU usage:${NC}"
echo -e "bash aws-setup/monitor_cpu_usage.sh"
echo ""
echo -e "${YELLOW}To check training files:${NC}"
echo -e "bash aws-setup/check_training_files.sh"
echo -e "${BLUE}=================================================================${NC}"
