#!/bin/bash
# Script to create Conda environment for LSTM attention model training

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Existing instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     CREATING CONDA ENVIRONMENT FOR LSTM ATTENTION MODEL         ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# SSH into the instance and create the environment
echo -e "${YELLOW}Setting up Conda environment on EC2 instance...${NC}"
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP << 'EOF'
    # Make sure we have conda
    if ! command -v conda &> /dev/null; then
        echo "Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
    fi

    # Source conda command
    source $HOME/miniconda/bin/activate

    # Create conda environment for TensorFlow
    echo "Creating emotion_model environment..."
    conda create -y -n emotion_model python=3.8

    # Activate environment and install packages
    source activate emotion_model
    
    echo "Installing TensorFlow and other requirements..."
    conda install -y numpy pandas scipy scikit-learn matplotlib
    conda install -y tensorflow=2.9.0
    conda install -y h5py
    
    # Fix the sequence_data_generator issue
    if [ ! -f ~/emotion_training/scripts/sequence_data_generator.py ]; then
        echo "sequence_data_generator.py not found. Checking if it's in train_branched_attention.py..."
        if grep -q "class SequenceDataGenerator" ~/emotion_training/scripts/train_branched_attention.py; then
            echo "Found SequenceDataGenerator class in train_branched_attention.py"
            echo "Creating sequence_data_generator.py..."
            
            # Extract SequenceDataGenerator class
            cat > ~/emotion_training/scripts/sequence_data_generator.py << 'EOFINNER'
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
EOFINNER
        fi
    fi
    
    # Create a wrapper script to run the training with the conda environment
    cat > ~/emotion_training/run_with_conda.sh << 'EOFINNER'
#!/bin/bash
# Activate the conda environment and run training
source $HOME/miniconda/bin/activate emotion_model

cd ~/emotion_training
export PYTHONPATH=$HOME/emotion_training:$PYTHONPATH

echo "Starting LSTM attention model training at $(date)"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Modules in PYTHONPATH: $PYTHONPATH"

python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log
echo "Training completed at $(date) with exit code $?"
EOFINNER
    
    chmod +x ~/emotion_training/run_with_conda.sh
    
    # Kill any existing training sessions
    pkill -f train_branched_attention.py || true
    tmux kill-session -t lstm_attention 2>/dev/null || true
    
    # Create model output directory
    mkdir -p ~/emotion_training/models/attention_focal_loss
    
    # Launch new training in tmux
    tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_with_conda.sh"
    
    # Check if it's running
    echo "Waiting 10 seconds for training to start..."
    sleep 10
    if pgrep -f train_branched_attention.py > /dev/null; then
        echo "LSTM attention model training started successfully."
    else
        echo "Warning: Training process may not have started successfully."
        
        # Check for errors in log
        if [ -f ~/emotion_training/lstm_attention_training.log ]; then
            echo "Last 15 lines of training log:"
            tail -15 ~/emotion_training/lstm_attention_training.log
        fi
    fi
EOF

echo -e "${GREEN}Conda environment setup for LSTM attention model training completed.${NC}"
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
