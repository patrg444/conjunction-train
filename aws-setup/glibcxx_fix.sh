#!/bin/bash
# Script to fix GLIBCXX_3.4.29 issue

# Install the development tools
sudo yum group install -y "Development Tools"

# Install centos-release-scl repository
sudo yum install -y centos-release-scl

# Install devtoolset-11
sudo yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++

# Create directory for the newer libstdc++
mkdir -p ~/lib

# Enable the devtoolset-11
source /opt/rh/devtoolset-11/enable

# Copy the newer libstdc++ libraries to our custom directory
cp /opt/rh/devtoolset-11/root/usr/lib/gcc/x86_64-redhat-linux/11/libstdc++.so.6* ~/lib/

# Check if the newer version of GLIBCXX is now available in the copied library
strings ~/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29
if [ $? -eq 0 ]; then
    echo "GLIBCXX_3.4.29 is available in the new library"
else
    echo "GLIBCXX_3.4.29 is still not available. Further troubleshooting required."
    exit 1
fi

# Create a wrapper script to run Python with the new library
cat > ~/emotion_training/run_with_new_libstdcxx.sh << 'INNEREOF'
#!/bin/bash
# Script to run Python with the newer libstdc++

# Set the LD_LIBRARY_PATH to use our custom libstdc++ before the system one
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

# Activate the conda environment
source $HOME/miniconda/bin/activate emotion_model

# Set the Python path to include the emotion_training directory
export PYTHONPATH=$HOME/emotion_training:$PYTHONPATH

# Go to the emotion_training directory
cd $HOME/emotion_training

# Run the training script with the newer libstdc++
echo "Starting LSTM attention model training at $(date)"
echo "Using libstdc++ from: $HOME/lib"
echo "Checking for GLIBCXX_3.4.29:"
strings $HOME/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

# Install TensorFlow with pip instead of conda
pip install tensorflow==2.9.0 scipy==1.10.1

# Run the training script
python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log
echo "Training completed at $(date) with exit code $?"
INNEREOF

chmod +x ~/emotion_training/run_with_new_libstdcxx.sh

# Kill any existing training sessions
pkill -f train_branched_attention.py || true
tmux kill-session -t lstm_attention 2>/dev/null || true

# Launch new training in tmux with the fixed library
tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_with_new_libstdcxx.sh"

echo "LSTM attention model training launched with fixed GLIBCXX library"
