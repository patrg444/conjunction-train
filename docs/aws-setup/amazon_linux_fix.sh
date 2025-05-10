#!/bin/bash
# Fix GLIBCXX_3.4.29 issue on Amazon Linux 2

# Install GCC 9 from Amazon Linux repositories
echo "Installing GCC 9 from Amazon Linux Extra repository..."
sudo amazon-linux-extras install -y gcc9

# Install development tools
sudo yum group install -y "Development Tools"
sudo yum install -y gcc9-c++

# Setup a directory for the newer libraries
mkdir -p ~/custom_lib

# Compile a newer libstdc++ from source
echo "Building libstdc++ compatible with GLIBCXX_3.4.29..."
cd ~
if [ ! -d "gcc-10.2.0" ]; then
    wget https://ftp.gnu.org/gnu/gcc/gcc-10.2.0/gcc-10.2.0.tar.gz
    tar -xzf gcc-10.2.0.tar.gz
    cd gcc-10.2.0
    ./contrib/download_prerequisites
    cd ~
fi

cd gcc-10.2.0
mkdir -p build
cd build

# Configure and build only libstdc++
../configure --prefix=$HOME/gcc-10-libs --disable-multilib --enable-languages=c,c++ --disable-bootstrap
make -j$(nproc) all-target-libstdc++-v3
make install-target-libstdc++-v3

# Copy the libstdc++ library to our custom directory
cp ~/gcc-10-libs/lib64/libstdc++.so.6* ~/custom_lib/

# Verify the GLIBCXX version
echo "Verifying GLIBCXX versions in the newly built library:"
strings ~/custom_lib/libstdc++.so.6 | grep GLIBCXX

# Create a wrapper script to use the new library with Python
cat > ~/emotion_training/run_with_fixed_libs.sh << 'INNEREOF'
#!/bin/bash
# Run Python with the newer libstdc++ library

# Set LD_LIBRARY_PATH to use our custom libstdc++ library first
export LD_LIBRARY_PATH=$HOME/custom_lib:$LD_LIBRARY_PATH

# Activate conda environment
source $HOME/miniconda/bin/activate emotion_model

# Setup Python path for our project
export PYTHONPATH=$HOME/emotion_training:$PYTHONPATH

# Change to the emotion_training directory
cd $HOME/emotion_training

# Install libraries with pip (not conda) which avoids some dependency conflicts
pip install tensorflow==2.8.0
pip install scipy matplotlib pandas scikit-learn h5py

# Run the training script with the newer libstdc++
echo "Starting LSTM attention model training at $(date)"
echo "Using custom libstdc++ from: $HOME/custom_lib"
echo "Checking for GLIBCXX_3.4.29:"
strings $HOME/custom_lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

python scripts/train_branched_attention.py 2>&1 | tee lstm_attention_training.log
echo "Training completed at $(date) with exit code $?"
INNEREOF

chmod +x ~/emotion_training/run_with_fixed_libs.sh

# Kill any existing training sessions
pkill -f train_branched_attention.py || true
tmux kill-session -t lstm_attention 2>/dev/null || true

# Launch new training in tmux with the fixed library
tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_with_fixed_libs.sh"

echo "LSTM attention model training launched with fixed libraries"
echo "You can check the logs with: 'tmux attach -t lstm_attention'"
