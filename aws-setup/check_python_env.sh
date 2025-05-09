#!/bin/bash
# Script to check Python environments on the EC2 instance

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Checking Python environments on the server..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
echo "System PATH:"
echo $PATH
echo

echo "Available Python versions:"
which -a python python2 python3 python2.7 python3.6 python3.7 python3.8 python3.9
echo

echo "Python default version:"
python --version
echo

echo "Python3 version (if available):"
python3 --version 2>/dev/null || echo "python3 not found"
echo

echo "Checking for conda/virtualenv environments:"
conda info -e 2>/dev/null || echo "conda not found"
echo

echo "Checking for virtualenv:"
pip list | grep virtualenv 2>/dev/null || echo "pip virtualenv not found"
echo

echo "Checking if numpy is available in default Python:"
python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null || echo "NumPy not available in default Python"
echo

echo "Checking if numpy is available in Python3 (if available):"
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null || echo "NumPy not available in Python3 or Python3 not found"
echo

echo "Checking for TensorFlow in default Python:"
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)" 2>/dev/null || echo "TensorFlow not available in default Python"
echo

echo "Checking for TensorFlow in Python3 (if available):"
python3 -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)" 2>/dev/null || echo "TensorFlow not available in Python3 or Python3 not found"
echo

echo "Looking for virtual environments in the home directory:"
find ~/ -name "activate" | grep -E "bin/activate$"
echo

echo "Checking previous successful Python runs (logs or history):"
grep -r "import numpy" ~/emotion_training/scripts/ 2>/dev/null | head -5
grep -r "import tensorflow" ~/emotion_training/scripts/ 2>/dev/null | head -5
echo

echo "Checking if there's an existing virtualenv in the emotion training directory:"
ls -la ~/emotion_training/.venv/bin/activate 2>/dev/null || echo "No virtualenv found in emotion_training directory"
ls -la ~/emotion_training/venv/bin/activate 2>/dev/null || echo "No venv found in emotion_training directory"
EOF

echo "Environment check complete."
