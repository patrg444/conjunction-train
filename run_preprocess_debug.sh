#!/bin/bash
# Simple script to activate conda, run preprocessing, and log everything.
LOG_FILE='/home/ec2-user/emotion_training/preprocess_spectrograms.log'

# Redirect all output from this point onwards to the log file
exec > "\$LOG_FILE" 2>&1

echo "Starting preprocessing debug run..."
echo "Log file: \$LOG_FILE"

# --- Conda Initialization ---
echo "Attempting Conda Initialization..."
CONDA_SH_PATH=""

# Try common paths for conda.sh
if [[ -f "/home/ec2-user/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH_PATH="/home/ec2-user/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    CONDA_SH_PATH="/opt/conda/etc/profile.d/conda.sh"
fi

if [[ -n "\$CONDA_SH_PATH" ]]; then
    echo "Sourcing \$CONDA_SH_PATH..."
    # Use '.' instead of 'source' for better compatibility
    . "\$CONDA_SH_PATH"
    INIT_EXIT_CODE=\$?
    if [[ "\$INIT_EXIT_CODE" -ne 0 ]]; then
        echo "ERROR: Sourcing conda.sh failed with exit code \$INIT_EXIT_CODE."
        # Try activating directly anyway, might work if shell is already initialized
    else
        echo "conda.sh sourced successfully."
    fi
else
    echo "WARNING: Could not find conda.sh at common paths. Will attempt direct activation."
fi
# --- End Conda Initialization ---

echo "Attempting to activate Conda environment 'emotion_env'..."
conda activate emotion_env
ACTIVATE_EXIT_CODE=\$?
if [[ "\$ACTIVATE_EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: Failed to activate conda environment 'emotion_env' (Exit Code: \$ACTIVATE_EXIT_CODE)."
    exit 1
fi
echo "Conda environment activated successfully."

echo "Attempting to change directory..."
cd /home/ec2-user/emotion_training/scripts || { echo 'ERROR: Failed to cd into scripts directory.'; exit 1; }
echo "Changed directory successfully."

echo "Running preprocessing script..."
# Run the python script
python preprocess_spectrograms.py
PREPROCESS_EXIT_CODE=\$?

echo "Preprocessing script finished with exit code \$PREPROCESS_EXIT_CODE."
exit \$PREPROCESS_EXIT_CODE
