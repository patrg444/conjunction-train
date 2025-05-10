#!/bin/bash
# Script to run a smoke test of the WAV2VEC v9 attention model
# This will verify the model structure compiles and can process a mini batch

echo "===== Running WAV2VEC v9 Attention Model Smoke Test ====="
echo "Setting up environment for smoke test..."

# Set the smoke test environment variable
export SMOKE_TEST=1

# Run the model with smoke test enabled
echo "Running the model with smoke test mode enabled..."
python fixed_v9_attention.py

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo "===== SMOKE TEST PASSED ====="
    echo "The model structure is valid and can process data correctly."
    echo "You can now deploy to EC2 using: ./deploy_v9_fix.sh"
else
    echo "===== SMOKE TEST FAILED ====="
    echo "Please check the error messages above and fix any issues before deploying."
fi

# Clean up environment variable
unset SMOKE_TEST
