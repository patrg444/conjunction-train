#!/bin/bash
# Script to test the WAV2VEC V9 attention model locally

echo "===== Running WAV2VEC v9 Attention Model Local Test ====="

# Create local test directories
mkdir -p checkpoints_test

# Run Python script with mock data
python test_v9_attention_locally.py --samples 20 --seq_length 100 --feature_dim 768 --classes 6

if [ $? -eq 0 ]; then
    echo "✅ LOCAL TEST PASSED: The model structure compiles correctly"
    echo "✅ Forward and backward passes successful"
    echo "✅ Model saving/loading works correctly"
    echo ""
    echo "You can now deploy to EC2 using: ./deploy_v9_fix.sh"
else
    echo "❌ LOCAL TEST FAILED"
    echo "Please check the error messages above and fix any issues before deploying."
    exit 1
fi
