#!/bin/bash
# Script to check and ensure the sequence_data_generator.py file includes all fixes

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if connection file exists
if [ ! -f aws-setup/lstm_attention_model_connection.txt ]; then
    echo -e "${RED}Connection details not found. Please run deploy_lstm_attention_model.sh first.${NC}"
    exit 1
fi

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}     CHECKING SEQUENCE DATA GENERATOR IMPLEMENTATION             ${NC}"
echo -e "${BLUE}==================================================================${NC}"

# Create a temporary file with verification code
cat > tmp_verifier.py << 'EOF'
"""
Verifies that the sequence_data_generator.py has all the required fixes
for handling variable-length sequences correctly.
"""
import sys
import os

# Find the script
generator_path = os.path.join('scripts', 'sequence_data_generator.py')
if not os.path.exists(generator_path):
    print(f"ERROR: {generator_path} not found!")
    sys.exit(1)

with open(generator_path, 'r') as f:
    content = f.read()

# Check for essential features
checks = {
    "Empty batch handling": "if len(batch_indices) == 0:",
    "Default parameter in max()": "max((len(video) for video in batch_video), default=1)",
    "Empty tensor fallback": "empty_video = np.zeros((1, 1, self.video_feature_dim)",
    "Re-normalization": "a_sum = tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon()",
    "Mask application": "mask = tf.reduce_any(tf.not_equal(x, 0), axis=-1, keepdims=True)",
    "Masking preservation": "a = a * mask",
}

results = {}
for name, pattern in checks.items():
    results[name] = pattern in content

# Print results
print("\nSequence Data Generator Verification:")
print("-" * 50)

all_passed = True
for name, found in results.items():
    status = f"{GREEN}FOUND{NC}" if found else f"{RED}MISSING{NC}"
    print(f"  {name}: {status}")
    if not found:
        all_passed = False

print("-" * 50)
if all_passed:
    print(f"{GREEN}All required fixes are present in the sequence_data_generator.py file.{NC}")
    print(f"{GREEN}The implementation should handle variable-length sequences correctly.{NC}")
else:
    print(f"{RED}Some fixes are missing! The sequence_data_generator.py file needs to be updated.{NC}")
    print(f"{YELLOW}Consider copying the fixed version from your local machine.{NC}")
    sys.exit(1)

EOF

# Copy and run the verification script on the AWS instance
echo -e "${YELLOW}Copying verification script to AWS instance...${NC}"
scp -i $KEY_FILE -o StrictHostKeyChecking=no tmp_verifier.py ec2-user@$INSTANCE_IP:~/emotion_training/tmp_verifier.py

echo -e "${YELLOW}Running verification on AWS instance...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cd ~/emotion_training && python3 tmp_verifier.py"

# Check if verification failed
if [ $? -ne 0 ]; then
    echo -e "${RED}Verification failed! Uploading fixed sequence_data_generator.py...${NC}"
    
    # Copy the local fixed version to the AWS instance
    scp -i $KEY_FILE -o StrictHostKeyChecking=no scripts/sequence_data_generator.py ec2-user@$INSTANCE_IP:~/emotion_training/scripts/sequence_data_generator.py
    
    echo -e "${YELLOW}Re-running verification...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "cd ~/emotion_training && python3 tmp_verifier.py"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Fixed sequence_data_generator.py successfully uploaded and verified.${NC}"
    else
        echo -e "${RED}Verification still failing. Please check manually.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Sequence data generator implementation verified successfully.${NC}"
fi

# Clean up
rm -f tmp_verifier.py
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "rm -f ~/emotion_training/tmp_verifier.py"

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}Verification complete!${NC}"
echo -e "${BLUE}==================================================================${NC}"
