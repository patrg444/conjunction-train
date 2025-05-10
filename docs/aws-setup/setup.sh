#!/bin/bash
# Master setup script for AWS training

# Make all scripts executable
echo "Making scripts executable..."
chmod +x prepare-data.sh
chmod +x aws-instance-setup.sh
chmod +x train-on-aws.sh

# Verify required files exist
echo "Verifying required files..."
required_scripts=(
  "../scripts/train_branched_6class.py"
  "../scripts/sequence_data_generator.py"
  "../scripts/train_branched_dynamic_funcs.py"
  "../requirements.txt"
)

for script in "${required_scripts[@]}"; do
  if [ ! -f "$script" ]; then
    echo "Error: Required file $script not found!"
    echo "Please run this script from the aws-setup directory."
    exit 1
  fi
done

# Verify feature directories exist
if [ ! -d "../ravdess_features_facenet" ]; then
  echo "Warning: Directory ../ravdess_features_facenet not found!"
  echo "Make sure this directory exists before running prepare-data.sh."
fi

if [ ! -d "../crema_d_features_facenet" ]; then
  echo "Warning: Directory ../crema_d_features_facenet not found!"
  echo "Make sure this directory exists before running prepare-data.sh."
fi

# Display next steps
echo ""
echo "Setup complete! Next steps:"
echo "1. Run ./prepare-data.sh to package your code and data"
echo "2. Launch an AWS EC2 instance (see README.md for details)"
echo "3. Transfer files to your EC2 instance"
echo "4. SSH into your instance and run aws-instance-setup.sh"
echo "5. Start training with train-on-aws.sh"
echo ""
echo "For detailed instructions, see README.md"
