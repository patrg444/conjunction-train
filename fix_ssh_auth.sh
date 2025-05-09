#!/bin/bash
# Fix script to update SSH key paths in our scripts

# Find the likely correct SSH key path from the original script
ORIGINAL_KEY_PATH=$(grep -o "EC2_KEY=\"[^\"]*\"" wav2vec_data_setup.sh | cut -d'"' -f2)
echo "Found original key path: $ORIGINAL_KEY_PATH"

# Update the improved_wav2vec_data_setup.sh to use the correct key
sed -i'.bak' "s|SSH_KEY=\"~/.ssh/id_rsa\"|SSH_KEY=\"$ORIGINAL_KEY_PATH\"|g" improved_wav2vec_data_setup.sh

# Update the deploy_attn_crnn_v2.sh to use the correct key
sed -i'.bak' "s|SSH_KEY=\"~/.ssh/id_rsa\"|SSH_KEY=\"$ORIGINAL_KEY_PATH\"|g" deploy_attn_crnn_v2.sh

# Update the monitor_attn_crnn_v2.sh to use the correct key
sed -i'.bak' "s|SSH_KEY=\"~/.ssh/id_rsa\"|SSH_KEY=\"$ORIGINAL_KEY_PATH\"|g" monitor_attn_crnn_v2.sh

# Update the download_attn_crnn_v2_model.sh to use the correct key
sed -i'.bak' "s|SSH_KEY=\"~/.ssh/id_rsa\"|SSH_KEY=\"$ORIGINAL_KEY_PATH\"|g" download_attn_crnn_v2_model.sh

# Update the verify_wav2vec_features_detection.sh to use the correct key
sed -i'.bak' "s|SSH_KEY=\"~/.ssh/id_rsa\"|SSH_KEY=\"$ORIGINAL_KEY_PATH\"|g" verify_wav2vec_features_detection.sh

echo "Updated SSH key paths in all scripts to use: $ORIGINAL_KEY_PATH"
echo "You can now run improved_wav2vec_data_setup.sh again."
