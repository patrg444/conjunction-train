#!/bin/bash

# Source AWS credentials
if [ -f ./aws-setup/aws_credentials.sh ]; then
  source ./aws-setup/aws_credentials.sh
else
  echo "Error: aws_credentials.sh not found in aws-setup directory."
  echo "Please create it and add your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
  exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI could not be found, please install it first."
    exit 1
fi

# Configure AWS CLI with sourced credentials
# Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set from the sourced file
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Error: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set."
  echo "Please ensure they are correctly defined in aws-setup/aws_credentials.sh."
  exit 1
fi

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
# You might want to set default region and output format as well
# aws configure set default.region us-east-1
# aws configure set default.output json

echo "AWS CLI has been configured using credentials from aws-setup/aws_credentials.sh."
echo "Note: These credentials are set for the current AWS CLI profile."
echo "If you use named profiles, you might need to adjust the 'aws configure set' commands accordingly (e.g., --profile yourprofilename)."
