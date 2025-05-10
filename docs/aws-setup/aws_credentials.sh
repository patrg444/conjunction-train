#!/bin/bash
# Store your AWS credentials in this file.
# IMPORTANT: Make sure this file is in your .gitignore and NEVER committed to version control with actual credentials.

export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID_HERE"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY_HERE"
export AWS_DEFAULT_REGION="us-east-1" # Or your preferred region

# You can also set other AWS CLI environment variables here if needed
# export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN_IF_USING_TEMPORARY_CREDENTIALS"

echo "AWS credentials sourced from aws_credentials.sh"
