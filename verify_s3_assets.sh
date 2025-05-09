#!/usr/bin/env bash
# Script to verify the presence of required feature archives in S3
# Run this before running fix_and_restart_g5_training.sh

# Set constants
S3_BUCKET="emotion-recognition-data-324037291814"
RAVDESS_ARCHIVE="ravdess_features_facenet.tar.gz"
CREMAD_ARCHIVE="crema_d_features_facenet.tar.gz"

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== S3 Assets Verification - $(date) =====${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    echo "Installation instructions: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check AWS CLI configuration
echo "Checking AWS CLI configuration..."
aws sts get-caller-identity &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: AWS CLI is not properly configured. Please configure it with valid credentials.${NC}"
    echo "Run: aws configure"
    exit 1
fi
echo -e "${GREEN}AWS CLI is properly configured.${NC}"

# Check S3 bucket existence
echo "Checking S3 bucket existence..."
aws s3 ls "s3://$S3_BUCKET" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: S3 bucket '$S3_BUCKET' does not exist or you don't have permission to access it.${NC}"
    echo "Update the S3_BUCKET variable in this script and in fix_and_restart_g5_training.sh"
    exit 1
fi
echo -e "${GREEN}S3 bucket '$S3_BUCKET' exists and is accessible.${NC}"

# Check RAVDESS archive
echo "Checking RAVDESS feature archive..."
aws s3 ls "s3://$S3_BUCKET/$RAVDESS_ARCHIVE" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: RAVDESS feature archive '$RAVDESS_ARCHIVE' not found in S3 bucket.${NC}"
    echo -e "${YELLOW}You need to upload it first:${NC}"
    echo -e "${BLUE}aws s3 cp $RAVDESS_ARCHIVE s3://$S3_BUCKET/${NC}"
    RAVDESS_MISSING=true
else
    echo -e "${GREEN}RAVDESS feature archive found in S3 bucket.${NC}"
    # Get file size
    RAVDESS_SIZE=$(aws s3api head-object --bucket $S3_BUCKET --key $RAVDESS_ARCHIVE --query 'ContentLength' --output text)
    RAVDESS_SIZE_MB=$(echo "scale=2; $RAVDESS_SIZE / 1048576" | bc)
    echo -e "Size: ${YELLOW}$RAVDESS_SIZE_MB MB${NC}"
    
    # Verify size is reasonable (at least 100MB)
    if (( $(echo "$RAVDESS_SIZE_MB < 100" | bc -l) )); then
        echo -e "${RED}Warning: RAVDESS archive is smaller than expected (should be ~1.6 GB).${NC}"
        echo -e "${YELLOW}This might be an incomplete or corrupted archive.${NC}"
    fi
fi

# Check CREMA-D archive
echo "Checking CREMA-D feature archive..."
aws s3 ls "s3://$S3_BUCKET/$CREMAD_ARCHIVE" &> /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CREMA-D feature archive '$CREMAD_ARCHIVE' not found in S3 bucket.${NC}"
    echo -e "${YELLOW}You need to upload it first:${NC}"
    echo -e "${BLUE}aws s3 cp $CREMAD_ARCHIVE s3://$S3_BUCKET/${NC}"
    CREMAD_MISSING=true
else
    echo -e "${GREEN}CREMA-D feature archive found in S3 bucket.${NC}"
    # Get file size
    CREMAD_SIZE=$(aws s3api head-object --bucket $S3_BUCKET --key $CREMAD_ARCHIVE --query 'ContentLength' --output text)
    CREMAD_SIZE_MB=$(echo "scale=2; $CREMAD_SIZE / 1048576" | bc)
    echo -e "Size: ${YELLOW}$CREMAD_SIZE_MB MB${NC}"
    
    # Verify size is reasonable (at least 100MB)
    if (( $(echo "$CREMAD_SIZE_MB < 100" | bc -l) )); then
        echo -e "${RED}Warning: CREMA-D archive is smaller than expected (should be ~1 GB).${NC}"
        echo -e "${YELLOW}This might be an incomplete or corrupted archive.${NC}"
    fi
fi

# Summary
echo -e "\n${BLUE}===== Verification Summary =====${NC}"
if [ "$RAVDESS_MISSING" == "true" ] || [ "$CREMAD_MISSING" == "true" ]; then
    echo -e "${RED}One or more required archives are missing in S3.${NC}"
    echo -e "${YELLOW}You must upload the missing archives before running fix_and_restart_g5_training.sh${NC}"
    
    echo -e "\n${BLUE}Upload commands:${NC}"
    if [ "$RAVDESS_MISSING" == "true" ]; then
        echo -e "${BLUE}aws s3 cp $RAVDESS_ARCHIVE s3://$S3_BUCKET/${NC}"
    fi
    if [ "$CREMAD_MISSING" == "true" ]; then
        echo -e "${BLUE}aws s3 cp $CREMAD_ARCHIVE s3://$S3_BUCKET/${NC}"
    fi
    exit 1
else
    echo -e "${GREEN}All required archives are present in S3 bucket '$S3_BUCKET'.${NC}"
    echo -e "${GREEN}You can now run fix_and_restart_g5_training.sh to complete the fix.${NC}"
    echo -e "${BLUE}./fix_and_restart_g5_training.sh${NC}"
fi
