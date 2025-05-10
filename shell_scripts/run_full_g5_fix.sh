#!/usr/bin/env bash
# One-command solution to fix G5 training and ensure real data is being used
# This script ties together verification, fix application, and monitoring

# Set constants
S3_BUCKET="emotion-recognition-data-324037291814"
MONITOR_INTERVAL=15  # Seconds between monitoring updates

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========== G5 TRAINING FIX: COMPLETE SOLUTION ==========${NC}"
echo -e "${BLUE}=========== $(date) ===========${NC}"

# Check script permissions
echo -e "\n${YELLOW}[1/4] Checking script permissions...${NC}"
if [ ! -x "./verify_s3_assets.sh" ] || [ ! -x "./fix_and_restart_g5_training.sh" ] || [ ! -x "./continuous_g5_monitor.sh" ]; then
    echo -e "Making scripts executable..."
    chmod +x verify_s3_assets.sh fix_and_restart_g5_training.sh continuous_g5_monitor.sh download_g5_model.sh setup_tensorboard_tunnel.sh
fi
echo -e "${GREEN}✓ All scripts are executable${NC}"

# Step 1: Verify S3 assets
echo -e "\n${YELLOW}[2/4] Verifying required assets in S3...${NC}"
./verify_s3_assets.sh
if [ $? -ne 0 ]; then
    echo -e "\n${RED}Error: S3 asset verification failed.${NC}"
    echo -e "Please upload the required feature archives to S3 bucket '$S3_BUCKET' first:"
    echo -e "${BLUE}aws s3 cp ravdess_features_facenet.tar.gz s3://$S3_BUCKET/${NC}"
    echo -e "${BLUE}aws s3 cp crema_d_features_facenet.tar.gz s3://$S3_BUCKET/${NC}"
    exit 1
fi

# Step 2: Apply the fix
echo -e "\n${YELLOW}[3/4] Applying G5 training fix...${NC}"
./fix_and_restart_g5_training.sh
if [ $? -ne 0 ]; then
    echo -e "\n${RED}Error: Failed to apply G5 training fix.${NC}"
    echo -e "Please check the error messages above and resolve any issues."
    exit 1
fi

# Step 3: Start monitoring
echo -e "\n${YELLOW}[4/4] Starting continuous monitoring...${NC}"
echo -e "Training should now be using real data with GPU utilization in the 60-90% range."
echo -e "Monitoring will update every $MONITOR_INTERVAL seconds."
echo -e "Press Ctrl+C to stop monitoring at any time.\n"
echo -e "${BLUE}TIP: To set up TensorBoard visualization in another terminal:${NC}"
echo -e "${BLUE}./setup_tensorboard_tunnel.sh${NC}\n"
echo -e "${BLUE}TIP: After training completes (≈8-10 hours), download the model:${NC}"
echo -e "${BLUE}./download_g5_model.sh${NC}\n"

# Give user time to read instructions
sleep 3

# Start monitoring
./continuous_g5_monitor.sh $MONITOR_INTERVAL
