#!/bin/bash
# This script simulates running the complete wav2vec training workflow
# It shows the expected output when running on a real AWS instance

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}${BOLD}     WAV2VEC AUDIO EMOTION TRAINING WORKFLOW SIMULATION     ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo

echo -e "${CYAN}█ STEP 1: Run the optimized training script${NC}"
echo -e "${YELLOW}Running: ./run_wav2vec_audio_only_optimized.sh${NC}"

echo -e "\nStarting optimized audio-only wav2vec emotion recognition training..."
echo "Uploading scripts to EC2..."
echo "Running data preparation script..."
echo "Stopping any existing training processes..."
echo "Starting optimized audio-only training..."
echo -e "${GREEN}✓ Optimized audio-only training started!${NC}"
echo "Monitor training with:"
echo "./monitor_wav2vec_training.sh ~/Downloads/gpu-key.pem 54.162.134.77"
echo
echo "Set up TensorBoard with:"
echo "./setup_tensorboard_tunnel.sh ~/Downloads/gpu-key.pem 54.162.134.77"
echo
echo "When training completes, download the model with:"
echo "./download_wav2vec_model.sh ~/Downloads/gpu-key.pem 54.162.134.77"

sleep 3
echo
echo -e "${CYAN}█ STEP 2: Monitor training progress${NC}"
echo -e "${YELLOW}Running: ./monitor_wav2vec_training.sh ~/Downloads/gpu-key.pem 54.162.134.77${NC}"

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}          WAV2VEC AUDIO EMOTION TRAINING MONITOR           ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}54.162.134.77${NC}"
echo -e "Remote path: ${CYAN}/home/ubuntu/audio_emotion/train_wav2vec_audio_only.log${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}Checking if wav2vec training is running...${NC}"
echo -e "${GREEN}✅ Training process active (PID: 12345)${NC}"

echo -e "\n${YELLOW}Checking GPU utilization...${NC}"
echo -e "GPU Utilization: ${GREEN}87%${NC} (good)"
echo -e "Memory Usage: ${GREEN}7568 MB / 16384 MB${NC} (46%)"
echo -e "GPU Temperature: ${GREEN}68°C${NC}"

echo -e "\n${YELLOW}Checking if TensorBoard is running...${NC}"
echo -e "${YELLOW}TensorBoard not running.${NC}"
echo -e "You can start TensorBoard with: ./setup_tensorboard_tunnel.sh ~/Downloads/gpu-key.pem 54.162.134.77"

echo -e "\n${YELLOW}Extracting training progress...${NC}"
echo -e "\n${BLUE}Training Progress:${NC}"
echo -e "Epoch 12/200: 231/231 [==============================] - 45s 196ms/step - loss: 0.8954 - accuracy: 0.6824"

echo -e "\n${BLUE}Recent Metrics:${NC}"
echo -e "${GREEN}Validation Accuracy:${NC}"
echo -e "val_accuracy: 0.6731"
echo -e "val_accuracy: 0.6842"
echo -e "val_accuracy: 0.6953"

echo -e "\n${BLUE}Checking for Errors:${NC}"
echo -e "${GREEN}No errors detected in log.${NC}"

echo -e "\n${BLUE}Live Log Stream (press Ctrl+C to exit):${NC}"
echo "Epoch 12/200: 98/231 [===========>..................] - ETA: 26s - loss: 0.8884 - accuracy: 0.6877"
sleep 1
echo "Epoch 12/200: 99/231 [===========>..................] - ETA: 26s - loss: 0.8883 - accuracy: 0.6879"
sleep 1
echo "Epoch 12/200: 100/231 [===========>..................] - ETA: 25s - loss: 0.8881 - accuracy: 0.6880"

sleep 3
echo
echo -e "${CYAN}█ STEP 3: Set up TensorBoard for visualization${NC}"
echo -e "${YELLOW}Running: ./setup_tensorboard_tunnel.sh ~/Downloads/gpu-key.pem 54.162.134.77${NC}"

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}          TENSORBOARD SETUP FOR WAV2VEC TRAINING           ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}54.162.134.77${NC}"
echo -e "TensorBoard port: ${CYAN}6006${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}Checking if TensorBoard is already running...${NC}"
echo -e "${YELLOW}TensorBoard not running. Starting it now...${NC}"
echo -e "${GREEN}✅ TensorBoard started successfully (PID: 23456)${NC}"

echo -e "\n${YELLOW}Setting up SSH tunnel for TensorBoard...${NC}"
echo -e "${GREEN}Run the following command in a new terminal to create the tunnel:${NC}"
echo -e "${CYAN}ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77${NC}"

echo -e "\n${YELLOW}Once the tunnel is established, open TensorBoard in your browser:${NC}"
echo -e "${CYAN}http://localhost:6006${NC}"

echo -e "\n${YELLOW}To check TensorBoard logs if there are issues:${NC}"
echo -e "${CYAN}ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \"cat /home/ubuntu/audio_emotion/tensorboard.log\"${NC}"

sleep 3
echo
echo -e "${CYAN}█ STEP 4: Download the trained model and history files${NC}"
echo -e "${YELLOW}Running: ./download_wav2vec_model.sh ~/Downloads/gpu-key.pem 54.162.134.77${NC}"

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}          DOWNLOAD WAV2VEC EMOTION RECOGNITION MODEL        ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "EC2 Instance: ${CYAN}54.162.134.77${NC}"
echo -e "Remote path: ${CYAN}/home/ubuntu/audio_emotion/checkpoints/${NC}"
echo -e "Local download directory: ${CYAN}wav2vec_models${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}Creating local directory for downloads...${NC}"
echo -e "\n${YELLOW}Checking if training is still in progress...${NC}"
echo -e "${GREEN}✅ Training is not running. Proceeding with download.${NC}"

echo -e "\n${YELLOW}Checking available model files...${NC}"
echo -e "${GREEN}Found model files:${NC}"
echo -e "/home/ubuntu/audio_emotion/checkpoints/wav2vec_audio_only_20250422_033015_best.h5"
echo -e "/home/ubuntu/audio_emotion/checkpoints/wav2vec_audio_only_20250422_033015_final.h5"
echo -e "/home/ubuntu/audio_emotion/checkpoints/wav2vec_audio_only_20250422_033015_architecture.json"
echo -e "/home/ubuntu/audio_emotion/checkpoints/wav2vec_audio_only_20250422_033015_history.json"

echo -e "\n${YELLOW}Downloading best model and history files...${NC}"

echo -e "\n${YELLOW}Starting download...${NC}"
echo -e "${CYAN}Downloading wav2vec_audio_only_20250422_033015_best.h5...${NC}"
echo -e "${CYAN}Downloading wav2vec_audio_only_20250422_033015_history.json...${NC}"
echo -e "${CYAN}Downloading wav2vec_audio_only_20250422_033015_architecture.json...${NC}"

echo -e "\n${GREEN}✅ Models downloaded successfully to wav2vec_models${NC}"

echo -e "\n${YELLOW}Suggested next steps:${NC}"
echo -e "1. Plot training curves with:"
echo -e "${CYAN}   python scripts/plot_training_curve.py --history_file wav2vec_models/wav2vec_audio_only_20250422_033015_history.json --metric both${NC}"

echo -e "2. For model inference, you can load the model with:"
echo -e "${CYAN}   from tensorflow.keras.models import load_model${NC}"
echo -e "${CYAN}   model = load_model('wav2vec_models/wav2vec_audio_only_20250422_033015_best.h5')${NC}"

echo -e "3. Consider exporting to SavedModel format for deployment:"
echo -e "${CYAN}   model.save('wav2vec_audio_only_saved', include_optimizer=False)${NC}"

echo
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}${BOLD}                WORKFLOW COMPLETED SUCCESSFULLY               ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "The wav2vec audio emotion recognition model has been successfully:"
echo -e "  ✓ Trained with mixed precision and optimized hyperparameters"
echo -e "  ✓ Monitored during training with GPU utilization tracking"
echo -e "  ✓ Visualized with TensorBoard"
echo -e "  ✓ Downloaded for evaluation and deployment"
echo
echo -e "For more details, see ${BOLD}WAV2VEC_TRAINING_TOOLKIT_README.md${NC}"
