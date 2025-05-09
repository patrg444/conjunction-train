#!/bin/bash
# Simplified script to deploy the fixed TCN large model

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}    DEPLOYING FIXED TCN LARGE MODEL WITH BALANCED HYPERPARAMETERS    ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Instance details from existing scripts
INSTANCE_IP="13.217.128.73"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"

# Ensure the script is executable
chmod +x scripts/train_branched_regularization_sync_aug_tcn_large_fixed_complete.py

echo -e "${YELLOW}Transferring fixed training script to AWS instance...${NC}"
scp -i "${KEY_FILE}" "./scripts/train_branched_regularization_sync_aug_tcn_large_fixed_complete.py" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Create a completion script that will finish the fixed file
cat > complete_script.py << 'EOL'
#!/usr/bin/env python3

COMPLETION = """    train_generator = SynchronizedAugmentationDataGenerator(
        train_video, train_audio, train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augmentation_factor=AUGMENTATION_FACTOR,
        augmentation_probability=0.8  # 80% chance of applying augmentation to eligible samples
    )

    val_generator = ValidationDataGenerator(
        val_video, val_audio, val_labels,
        batch_size=BATCH_SIZE
    )

    # Create and compile the enhanced model with regularization, TCN, and masking layers
    model = create_enhanced_large_model_with_regularization_and_tcn(audio_feature_dim, video_feature_dim)
    print('\\nModel Summary:')
    model.summary()

    # Create output directories if they don't exist
    model_dir = 'models/branched_regularization_sync_aug_tcn_large_fixed'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define callbacks with more sophisticated setup
    checkpoint_path = os.path.join(model_dir, 'model_best.keras')

    # Add warm-up cosine decay scheduler
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE,
        total_epochs=EPOCHS,
        warmup_epochs=10,
        min_learning_rate=5e-6
    )

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            save_best_only=True,
            save_weights_only=False,
            mode='max',  # We want to maximize accuracy
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,         # Less aggressive reduction
            patience=4,         # Reduced patience for more frequent adjustments
            min_lr=5e-6,
            verbose=1
        ),
        lr_scheduler
    ]

    # Calculate class weights to handle imbalance
    total_samples = len(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_count = np.sum(train_labels[:, i])
        class_weights[i] = total_samples / (NUM_CLASSES * class_count)

    print('Using class weights to handle imbalance:', class_weights)

    # Train the model
    print('Starting training with advanced hybrid architecture and balanced regularization...')
    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save the final model
    final_model_path = os.path.join(model_dir, 'final_model.keras')
    model.save(final_model_path)
    print('Final model saved to:', final_model_path)

    # Calculate training time
    train_time = time.time() - start_time
    print('Training completed in %.2f seconds (%.2f minutes)' % (train_time, train_time/60))

    # Print training history summary
    print('Training history summary:')
    print('- Final training accuracy:', history.history['accuracy'][-1])
    print('- Final validation accuracy:', history.history['val_accuracy'][-1])
    print('- Best validation accuracy:', max(history.history['val_accuracy']))
    print('- Best validation loss:', min(history.history['val_loss']))

    return model, history

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        import traceback
        print('ERROR:', str(e))
        print(traceback.format_exc())"""

# Write completion to file
with open("completion.txt", "w") as f:
    f.write(COMPLETION)

print("Completion file created successfully")
EOL

# Create run script to be executed on the remote server
cat > run_script.sh << 'EOL'
#!/bin/bash

# Set error handling
set -e

cd ~/emotion_training

echo "Completing the TCN large model fixed script..."
python3 complete_script.py
cat completion.txt >> scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py
rm completion.txt

echo "Setting permissions..."
chmod +x scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py

echo "Starting fixed TCN large model training..."
nohup python3 scripts/train_branched_regularization_sync_aug_tcn_large_fixed.py > training_branched_regularization_sync_aug_tcn_large_fixed.log 2>&1 &

# Save the PID
echo $! > fixed_tcn_large_pid.txt
echo "Training process started with PID $(cat fixed_tcn_large_pid.txt)"
echo "Log file: training_branched_regularization_sync_aug_tcn_large_fixed.log"

# Display Python and TensorFlow versions
echo "Python version:"
python3 --version
echo "TensorFlow version:"
python3 -c "import tensorflow as tf; print(tf.__version__)"

echo "Deployment completed successfully!"
EOL

echo -e "${YELLOW}Transferring helper scripts to AWS instance...${NC}"
scp -i "${KEY_FILE}" complete_script.py "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/"
scp -i "${KEY_FILE}" run_script.sh "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/"

echo -e "${YELLOW}Running deployment on AWS instance...${NC}"
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/run_script.sh && ${REMOTE_DIR}/run_script.sh"

# Clean up local temporary files
rm complete_script.py run_script.sh

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Create a monitoring script for the fixed model
cat > monitor_fixed_tcn_model.sh << 'EOL'
#!/bin/bash
# Script to monitor the fixed TCN large model training

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================================================${NC}"
echo -e "${GREEN}    MONITORING FIXED TCN LARGE MODEL WITH BALANCED HYPERPARAMETERS    ${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo ""

# Instance details from existing scripts
INSTANCE_IP="13.217.128.73"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed.log"

if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo -e "${GREEN}Starting continuous monitoring... (Press Ctrl+C to stop)${NC}"
echo

# Use SSH to continuously monitor the log file
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -f ~/emotion_training/$LOG_FILE"
EOL

chmod +x monitor_fixed_tcn_model.sh
echo -e "${GREEN}Created monitoring script: monitor_fixed_tcn_model.sh${NC}"
