#!/bin/bash
# Script to run the fixed CNN feature extraction process
# This script uses the fixed implementation that correctly handles spectrogram shapes

set -e # Exit on error

# Define directories
DATA_DIR="data"
CREMA_D_SPEC_DIR="${DATA_DIR}/crema_d_features_spectrogram"
RAVDESS_SPEC_DIR="${DATA_DIR}/ravdess_features_spectrogram" 
CREMA_D_CNN_DIR="${DATA_DIR}/crema_d_features_cnn_audio"
RAVDESS_CNN_DIR="${DATA_DIR}/ravdess_features_cnn_audio"
SCRIPTS_DIR="scripts"
LOG_DIR="logs"

# Ensure directories exist
mkdir -p "${CREMA_D_CNN_DIR}" "${RAVDESS_CNN_DIR}" "${LOG_DIR}"

echo "=== Starting CNN Feature Extraction with Fixed Implementation ==="
echo "Input directories:"
echo "  - CREMA-D: ${CREMA_D_SPEC_DIR}"
echo "  - RAVDESS: ${RAVDESS_SPEC_DIR}"
echo "Output directories:"
echo "  - CREMA-D: ${CREMA_D_CNN_DIR}"
echo "  - RAVDESS: ${RAVDESS_CNN_DIR}"

# Copy our fixed implementation to the scripts directory to not overwrite original
cp fixed_preprocess_cnn_audio_features.py "${SCRIPTS_DIR}/fixed_preprocess_cnn_audio_features.py"
chmod +x "${SCRIPTS_DIR}/fixed_preprocess_cnn_audio_features.py"

# Number of worker processes (adjust based on CPU cores)
WORKERS=$(nproc --all)
if [ $WORKERS -gt 8 ]; then
    # Limit to 8 workers max to avoid memory issues
    WORKERS=8
fi
echo "Using ${WORKERS} worker processes"

# Run the extraction
echo "Running extraction..."
python3 "${SCRIPTS_DIR}/fixed_preprocess_cnn_audio_features.py" \
    --crema_d_dir "${CREMA_D_SPEC_DIR}" \
    --ravdess_dir "${RAVDESS_SPEC_DIR}" \
    --crema_d_output "${CREMA_D_CNN_DIR}" \
    --ravdess_output "${RAVDESS_CNN_DIR}" \
    --workers "${WORKERS}" \
    --verbose

# Verify results
CREMA_D_FILES=$(find "${CREMA_D_CNN_DIR}" -type f -name "*.npy" | wc -l)
RAVDESS_FILES=$(find "${RAVDESS_CNN_DIR}" -type f -name "*.npy" | wc -l)
TOTAL_FILES=$((CREMA_D_FILES + RAVDESS_FILES))

echo ""
echo "=== Extraction Completed ==="
echo "Extracted features:"
echo "  - CREMA-D: ${CREMA_D_FILES} files"
echo "  - RAVDESS: ${RAVDESS_FILES} files"
echo "  - Total: ${TOTAL_FILES} files"
echo ""
echo "To check the results:"
echo "  ls -la ${CREMA_D_CNN_DIR} | head"
echo "  ls -la ${RAVDESS_CNN_DIR} | head"
echo ""
echo "To continue with training:"
echo "  ./run_audio_pooling_with_laughter.sh"
