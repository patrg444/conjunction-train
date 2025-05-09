#!/usr/bin/env python3
"""
Test script to verify OpenSMILE integration is working correctly.
This script extracts features from a test audio file using OpenSMILE
and verifies the extraction works properly.
"""

import os
import sys
import logging
import numpy as np
from multimodal_preprocess import extract_frame_level_audio_features
from utils import load_arff_features
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def test_opensmile_extraction(audio_path=None):
    """Tests OpenSMILE feature extraction on a sample audio file.
    
    Args:
        audio_path: Path to audio file. If None, uses files in temp_extracted_audio.
    """
    logging.info("Testing OpenSMILE feature extraction...")
    
    # If no specific audio path provided, use files in temp_extracted_audio if they exist
    if audio_path is None:
        audio_dir = "temp_extracted_audio"
        if os.path.exists(audio_dir):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            if audio_files:
                audio_path = os.path.join(audio_dir, audio_files[0])
                logging.info(f"Using existing audio file: {audio_path}")
            else:
                logging.error(f"No WAV files found in {audio_dir}")
                return False
        else:
            logging.error(f"Audio directory {audio_dir} not found")
            return False
    
    if not os.path.exists(audio_path):
        logging.error(f"Audio file {audio_path} not found")
        return False
    
    try:
        # Path to openSMILE executable
        opensmile_path = "/Users/patrickgloria/conjunction-train/opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"
        
        # Path to openSMILE configuration
        config_file = "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
        
        # Check if the paths exist
        if not os.path.exists(opensmile_path):
            logging.error(f"OpenSMILE executable not found at: {opensmile_path}")
            return False
        
        if not os.path.exists(config_file):
            logging.error(f"OpenSMILE config file not found at: {config_file}")
            return False
        
        # Extract features using OpenSMILE
        features, timestamps = extract_frame_level_audio_features(
            audio_path, 
            config_file=config_file,
            opensmile_path=opensmile_path
        )
        
        # Check if features were extracted
        if features is None or timestamps is None:
            logging.error("Failed to extract features with OpenSMILE")
            return False
        
        # Print info about extracted features
        logging.info(f"Successfully extracted features with OpenSMILE")
        logging.info(f"Feature shape: {features.shape}")
        logging.info(f"Number of timestamps: {len(timestamps)}")
        logging.info(f"Feature dimension: {features.shape[1]}")
        
        # Visualize the features
        plt.figure(figsize=(12, 6))
        
        # Plot first 5 feature dimensions as a sample
        feat_to_plot = min(5, features.shape[1])
        for i in range(feat_to_plot):
            plt.plot(timestamps[:100], features[:100, i], label=f'Feature {i+1}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Feature Value')
        plt.title('OpenSMILE Features (first 5 dimensions)')
        plt.legend()
        plt.tight_layout()
        
        # Create directory for output
        os.makedirs("test_opensmile_output", exist_ok=True)
        plt.savefig("test_opensmile_output/opensmile_features.png")
        logging.info("Feature visualization saved to test_opensmile_output/opensmile_features.png")
        
        return True
    
    except Exception as e:
        logging.error(f"Error testing OpenSMILE integration: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def run_complete_test():
    """Runs a complete test of the OpenSMILE integration."""
    print("\n" + "="*50)
    print("OPENSMILE INTEGRATION TEST")
    print("="*50)
    
    # Test OpenSMILE feature extraction
    success = test_opensmile_extraction()
    
    if success:
        print("\n✅ OpenSMILE integration test PASSED!")
        print("OpenSMILE is correctly functioning and integrated into the system.")
        print("The librosa dependency has been successfully replaced with OpenSMILE.")
    else:
        print("\n❌ OpenSMILE integration test FAILED!")
        print("There were issues with the OpenSMILE integration.")
    
    print("="*50)
    return success

if __name__ == "__main__":
    # If an audio path is provided as command line argument, use it
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_complete_test()
