#!/usr/bin/env python3
"""
Script to test audio and video feature extraction on a single downsampled CREMA-D file.
This uses the multimodal_preprocess_fixed.py module to process the file.
"""

import os
import sys
import logging
import numpy as np
from multimodal_preprocess_fixed import process_video_for_single_segment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_single_extraction.log"),
        logging.StreamHandler()
    ]
)

def main():
    # Define paths
    test_video = "downsampled_videos/CREMA-D-audio-complete/1001_DFA_ANG_XX.flv"
    output_dir = "test_single_file_output"
    
    # Make sure the video file exists
    if not os.path.exists(test_video):
        test_video = "downsampled_videos/CREMA-D-test/1001_IWL_SAD_XX.flv"
        if not os.path.exists(test_video):
            logging.error(f"Test video file not found: {test_video}")
            print(f"The test video file does not exist at either path. Please check the path.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Processing test file: {test_video}")
    logging.info(f"Output directory: {output_dir}")
    
    # Process the single file
    try:
        output_file = process_video_for_single_segment(
            video_path=test_video,
            output_dir=output_dir,
            model_name="VGG-Face"  # Default model used in the script
        )
        
        if output_file:
            logging.info(f"Successfully processed file. Output saved to: {output_file}")
            
            # Load and print a summary of the extracted features
            data = np.load(output_file)
            logging.info(f"Feature extraction summary:")
            logging.info(f"Video features shape: {data['video_features'].shape}")
            logging.info(f"Audio features shape: {data['audio_features'].shape}")
            logging.info(f"Video timestamps shape: {data['video_timestamps'].shape}")
            logging.info(f"Audio timestamps shape: {data['audio_timestamps'].shape}")
            logging.info(f"Emotion label: {data['emotion_label']}")
            
            print(f"\nFeature extraction successful!")
            print(f"Video features: {data['video_features'].shape}")
            print(f"Audio features: {data['audio_features'].shape}")
            print(f"Output file: {output_file}")
        else:
            logging.error("Processing failed - no output file was generated")
            print("Processing failed. Check the logs for details.")
    
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
