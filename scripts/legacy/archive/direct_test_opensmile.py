#!/usr/bin/env python3
"""
Direct test of openSMILE with the command that's known to work.
"""

import os
import sys
import logging
import subprocess
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("direct_test_opensmile.log"),
        logging.StreamHandler()
    ]
)

def main():
    print("\n----- Direct openSMILE Test -----\n")
    
    # Use our test videos
    test_videos_dir = "test_videos"
    first_test_video = os.path.join(test_videos_dir, "1001_TEST_ANG_XX.mp4")
    
    # Extract the audio to a temporary directory
    temp_dir = "direct_test_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract audio using moviepy
    video = VideoFileClip(first_test_video)
    audio = video.audio
    audio_path = os.path.join(temp_dir, "test_audio.wav")
    audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    print(f"Extracted test audio to: {audio_path}")
    
    # Create output ARFF file path
    audio_basename = os.path.basename(audio_path)
    output_file = os.path.join(temp_dir, f"{os.path.splitext(audio_basename)[0]}_egemaps.arff")
    
    # Path to openSMILE executable and config file
    opensmile_path = "/Users/patrickgloria/conjunction-train/opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"
    config_file = "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
    
    # Run openSMILE with the EXACT command that's working in multimodal_preprocess.py
    command = [
        opensmile_path,
        "-C", config_file,
        "-I", audio_path,
        "-O", output_file,
        "-instname", audio_basename,
        "-loglevel", "1"
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"❌ Direct openSMILE execution failed with code {result.returncode}")
            return False
        else:
            print("✅ Direct openSMILE execution succeeded!")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"Output file created: {output_file}")
                print(f"Output file size: {file_size} bytes")
                
                # Print the first few lines of the file to understand its format
                print("\nFirst 10 lines of output file:")
                with open(output_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 10:
                            print(line.strip())
                        else:
                            break
                
                # Now try our ARFF parser
                from utils import load_arff_features
                features, timestamps = load_arff_features(temp_dir, frame_size=0.025, frame_step=0.01)
                
                if features.size > 0:
                    print(f"\nSuccessfully parsed ARFF file!")
                    print(f"Features shape: {features.shape}")
                    print(f"Feature dimension: {features.shape[1]}")
                    return True
                else:
                    print(f"\n❌ Failed to parse ARFF file!")
                    return False
            else:
                print(f"❌ Output file not created: {output_file}")
                return False
    except Exception as e:
        print(f"❌ Error executing openSMILE: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
