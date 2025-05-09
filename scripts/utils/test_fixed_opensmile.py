#!/usr/bin/env python3
"""
Test script to verify that openSMILE is working correctly with the fixed configuration.
"""

import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def test_opensmile_extraction():
    """Test openSMILE extraction with the fixed configuration."""
    # Create test directories
    os.makedirs("temp_test_opensmile", exist_ok=True)
    
    # Set paths - use the original config file in the openSMILE installation
    config_path = os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf")
    
    # Find the sample audio file
    sample_audio = None
    for path in [
        "temp_extracted_audio/01-01-06-02-02-02-18.wav",  # From error message
        "opensmile-3.0.2-macos-armv8/example-audio/opensmile.wav",  # Default openSMILE example
    ]:
        if os.path.exists(path):
            sample_audio = path
            break
    
    if not sample_audio:
        logging.error("No sample audio file found. Please provide a valid audio file.")
        return False
    
    # Prepare output path
    output_csv = os.path.join("temp_test_opensmile", "test_output.csv")
    
    # Find openSMILE executable
    opensmile_path = None
    for path in [
        os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
        os.path.join(os.getcwd(), "opensmile-3.0.2-macos-armv8/bin/SMILExtract"),
        "opensmile-3.0.2-macos-armv8/bin/SMILExtract"
    ]:
        if os.path.exists(path):
            opensmile_path = path
            break
    
    if not opensmile_path:
        logging.error("Could not find openSMILE executable.")
        return False
    
    # Build openSMILE command
    command = [
        opensmile_path,
        "-C", config_path,
        "inputfile", sample_audio,
        "csvoutput", output_csv,
        "instname", os.path.basename(sample_audio)
    ]
    
    logging.info(f"Running test openSMILE command: {' '.join(command)}")
    
    # Execute command
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("openSMILE extraction successful!")
            logging.info(f"Output saved to: {output_csv}")
            
            # Check if output file exists
            if os.path.exists(output_csv):
                file_size = os.path.getsize(output_csv)
                logging.info(f"Output file size: {file_size} bytes")
                return True
            else:
                logging.error("Output file was not created despite successful return code.")
                return False
        else:
            logging.error(f"openSMILE extraction failed with return code {result.returncode}")
            logging.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error executing openSMILE: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_opensmile_extraction()
    print(f"Test {'PASSED' if success else 'FAILED'}")
