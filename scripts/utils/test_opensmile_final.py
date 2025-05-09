#!/usr/bin/env python3
"""
Test script to verify that openSMILE is working correctly.
Uses command-line parameters to override configuration options.
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
    """Test openSMILE extraction using command-line parameters."""
    # Create test directories
    test_dir = "temp_test_opensmile"
    os.makedirs(test_dir, exist_ok=True)
    
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
    
    # Set paths
    main_config_path = "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
    output_csv = os.path.join(test_dir, "test_output.csv")
    
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
    
    # Build openSMILE command using the command-line parameter approach
    command = [
        opensmile_path,
        "-C", main_config_path,
        "inputfile", sample_audio,  # Mapped as inputfile in standard_wave_input.conf.inc
        "csvoutput", output_csv,    # Mapped as csvoutput in standard_data_output.conf.inc
        "instname", os.path.basename(sample_audio)  # Mapped as instname
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
                
                # Show the first few lines of the output file
                with open(output_csv, 'r') as f:
                    head = "".join(f.readlines()[:5])
                logging.info(f"Output file content (first few lines):\n{head}")
                
                return True
            else:
                logging.error("Output file was not created despite successful return code.")
                return False
        else:
            logging.error(f"openSMILE extraction failed with return code {result.returncode}")
            logging.error(f"Error output: {result.stderr}")
            logging.error(f"Standard output: {result.stdout}")
            return False
            
    except Exception as e:
        logging.error(f"Error executing openSMILE: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_opensmile_extraction()
    print(f"Test {'PASSED' if success else 'FAILED'}")
