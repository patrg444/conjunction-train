#!/usr/bin/env python3
"""
Test script to verify that openSMILE is working correctly with the fixed configuration.
Uses the correct configuration override approach.
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
    """Test openSMILE extraction with the correct configuration approach."""
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
    
    # Create a temporary config file that includes the standard config and overrides
    temp_config_path = os.path.join(test_dir, "temp_config.conf")
    main_config_path = "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
    output_csv = os.path.join(test_dir, "test_output.csv")
    
    # Write the custom config file
    with open(temp_config_path, 'w') as f:
        f.write(f"""// Include the standard eGeMAPS configuration
\\{{{main_config_path}}}

// Override the input and output file settings
[componentInstances:cComponentManager]

// Set the specific input file
instance[waveIn].configStr = \\cm[inputfile(I){{{sample_audio}}}:file name of the input wave file]

// Set an output file for the CSV output
instance[csvSink].configStr = \\cm[csvoutput(O){{{output_csv}}}:file name of the output CSV file]
""")
    
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
        "-C", temp_config_path
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
