#!/usr/bin/env python3
"""
Script to copy the contents of key Python scripts to a single text file for documentation.
"""

import os

# List of key script files to copy
key_files = [
    "scripts/process_ravdess_dataset.py",
    "scripts/multimodal_preprocess.py",
    "scripts/train_branched.py", 
    "scripts/analyze_dataset.py",
    "scripts/train_dual_stream.py"
]

# Output file path
output_file = "key_scripts_documentation.txt"

def copy_scripts_to_txt():
    """Copy the content of key scripts to a single text file."""
    with open(output_file, 'w') as outfile:
        outfile.write("KEY SCRIPTS DOCUMENTATION\n")
        outfile.write("=========================\n\n")
        
        for file_path in key_files:
            if not os.path.exists(file_path):
                outfile.write(f"ERROR: File not found: {file_path}\n\n")
                continue
                
            outfile.write(f"FILE: {file_path}\n")
            outfile.write("=" * (len(file_path) + 6) + "\n\n")
            
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"ERROR reading file: {str(e)}\n")
            
            outfile.write("\n\n" + "=" * 80 + "\n\n")
        
        outfile.write("\nEND OF DOCUMENTATION\n")
    
    print(f"Documentation created: {output_file}")

if __name__ == "__main__":
    copy_scripts_to_txt()
