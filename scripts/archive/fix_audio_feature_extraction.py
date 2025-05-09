#!/usr/bin/env python3
"""
Test script to verify eGeMAPSv02 feature extraction with functionals.
"""

import os
import subprocess
import pandas as pd
import numpy as np

# Create temporary output directory if it doesn't exist
temp_dir = "temp_test_opensmile"
os.makedirs(temp_dir, exist_ok=True)

# Find a sample audio file
wav_files = []
for root, dirs, files in os.walk("temp_extracted_audio"):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))
            break
    if wav_files:
        break

if not wav_files:
    print("Error: No WAV files found in the temp_extracted_audio directory.")
    exit(1)

audio_path = wav_files[0]
print(f"Using sample audio file: {audio_path}")

# Test with LLD CSV output (which is what our current pipeline uses)
lld_output = os.path.join(temp_dir, "test_lld.csv")
# Test with functionals (what we actually want)
func_output = os.path.join(temp_dir, "test_func.csv")

# Run openSMILE with LLDs (current implementation)
lld_command = [
    "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract",
    "-C", "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf",
    "-I", audio_path,
    "-lldcsvoutput", lld_output
]

# Run openSMILE with functionals (what we want)
func_command = [
    "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract",
    "-C", "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02_func.conf",
    "-I", audio_path, 
    "-csvoutput", func_output
]

print("Running openSMILE with LLDs (current implementation)...")
subprocess.run(lld_command)

print("Running openSMILE with functionals (what we want)...")
subprocess.run(func_command)

# Analyze the results
if os.path.exists(lld_output):
    lld_data = pd.read_csv(lld_output, sep=';')
    print(f"LLD output has {lld_data.shape[1]-2} features (excluding name, frameTime)")
    print(f"LLD sample: {list(lld_data.columns[:10])}")
else:
    print("Failed to generate LLD output")

if os.path.exists(func_output):
    func_data = pd.read_csv(func_output, sep=';')
    print(f"Functionals output has {func_data.shape[1]-1} features (excluding name)")
    print(f"Functionals sample: {list(func_data.columns[:10])}")
else:
    print("Failed to generate functionals output")

print("\nBased on these results, we need to modify the extract_frame_level_audio_features function to use eGeMAPSv02_func.conf instead of eGeMAPSv02.conf")
