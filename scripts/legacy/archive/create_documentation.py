#!/usr/bin/env python3
"""
Enhanced script to generate documentation for key files in the project.
Includes metadata, descriptions, and relationships between files.
"""

import os
import datetime

# Key files with metadata
key_files_metadata = [
    {
        "path": "scripts/process_ravdess_dataset.py",
        "description": "Processes videos from the RAVDESS emotion dataset to extract audio and video features using OpenSMILE.",
        "dependencies": ["multimodal_preprocess.py", "synchronize_test.py"],
        "inputs": "Raw RAVDESS dataset videos",
        "outputs": "Processed feature files (.npz) with video and audio sequences"
    },
    {
        "path": "scripts/multimodal_preprocess.py",
        "description": "Core preprocessing library for extracting audio and video features from video files.",
        "dependencies": ["utils.py"],
        "inputs": "Video files (.mp4)",
        "outputs": "Synchronized audio-visual features"
    },
    {
        "path": "scripts/train_branched.py",
        "description": "Trains the branched LSTM model for emotion recognition using extracted features.",
        "dependencies": ["synchronize_test.py"],
        "inputs": "Processed feature files from process_ravdess_dataset.py",
        "outputs": "Trained model (.h5) and evaluation metrics"
    },
    {
        "path": "scripts/analyze_dataset.py",
        "description": "Analyzes the processed dataset to get statistics about emotions, actors, and sequence counts.",
        "dependencies": [],
        "inputs": "Processed feature files (.npz)",
        "outputs": "Statistics and visualizations of the dataset"
    },
    {
        "path": "scripts/train_dual_stream.py",
        "description": "Alternative model architecture using dual-stream approach for emotion recognition.",
        "dependencies": ["synchronize_test.py"],
        "inputs": "Processed feature files from process_ravdess_dataset.py",
        "outputs": "Trained dual-stream model (.h5) and evaluation metrics"
    }
]

# Output file path
output_file = "enhanced_documentation.txt"

def create_enhanced_documentation():
    """Create enhanced documentation with metadata and relationships."""
    with open(output_file, 'w') as outfile:
        # Write header
        outfile.write("ENHANCED PROJECT DOCUMENTATION\n")
        outfile.write("============================\n\n")
        outfile.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write table of contents
        outfile.write("TABLE OF CONTENTS\n")
        outfile.write("-----------------\n")
        for i, metadata in enumerate(key_files_metadata, 1):
            outfile.write(f"{i}. {os.path.basename(metadata['path'])} - {metadata['description'][:60]}...\n")
        outfile.write("\n")
        
        # Write project overview
        outfile.write("PROJECT OVERVIEW\n")
        outfile.write("---------------\n")
        outfile.write("This project implements a multimodal emotion recognition system using audio and video features\n")
        outfile.write("extracted from the RAVDESS dataset. The system uses deep learning models with either branched\n")
        outfile.write("LSTM or dual-stream architectures to classify emotions from synchronized audio-visual data.\n\n")
        
        # Write workflow diagram (ASCII)
        outfile.write("WORKFLOW\n")
        outfile.write("--------\n")
        outfile.write("  [RAVDESS Videos]\n")
        outfile.write("        |\n")
        outfile.write("        v\n")
        outfile.write("[process_ravdess_dataset.py] --> [multimodal_preprocess.py]\n")
        outfile.write("        |\n")
        outfile.write("        v\n")
        outfile.write("  [Processed Features] --> [analyze_dataset.py]\n")
        outfile.write("        |                         |\n")
        outfile.write("        |                         v\n")
        outfile.write("        |                  [Dataset Statistics]\n")
        outfile.write("        |\n")
        outfile.write("        |--> [train_branched.py]\n")
        outfile.write("        |         |\n")
        outfile.write("        |         v\n")
        outfile.write("        |    [Branched Model]\n")
        outfile.write("        |\n")
        outfile.write("        +--> [train_dual_stream.py]\n")
        outfile.write("                  |\n")
        outfile.write("                  v\n")
        outfile.write("             [Dual-Stream Model]\n\n")
        
        # Write detailed file documentation
        outfile.write("DETAILED FILE DOCUMENTATION\n")
        outfile.write("==========================\n\n")
        
        for metadata in key_files_metadata:
            file_path = metadata["path"]
            outfile.write(f"FILE: {file_path}\n")
            outfile.write("=" * (len(file_path) + 6) + "\n\n")
            
            # Write metadata
            outfile.write("METADATA:\n")
            outfile.write(f"Description: {metadata['description']}\n")
            outfile.write(f"Dependencies: {', '.join(metadata['dependencies'])}\n")
            outfile.write(f"Inputs: {metadata['inputs']}\n")
            outfile.write(f"Outputs: {metadata['outputs']}\n\n")
            
            # Write file content
            outfile.write("CONTENT:\n")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as infile:
                        content = infile.read()
                        outfile.write(content)
                except Exception as e:
                    outfile.write(f"ERROR reading file: {str(e)}\n")
            else:
                outfile.write(f"ERROR: File not found: {file_path}\n")
            
            outfile.write("\n\n" + "=" * 80 + "\n\n")
        
        outfile.write("\nEND OF DOCUMENTATION\n")
    
    print(f"Enhanced documentation created: {output_file}")

if __name__ == "__main__":
    create_enhanced_documentation()
