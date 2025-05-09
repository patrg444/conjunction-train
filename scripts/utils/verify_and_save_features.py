#!/usr/bin/env python3
"""
Advanced verification script that explicitly saves extracted features to files
for detailed examination. This script captures both FaceNet facial embeddings
and OpenSMILE audio features, saving them to data files for analysis.
"""

import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import logging
import subprocess
import pyaudio
import wave
import json
import tensorflow as tf
import argparse
from datetime import datetime
from facenet_extractor import FaceNetExtractor

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("features_verification_detailed.log"),
        logging.StreamHandler()
    ]
)

def save_features_to_file(features, filename):
    """Save extracted features to a NumPy file."""
    np.savez(filename, features=features)
    logging.info(f"Features saved to {filename}")

def save_metadata(metadata, filename):
    """Save metadata to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Metadata saved to {filename}")

def extract_and_save_facenet_features():
    """Extract facial features from webcam frames and save them."""
    logging.info("=== EXTRACTING AND SAVING FACENET FEATURES ===")
    
    # Create output directory
    output_dir = "feature_verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        return None
    
    # Initialize FaceNetExtractor
    face_extractor = FaceNetExtractor(
        keep_all=False,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7]
    )
    
    # Capture multiple frames to increase chance of face detection
    max_frames = 10
    frames = []
    faces_detected = []
    features_list = []
    
    try:
        for i in range(max_frames):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {i+1}")
                continue
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Extract features
            features = face_extractor.extract_features(frame)
            
            # Check if face detected
            has_face = np.any(features != 0)
            faces_detected.append(has_face)
            
            if has_face:
                # Save frame with detected face
                frame_with_face_path = os.path.join(output_dir, f"frame_with_face_{i+1}.jpg")
                cv2.imwrite(frame_with_face_path, frame)
                
                # Add to features list
                features_list.append(features)
                
                # Log statistics
                logging.info(f"Frame {i+1}: Face detected!")
                logging.info(f"Feature shape: {features.shape}")
                logging.info(f"Non-zero elements: {np.count_nonzero(features)}/{features.size}")
                logging.info(f"Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")
                logging.info(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
                
                # Save features to file
                features_path = os.path.join(output_dir, f"facenet_features_{i+1}.npz")
                save_features_to_file(features, features_path)
            else:
                logging.warning(f"Frame {i+1}: No face detected")
            
            # Wait a bit between frames
            time.sleep(0.2)
    
    finally:
        cap.release()
    
    # Save summary
    success = any(faces_detected)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "frames_captured": len(frames),
        "faces_detected": sum(faces_detected),
        "feature_dimension": 512,
        "success": success
    }
    
    metadata_path = os.path.join(output_dir, "facenet_extraction_metadata.json")
    save_metadata(metadata, metadata_path)
    
    if success:
        # Return the first successful features
        return next((f for i, f in enumerate(features_list) if faces_detected[i]), None)
    else:
        logging.error("No faces detected in any frame")
        return None

def extract_and_save_opensmile_features(opensmile_path, config_file):
    """Extract audio features with OpenSMILE and save them."""
    logging.info("=== EXTRACTING AND SAVING OPENSMILE FEATURES ===")
    
    # Create output directory
    output_dir = "feature_verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(opensmile_path):
        logging.error(f"OpenSMILE executable not found at {opensmile_path}")
        return None
    
    if not os.path.exists(config_file):
        logging.error(f"OpenSMILE config not found at {config_file}")
        return None
    
    # Create temporary directory for audio files
    temp_dir = os.path.join(output_dir, "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = int(time.time())
    wav_path = os.path.join(temp_dir, f"audio_sample_{timestamp}.wav")
    csv_path = os.path.join(temp_dir, f"features_{timestamp}.csv")
    
    try:
        # Record audio
        logging.info("Recording audio sample...")
        p = pyaudio.PyAudio()
        sample_rate = 16000
        channels = 1
        chunk_size = 1024
        format = pyaudio.paInt16
        record_seconds = 3
        
        # Find available input device
        device_info = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                device_info.append(info)
                logging.info(f"Found input device {i}: {info['name']}")
        
        if not device_info:
            logging.error("No audio input devices found")
            
            # Create synthetic audio for testing
            logging.info("Creating synthetic test audio...")
            t = np.linspace(0, record_seconds, int(sample_rate * record_seconds))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
        else:
            # Use first available input device
            device_idx = int(device_info[0]['index'])
            logging.info(f"Using input device {device_idx}: {device_info[0]['name']}")
            
            # Open audio stream
            stream = p.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=chunk_size
            )
            
            # Record audio
            frames = []
            logging.info(f"Recording {record_seconds} seconds of audio...")
            for i in range(0, int(sample_rate / chunk_size * record_seconds)):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
            # Save audio to WAV file
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(format))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
        
        p.terminate()
        logging.info(f"Audio saved to {wav_path}")
        
        # Process with OpenSMILE
        logging.info(f"Processing audio with OpenSMILE...")
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", wav_path,
            "-csvoutput", csv_path,
            "-instname", "verification"
        ]
        
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            logging.error(f"OpenSMILE error: {proc.stderr}")
            return None
        
        # Read features from CSV
        if not os.path.exists(csv_path):
            logging.error(f"OpenSMILE output file not created: {csv_path}")
            return None
        
        # Copy the CSV to output directory
        features_file = os.path.join(output_dir, "opensmile_features.csv")
        with open(csv_path, 'r') as src, open(features_file, 'w') as dst:
            dst.write(src.read())
        
        # Read and parse the features
        df = pd.read_csv(csv_path, sep=';')
        feature_names = list(df.columns[1:])  # Skip 'name' column
        features = df[feature_names].values
        
        if features.shape[0] == 0:
            logging.error("No features extracted")
            return None
        
        # Get first row of features
        features_row = features[0]
        
        # Log statistics
        logging.info(f"Extracted {len(feature_names)} OpenSMILE features")
        logging.info(f"Feature shape: {features_row.shape}")
        logging.info(f"Min: {np.min(features_row):.4f}, Max: {np.max(features_row):.4f}")
        logging.info(f"Mean: {np.mean(features_row):.4f}, Std: {np.std(features_row):.4f}")
        
        # Log sample of feature names
        logging.info(f"Feature name samples: {feature_names[:10]}")
        
        # Save features to NumPy file
        features_npz = os.path.join(output_dir, "opensmile_features.npz")
        save_features_to_file(features_row, features_npz)
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "audio_duration": record_seconds,
            "sample_rate": sample_rate,
            "num_features": len(feature_names),
            "feature_names": feature_names[:20] + ["..."] if len(feature_names) > 20 else feature_names,
            "config_file": config_file,
            "success": True
        }
        
        metadata_path = os.path.join(output_dir, "opensmile_extraction_metadata.json")
        save_metadata(metadata, metadata_path)
        
        return features_row
        
    except Exception as e:
        logging.error(f"Error in OpenSMILE feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract and save features for emotion recognition')
    parser.add_argument('--opensmile', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract',
                        help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf',
                        help='Path to OpenSMILE config file')
    
    args = parser.parse_args()
    
    output_dir = "feature_verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print system info
    logging.info("=== STARTING FEATURE EXTRACTION AND VERIFICATION ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"OpenCV version: {cv2.__version__}")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Extract and save features
    face_features = extract_and_save_facenet_features()
    audio_features = extract_and_save_opensmile_features(args.opensmile, args.config)
    
    # Create overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "face_features_extracted": face_features is not None,
        "face_feature_dimension": 512 if face_features is not None else None,
        "audio_features_extracted": audio_features is not None,
        "audio_feature_dimension": len(audio_features) if audio_features is not None else None,
        "output_directory": os.path.abspath(output_dir)
    }
    
    summary_path = os.path.join(output_dir, "feature_extraction_summary.json")
    save_metadata(summary, summary_path)
    
    # Print summary
    logging.info("\n=== FEATURE EXTRACTION SUMMARY ===")
    logging.info(f"FaceNet Features: {'SUCCESS' if face_features is not None else 'FAILED'}")
    logging.info(f"OpenSMILE Features: {'SUCCESS' if audio_features is not None else 'FAILED'}")
    
    if face_features is not None and audio_features is not None:
        logging.info("✅ All features extracted successfully!")
        logging.info(f"Results saved to: {os.path.abspath(output_dir)}")
    else:
        logging.warning("❌ Some feature extractions failed. See above logs for details.")
    
    logging.info("=== END OF FEATURE EXTRACTION ===")

if __name__ == "__main__":
    main()
