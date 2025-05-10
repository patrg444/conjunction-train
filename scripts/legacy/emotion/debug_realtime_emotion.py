#!/usr/bin/env python3
"""
Debug script for the real-time emotion recognition system.
This script will try to diagnose issues with the webcam, model loading, or OpenSMILE integration.
"""

import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf
import pyaudio
import logging
import subprocess
from facenet_extractor import FaceNetExtractor
import pandas as pd
import wave
import argparse

# Configure more detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level instead of INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emotion_debug.log"),
        logging.StreamHandler()
    ]
)

def check_camera():
    """Test if camera can be accessed and captures frames."""
    logging.info("Testing camera access...")
    
    # Try different camera indices
    for cam_index in [0, 1, 2]:
        logging.debug(f"Trying camera index {cam_index}...")
        
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logging.warning(f"Failed to open camera at index {cam_index}")
            continue
        
        # Try to read a few frames
        frames_read = 0
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames_read += 1
                logging.debug(f"Successfully read frame {frames_read} from camera {cam_index}")
            else:
                logging.debug(f"Failed to read frame from camera {cam_index}")
            time.sleep(0.1)
        
        # Release the camera
        cap.release()
        
        if frames_read > 0:
            logging.info(f"Camera index {cam_index} is working (read {frames_read}/5 frames)")
            return cam_index
    
    logging.error("No working camera found")
    return -1

def check_model(model_path):
    """Test if model can be loaded correctly."""
    logging.info(f"Testing model loading from: {model_path}")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist: {model_path}")
        return False
    
    try:
        # Try to load the model with custom LSTM layer to handle parameters
        class LSTMCompatLayer(tf.keras.layers.LSTM):
            def __init__(self, *args, time_major=None, **kwargs):
                if time_major is not None:
                    logging.debug(f"Removing time_major={time_major} parameter from LSTM")
                    # Remove time_major parameter if present
                    kwargs.pop('time_major', None)
                super(LSTMCompatLayer, self).__init__(*args, **kwargs)
        
        logging.debug(f"Loading model with custom LSTM layer...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'LSTM': LSTMCompatLayer}
        )
        
        logging.info(f"Model loaded successfully!")
        logging.info(f"Model input shapes: {[input.shape for input in model.inputs]}")
        logging.info(f"Model output shapes: {[output.shape for output in model.outputs]}")
        
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        logging.debug("Model summary:\n" + "\n".join(model_summary))
        
        return True
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.exception("Stack trace:")
        return False

def check_facenet():
    """Test if FaceNet can be initialized and used."""
    logging.info("Testing FaceNet face detector...")
    
    try:
        face_extractor = FaceNetExtractor(
            keep_all=False,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]
        )
        logging.info(f"FaceNet initialized successfully with embedding dimension: {face_extractor.embedding_dim}")
        return True
    except Exception as e:
        logging.error(f"Error initializing FaceNet: {str(e)}")
        logging.exception("Stack trace:")
        return False

def check_opensmile(opensmile_path, config_file):
    """Test if OpenSMILE is accessible and working."""
    logging.info(f"Testing OpenSMILE at: {opensmile_path}")
    logging.info(f"With config: {config_file}")
    
    # Check if paths exist
    if not os.path.exists(opensmile_path):
        logging.error(f"OpenSMILE executable not found: {opensmile_path}")
        return False
    
    if not os.path.exists(config_file):
        logging.error(f"OpenSMILE config file not found: {config_file}")
        return False
    
    # Create a simple test audio file
    temp_dir = "temp_debug_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_wav = os.path.join(temp_dir, "test.wav")
    temp_csv = os.path.join(temp_dir, "test.csv")
    
    try:
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 1  # second
        t = np.linspace(0, duration, sample_rate * duration)
        data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        data = (data * 32767).astype(np.int16)  # Convert to 16-bit PCM
        
        # Write to WAV file
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
        
        logging.debug(f"Created test audio file: {temp_wav}")
        
        # Run OpenSMILE
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", temp_wav,
            "-csvoutput", temp_csv,
            "-instname", "debug"
        ]
        
        logging.debug(f"Running OpenSMILE command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"OpenSMILE execution failed with code {result.returncode}")
            logging.error(f"stderr: {result.stderr}")
            return False
        
        # Check if output file was created
        if not os.path.exists(temp_csv):
            logging.error(f"OpenSMILE did not create output file: {temp_csv}")
            return False
        
        # Try to read the features
        df = pd.read_csv(temp_csv, sep=';')
        feature_names = df.columns[1:]  # Skip the 'name' column
        features = df[feature_names].values
        
        logging.info(f"OpenSMILE extracted {len(feature_names)} features")
        logging.debug(f"Feature names: {', '.join(feature_names[:5])}...")
        
        return True
    except Exception as e:
        logging.error(f"Error testing OpenSMILE: {str(e)}")
        logging.exception("Stack trace:")
        return False
    finally:
        # Clean up
        for f in [temp_wav, temp_csv]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def check_audio_device():
    """Test if audio recording is working."""
    logging.info("Testing audio recording...")
    
    try:
        p = pyaudio.PyAudio()
        
        # Print available audio devices
        device_count = p.get_device_count()
        logging.info(f"Found {device_count} audio devices")
        
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                logging.debug(f"Device {i}: {device_info['name']}")
                logging.debug(f"  - Input channels: {device_info['maxInputChannels']}")
                logging.debug(f"  - Output channels: {device_info['maxOutputChannels']}")
                logging.debug(f"  - Default sample rate: {device_info['defaultSampleRate']}")
            except Exception as e:
                logging.error(f"Error getting device info for index {i}: {str(e)}")
        
        # Try to open a recording stream
        sample_rate = 16000
        channels = 1
        chunk_size = 1024
        format = pyaudio.paInt16
        
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                logging.debug(f"Trying to record from device {i}: {device_info['name']}")
                
                try:
                    stream = p.open(
                        format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=chunk_size
                    )
                    
                    # Record a small amount of audio
                    frames = []
                    for j in range(10):  # Record 10 chunks
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        frames.append(data)
                    
                    stream.stop_stream()
                    stream.close()
                    
                    # Check if we got audio data
                    audio_data = b''.join(frames)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
                    
                    logging.info(f"Recorded from device {i}, RMS: {rms}")
                    if rms > 0:
                        logging.info(f"Successfully recorded audio from device {i}")
                    else:
                        logging.warning(f"Recorded silence from device {i}")
                    
                except Exception as e:
                    logging.error(f"Error recording from device {i}: {str(e)}")
        
        p.terminate()
        return True
    except Exception as e:
        logging.error(f"Error testing audio: {str(e)}")
        logging.exception("Stack trace:")
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug real-time emotion recognition system')
    parser.add_argument('--model', type=str, default='models/dynamic_padding_no_leakage/model_best.h5', 
                        help='Path to model file')
    parser.add_argument('--opensmile', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract',
                        help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, 
                        default='./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf',
                        help='Path to OpenSMILE config file')
    
    args = parser.parse_args()
    
    logging.info("=== STARTING SYSTEM DIAGNOSTICS ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"OpenCV version: {cv2.__version__}")
    logging.info(f"PyAudio version: {pyaudio.__version__}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Current directory: {os.getcwd()}")
    
    # Run the checks
    camera_status = check_camera()
    model_status = check_model(args.model)
    facenet_status = check_facenet()
    opensmile_status = check_opensmile(args.opensmile, args.config)
    audio_status = check_audio_device()
    
    # Print summary
    logging.info("\n=== DIAGNOSTIC SUMMARY ===")
    logging.info(f"Camera: {'WORKING (index ' + str(camera_status) + ')' if camera_status >= 0 else 'FAILED'}")
    logging.info(f"Model loading: {'SUCCESS' if model_status else 'FAILED'}")
    logging.info(f"FaceNet: {'SUCCESS' if facenet_status else 'FAILED'}")
    logging.info(f"OpenSMILE: {'SUCCESS' if opensmile_status else 'FAILED'}")
    logging.info(f"Audio recording: {'SUCCESS' if audio_status else 'FAILED'}")
    
    if camera_status >= 0 and model_status and facenet_status and opensmile_status and audio_status:
        logging.info("All components are working correctly!")
    else:
        logging.warning("Some components failed. See above logs for details.")
    
    logging.info("=== END OF DIAGNOSTICS ===")

if __name__ == "__main__":
    main()
