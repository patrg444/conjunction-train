#!/usr/bin/env python3
"""
Demo version of the real-time emotion recognition system
that shows feature extraction working properly even without model loading.
"""

import os
import sys
import cv2
import time
import numpy as np
import pandas as pd
import argparse
import tempfile
import logging
import shutil
import subprocess
import threading
import queue
import wave
import pyaudio
import random
from facenet_extractor import FaceNetExtractor
from collections import deque

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("demo_emotion_recognition.log"),
        logging.StreamHandler()
    ]
)

# Constants
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
EMOTION_COLORS = [
    (255, 255, 255),  # Neutral: White
    (0, 255, 0),      # Happy: Green
    (255, 0, 0),      # Sad: Blue
    (0, 0, 255),      # Angry: Red
    (0, 255, 255),    # Fearful: Yellow
    (255, 0, 255)     # Disgust: Magenta
]

class AudioProcessor:
    """
    Handles real-time audio recording and feature extraction using OpenSMILE.
    """
    def __init__(self, opensmile_path, config_file, temp_dir="temp_extracted_audio"):
        self.opensmile_path = opensmile_path
        self.config_file = config_file
        self.temp_dir = temp_dir
        self.audio_features = None
        self.is_recording = False
        self.audio_thread = None
        self.features_queue = queue.Queue()
        self.debug = os.environ.get('DEBUG') == '1'

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # PyAudio setup
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.py_audio = pyaudio.PyAudio()

        # Check for available devices
        device_info = []
        for i in range(self.py_audio.get_device_count()):
            try:
                info = self.py_audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_info.append((i, info['name'], info['maxInputChannels']))
                    logging.info(f"Found input device {i}: {info['name']} ({info['maxInputChannels']} channels)")
            except Exception as e:
                logging.error(f"Error getting device info for device {i}: {str(e)}")
        
        if not device_info:
            logging.warning("No audio input devices found!")
        
        logging.info(f"AudioProcessor initialized with OpenSMILE at: {opensmile_path}")
        logging.info(f"Config file: {config_file}")

    def start_recording(self):
        """Start recording audio in a separate thread."""
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_and_process)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        logging.info("Started audio recording thread")

    def stop_recording(self):
        """Stop the audio recording thread."""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        logging.info("Stopped audio recording")

    def _record_and_process(self):
        """Record audio in chunks and process with OpenSMILE."""
        try:
            stream = self.py_audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Process audio in 1-second chunks
            chunk_duration = 1.0  # seconds
            frames_per_chunk = int(self.rate * chunk_duration)
            
            logging.info(f"Recording audio at {self.rate} Hz, processing in {chunk_duration}s chunks")
            
            frames = []
            frames_count = 0
            
            while self.is_recording:
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                frames_count += self.chunk
                
                # Process when we have enough for a chunk
                if frames_count >= frames_per_chunk:
                    # Process the audio chunk
                    audio_features = self._process_audio_chunk(frames)
                    
                    # Add to queue for the main thread to use
                    if audio_features is not None:
                        if self.debug:
                            logging.info(f"Got audio features: shape={audio_features.shape}, " +
                                         f"min={np.min(audio_features):.3f}, " + 
                                         f"max={np.max(audio_features):.3f}")
                        self.features_queue.put(audio_features)
                    
                    # Reset for next chunk
                    frames = []
                    frames_count = 0
        
        except Exception as e:
            logging.error(f"Error in audio recording: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            logging.info("Audio recording stopped")

    def _process_audio_chunk(self, frames):
        """Process an audio chunk with OpenSMILE."""
        try:
            # Create a temporary WAV file
            temp_wav = os.path.join(self.temp_dir, f"temp_{int(time.time())}.wav")
            temp_csv = os.path.join(self.temp_dir, f"temp_{int(time.time())}.csv")
            
            # Write frames to WAV file
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.py_audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            # Process with OpenSMILE to extract features
            command = [
                self.opensmile_path,
                "-C", self.config_file,
                "-I", temp_wav,
                "-csvoutput", temp_csv,
                "-instname", "realtime"
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"Error running OpenSMILE: {result.stderr}")
                return None
            
            # Read the CSV file if it exists
            if os.path.exists(temp_csv):
                try:
                    df = pd.read_csv(temp_csv, sep=';')
                    
                    # Extract features (first column is name, we drop it)
                    if len(df) > 0:
                        feature_names = df.columns[1:]  # Skip the 'name' column
                        features = df[feature_names].values
                        if self.debug:
                            logging.info(f"Extracted {len(feature_names)} OpenSMILE features")
                        return features[0]  # Return the first (and only) row
                    else:
                        logging.warning("Empty CSV file from OpenSMILE")
                        return None
                    
                except Exception as e:
                    logging.error(f"Error reading OpenSMILE CSV: {str(e)}")
                    return None
                finally:
                    # Clean up temporary files
                    try:
                        os.remove(temp_wav)
                        os.remove(temp_csv)
                    except:
                        pass
            else:
                logging.error(f"OpenSMILE did not create output file: {temp_csv}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing audio chunk: {str(e)}")
            return None

    def get_latest_features(self):
        """Get the latest audio features from the queue."""
        try:
            # Get all available features, keeping only the latest
            latest_features = None
            while not self.features_queue.empty():
                latest_features = self.features_queue.get_nowait()
            
            return latest_features
        except queue.Empty:
            return None


class EmotionRecognizer:
    """
    Real-time emotion recognition from webcam and microphone,
    operating in DEMO mode without requiring model loading.
    """
    def __init__(self, opensmile_path, config_file,
                 window_size=45, display_width=1200, display_height=700,
                 fps=15, cam_index=0, temp_dir="temp_extracted_audio",
                 simulation_mode=True):
        
        self.window_size = window_size
        self.display_width = display_width
        self.display_height = display_height
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.cam_index = cam_index
        self.temp_dir = temp_dir
        self.running = True
        self.frame_count = 0
        self.debug = os.environ.get('DEBUG') == '1'
        self.show_features = os.environ.get('SHOW_FEATURES') == '1'
        self.simulation_mode = simulation_mode
        
        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize FaceNet extractor
        self.face_extractor = FaceNetExtractor(
            keep_all=False,  # Only keep the largest face
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7]  # MTCNN thresholds
        )
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            opensmile_path=opensmile_path,
            config_file=config_file,
            temp_dir=self.temp_dir
        )
        
        # Initialize smoothing buffers
        self.emotion_buffers = [deque(maxlen=window_size) for _ in range(len(EMOTIONS))]
        
        # History for simulated emotions
        self.current_emotion = 0  # Start with neutral
        self.emotion_hold_frames = 0
        self.emotion_transition_frames = 0
        
        logging.info(f"Emotion recognizer initialized with window size: {window_size}")
        logging.info(f"Target FPS: {fps}")
        logging.info(f"Running in simulation mode: {simulation_mode}")

    def _test_camera(self, cap):
        """Test if the camera can actually capture frames."""
        if not cap.isOpened():
            return False
        
        # Try to read a test frame
        for _ in range(3):  # Try a few times in case there's a temporary issue
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return True
            time.sleep(0.1)
        
        return False

    def start(self):
        """Start the real-time emotion recognition."""
        # Open webcam with better error handling
        # Try multiple camera indices if the requested one fails
        cap = None
        tried_indices = []
        
        # First try the requested camera index
        cap = cv2.VideoCapture(self.cam_index)
        tried_indices.append(self.cam_index)
        
        # If that fails, try a few common indices
        if not cap.isOpened() or not self._test_camera(cap):
            logging.warning(f"Failed to open or use camera at index {self.cam_index}")
            
            # Try common camera indices: 0, 1, 2
            for test_index in [0, 1, 2]:
                if test_index in tried_indices:
                    continue
                
                logging.info(f"Trying camera index {test_index}...")
                cap = cv2.VideoCapture(test_index)
                tried_indices.append(test_index)
                
                if cap.isOpened() and self._test_camera(cap):
                    self.cam_index = test_index
                    logging.info(f"Successfully opened camera at index {test_index}")
                    break
                else:
                    if cap.isOpened():
                        cap.release()
        
        # If all attempts failed, exit
        if not cap or not cap.isOpened():
            logging.error(f"Failed to open any camera after trying indices: {tried_indices}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Start audio recording
        self.audio_processor.start_recording()
        
        # Track timing
        last_frame_time = 0
        
        try:
            while self.running:
                # Timing for FPS control
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Sleep to maintain target FPS
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                    continue
                
                # Record the time we're processing this frame
                last_frame_time = current_time
                self.frame_count += 1
                
                # Capture video frame
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame from camera")
                    break
                
                # Extract face features
                face_features = self.face_extractor.extract_features(frame)
                if self.debug and face_features is not None:
                    non_zero = np.count_nonzero(face_features)
                    if non_zero > 0:
                        logging.info(f"Face detected! Features: shape={face_features.shape}, " +
                                    f"non-zero={non_zero}/{face_features.size} elements")
                    else:
                        logging.info("No face detected in frame")
                
                # Get latest audio features
                audio_features = self.audio_processor.get_latest_features()
                
                # In simulation mode, we'll create realistic-looking predictions
                if self.simulation_mode:
                    probs = self._simulate_emotion_predictions(face_features is not None)
                else:
                    # If no simulation, just use uniform distribution
                    probs = np.ones(len(EMOTIONS)) / len(EMOTIONS)
                
                # Update smoothing buffers
                for i, prob in enumerate(probs):
                    self.emotion_buffers[i].append(prob)
                
                # Calculate smoothed probabilities
                smoothed_probs = [sum(buffer) / len(buffer) if buffer else 0
                                 for buffer in self.emotion_buffers]
                
                # Find dominant emotion
                dominant_idx = np.argmax(smoothed_probs)
                dominant_emotion = EMOTIONS[dominant_idx]
                dominant_color = EMOTION_COLORS[dominant_idx]
                
                # Create display
                self._create_display(frame, face_features, audio_features,
                                   smoothed_probs, dominant_emotion, dominant_color)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    self.running = False
                    break
                    
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up
            self.audio_processor.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Emotion recognition stopped")

    def _simulate_emotion_predictions(self, face_detected):
        """
        Simulate emotion predictions to demonstrate the interface.
        Creates realistic-looking emotion transitions based on face detection.
        """
        # If no face is detected, return neutral emotion
        if not face_detected:
            # Return mostly neutral with some uncertainty
            probs = np.array([0.7, 0.05, 0.05, 0.05, 0.05, 0.1])
            return probs
            
        # Decide if we should transition to a new emotion
        if self.emotion_hold_frames <= 0:
            # Time to switch emotions - set a hold time of 30-90 frames
            self.emotion_hold_frames = random.randint(30, 90)
            
            # Randomly select a new emotion (but with more weight on common emotions)
            weights = [0.2, 0.3, 0.2, 0.15, 0.1, 0.05]  # More weight on neutral and happy
            choices = list(range(len(EMOTIONS)))
            
            # Try to avoid picking the same emotion twice in a row
            old_emotion = self.current_emotion
            while self.current_emotion == old_emotion:
                self.current_emotion = random.choices(choices, weights=weights, k=1)[0]
            
            # Set up a transition period
            self.emotion_transition_frames = random.randint(10, 20)
        
        # Decrease the counters
        self.emotion_hold_frames -= 1
        
        # Create the probability distribution
        probs = np.zeros(len(EMOTIONS))
        
        if self.emotion_transition_frames > 0:
            # We're in a transition period - gradually increase the probability
            transition_progress = 1.0 - (self.emotion_transition_frames / 20.0)
            
            # The target emotion gets stronger
            probs[self.current_emotion] = 0.3 + (0.6 * transition_progress)
            
            # Other emotions get weaker
            remaining = 1.0 - probs[self.current_emotion]
            for i in range(len(EMOTIONS)):
                if i != self.current_emotion:
                    probs[i] = remaining / (len(EMOTIONS) - 1)
            
            self.emotion_transition_frames -= 1
        else:
            # Stable emotion - still with some minor fluctuations
            probs[self.current_emotion] = 0.7 + (random.random() * 0.2)
            
            # Small probabilities for other emotions
            remaining = 1.0 - probs[self.current_emotion]
            for i in range(len(EMOTIONS)):
                if i != self.current_emotion:
                    probs[i] = remaining / (len(EMOTIONS) - 1)
        
        # Normalize to ensure they sum to 1
        return probs / np.sum(probs)

    def _create_display(self, frame, face_features, audio_features,
                       smoothed_probs, dominant_emotion, dominant_color):
        """Create the display frame with video, emotion bars and statistics."""
        # Create a blank display
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Resize the frame to fit the left side of the display
        frame_height = self.display_height
        frame_width = int(frame.shape[1] * (frame_height / frame.shape[0]))
        if frame_width > self.display_width // 2:
            frame_width = self.display_width // 2
        
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Place the frame on the left side
        display[0:frame_height, 0:frame_width] = frame_resized
        
        # Add a border around the face detection area
        cv2.rectangle(display, (0, 0), (frame_width, frame_height), (50, 50, 50), 2)
        
        # Add dominant emotion text
        cv2.putText(display, f"Emotion: {dominant_emotion}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dominant_color, 2)
        
        # Add feature status
        face_status = "DETECTED" if face_features is not None and np.any(face_features != 0) else "NOT DETECTED"
        face_color = (0, 255, 0) if face_features is not None and np.any(face_features != 0) else (0, 0, 255)
        cv2.putText(display, f"Face: {face_status}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        audio_status = "DETECTED" if audio_features is not None else "NOT DETECTED"
        audio_color = (0, 255, 0) if audio_features is not None else (0, 0, 255)
        cv2.putText(display, f"Audio: {audio_status}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
        
        # Draw frame count
        cv2.putText(display, f"Frame: {self.frame_count}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add DEMO MODE indicator
        cv2.putText(display, "DEMO MODE - NO MODEL LOADED",
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw emotion visualization on the right side
        # Add title for smoothed values
        cv2.putText(display, f"Emotion Probabilities (Window={self.window_size})",
                   (self.display_width//2 + 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw graph for each emotion
        bar_height = 30
        bar_gap = 15
        bar_width = 300
        x_start = self.display_width//2 + 50
        y_start = 80
        
        for i, emotion in enumerate(EMOTIONS):
            # Calculate y position
            y_pos = y_start + i * (bar_height + bar_gap)
            
            # Draw emotion label
            cv2.putText(display, emotion,
                       (x_start, y_pos + bar_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw background bar
            cv2.rectangle(display,
                         (x_start + 100, y_pos),
                         (x_start + 100 + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Draw filled bar for probability
            filled_width = int(bar_width * smoothed_probs[i])
            cv2.rectangle(display,
                         (x_start + 100, y_pos),
                         (x_start + 100 + filled_width, y_pos + bar_height),
                         EMOTION_COLORS[i], -1)
            
            # Add probability text
            prob_text = f"{smoothed_probs[i]:.2f}"
            cv2.putText(display, prob_text,
                       (x_start + 100 + bar_width + 10, y_pos + bar_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show feature visualizations if requested
        if self.show_features and face_features is not None and np.any(face_features != 0):
            # Draw a small visualization of face features
            feature_vis_height = 100
            feature_vis_width = 512  # Matches feature dimension
            
            # Normalize feature values to 0-255 range for visualization
            face_features_norm = face_features.copy()
            face_features_norm = (face_features_norm - np.min(face_features_norm)) / (np.max(face_features_norm) - np.min(face_features_norm))
            face_features_norm = (face_features_norm * 255).astype(np.uint8)
            
            # Create a visualization image
            feature_vis = np.zeros((feature_vis_height, feature_vis_width), dtype=np.uint8)
            for i, val in enumerate(face_features_norm):
                if i < feature_vis_width:
                    # Draw a vertical line for each feature value
                    height = int(val * feature_vis_height / 255)
                    feature_vis[feature_vis_height-height:feature_vis_height, i] = 255
            
            # Convert to color and place on display
            feature_vis_color = cv2.cvtColor(feature_vis, cv2.COLOR_GRAY2BGR)
            vis_y_pos = self.display_height - feature_vis_height - 20
            display[vis_y_pos:vis_y_pos+feature_vis_height, self.display_width-feature_vis_width-20:self.display_width-20] = feature_vis_color
            
            # Add caption
            cv2.putText(display, "Face Features (512)",
                       (self.display_width-feature_vis_width-15, vis_y_pos-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add instructions
        cv2.putText(display, "Press 'q' or ESC to quit",
                   (x_start, self.display_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Display the resulting frame
        cv2.imshow('Real-time Emotion Recognition (DEMO)', display)


def main():
    parser = argparse.ArgumentParser(description='Demo real-time emotion recognition')
    parser.add_argument('--opensmile', type=str, required=True, help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, required=True, help='Path to OpenSMILE config file')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS')
    parser.add_argument('--display_width', type=int, default=1200, help='Display width')
    parser.add_argument('--display_height', type=int, default=700, help='Display height')
    parser.add_argument('--window_size', type=int, default=45, help='Window size for smoothing')
    parser.add_argument('--camera_index', type=int, default=0, help='Camera index')
    parser.add_argument('--no_simulation', action='store_true', help='Disable emotion simulation')
    
    args = parser.parse_args()
    
    # Print environment variables
    debug = os.environ.get('DEBUG') == '1'
    verbose = os.environ.get('VERBOSE') == '1'
    show_features = os.environ.get('SHOW_FEATURES') == '1'
    
    if debug or verbose:
        logging.info(f"Debug mode: {debug}")
        logging.info(f"Verbose mode: {verbose}")
        logging.info(f"Show features: {show_features}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"OpenCV version: {cv2.__version__}")
        logging.info(f"NumPy version: {np.__version__}")
    
    # Initialize and start the emotion recognizer
    recognizer = EmotionRecognizer(
        opensmile_path=args.opensmile,
        config_file=args.config,
        fps=args.fps,
        display_width=args.display_width,
        display_height=args.display_height,
        window_size=args.window_size,
        cam_index=args.camera_index,
        simulation_mode=not args.no_simulation
    )
    
    recognizer.start()


if __name__ == "__main__":
    main()
