#!/usr/bin/env python
"""
Enhanced Compatible Real-time Emotion Recognition

This improved script addresses audio device selection and processing issues.
It uses the enhanced audio device utilities to ensure audio data is properly captured,
and adds better error handling and visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
import pyaudio
import threading
import queue
from collections import deque
import audio_device_utils
from tensorflow_compatible_model import EmotionRecognitionModel
from facenet_pytorch import MTCNN, InceptionResnetV1

# Suppress TensorFlow and PyAudio warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables for synchronization
audio_features_queue = queue.Queue()
video_frames_queue = queue.Queue(maxsize=5)  # Limit size to avoid memory issues
audio_recording = True
lock = threading.Lock()

# Constants
RATE = 16000  # Audio sampling rate
CHUNK = 1024  # Audio chunk size
FPS = 15  # Target video FPS
DEFAULT_FEATURE_WINDOW = 3  # Default feature window in seconds
FACENET_FEATURE_DIM = 512  # FaceNet feature dimension
OPENSMILE_FEATURE_DIM = 89  # OpenSMILE eGeMAPSv02 feature dimension (89 dims in training model)
EMOTION_COLORS = {
    'anger': (0, 0, 255),      # Red
    'disgust': (0, 140, 255),  # Orange
    'fear': (0, 255, 255),     # Yellow
    'happiness': (0, 255, 0),  # Green
    'sadness': (255, 0, 0),    # Blue
    'neutral': (255, 255, 255) # White
}

class EmotionRecognitionPipeline:
    """
    Enhanced real-time emotion recognition pipeline using webcam and microphone
    """

    def __init__(self, args):
        """Initialize the pipeline with the given arguments"""
        self.args = args
        self.window_size = args.window_size
        self.feature_window = args.feature_window
        self.requested_audio_device = args.audio_device
        
        # Auto-detect best microphone if not provided
        if self.requested_audio_device is None:
            self.requested_audio_device = audio_device_utils.find_best_microphone()
            
        # Test the selected audio device and fall back if necessary
        self.audio_device, self.audio_format, self.audio_message = \
            audio_device_utils.find_working_microphone(self.requested_audio_device)
            
        # Print audio device info
        print(self.audio_message)
        print(f"Using audio device {self.audio_device} for input")
            
        # Initialize audio status
        self.audio_status = "Initializing..."
        self.audio_working = False
        self.audio_frames_count = 0

        # Load the model
        print(f"Loading emotion recognition model from: {args.model}")
        self.model = EmotionRecognitionModel(args.model)

        # Setup FaceNet for face detection and feature extraction
        print("Initializing FaceNet for face detection and feature extraction")
        self.face_detector = MTCNN(
            select_largest=True,
            post_process=False,
            device='cpu',
            keep_all=False
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        # Setup smoothing - IMPORTANT: Order must match training data
        self.predictions_history = deque(maxlen=self.window_size)
        # Original training data used this order: ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5 
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]

        # Setup video capture
        self.cap = cv2.VideoCapture(args.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        self.target_fps = args.fps

        # Setup display window
        self.window_name = "Real-time Emotion Recognition (Enhanced)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, args.display_width, args.display_height)

        # Buffer for audio frames
        self.audio_buffer = np.zeros((0, OPENSMILE_FEATURE_DIM))

        # Buffer for video frames
        self.video_buffer = np.zeros((0, FACENET_FEATURE_DIM))
        self.video_frames_count = 0

        # For frame rate calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # Last prediction and status
        self.last_prediction = None
        self.status_message = "Starting up..."
        
        # Start audio thread
        self.start_audio_thread()

    def start_audio_thread(self):
        """Start the audio processing thread"""
        if self.args.opensmile and self.args.config:
            self.audio_thread = threading.Thread(target=self.process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("Audio processing thread started")
        else:
            self.audio_status = "ERROR: OpenSMILE config not provided"
            print("OpenSMILE config not provided - audio features disabled")

    def process_audio(self):
        """Process audio in a separate thread using OpenSMILE"""
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            # Open stream with specific device index
            stream = p.open(
                format=self.audio_format,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=self.audio_device,
                frames_per_buffer=CHUNK
            )
            
            self.audio_status = f"Connected to device {self.audio_device}"
            
            # Process audio continuously
            global audio_recording
            while audio_recording:
                try:
                    # Record audio data for the feature window duration
                    frames = []
                    for _ in range(0, int(RATE / CHUNK * self.feature_window)):
                        if not audio_recording:
                            break
                        try:
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            frames.append(data)
                        except Exception as e:
                            self.audio_status = f"Stream read error: {str(e)}"
                            print(self.audio_status)
                            time.sleep(0.1)  # Short delay on error
                            break

                    # If we've collected enough audio data
                    if len(frames) > 0 and audio_recording:
                        # Convert frames to numpy array
                        if self.audio_format == pyaudio.paFloat32:
                            audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                        else:
                            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                            
                        # Check if audio has content
                        rms = np.sqrt(np.mean(np.square(audio_data)))
                        if rms < (0.01 if self.audio_format == pyaudio.paFloat32 else 50):
                            self.audio_status = f"Low audio level: {rms:.2f}"
                            # Use dummy/silent audio rather than failing
                            audio_features = np.zeros((1, OPENSMILE_FEATURE_DIM))
                        else:
                            # Extract features with OpenSMILE
                            features = audio_device_utils.extract_opensmile_features(
                                audio_data,
                                self.args.opensmile,
                                self.args.config,
                                self.audio_format,
                                RATE
                            )
                            
                            if features is not None and len(features) > 0:
                                audio_features = features
                                self.audio_working = True
                                self.audio_status = f"Processing audio frames: {len(features)}"
                                self.audio_frames_count = len(features)
                            else:
                                self.audio_status = "No features extracted"
                                audio_features = np.zeros((1, OPENSMILE_FEATURE_DIM))
                                
                        # Update the audio buffer
                        with lock:
                            self.audio_buffer = np.vstack([self.audio_buffer, audio_features])
                            
                            # Trim buffer to feature window * fps frames 
                            # (synchronized with video buffer)
                            max_frames = int(self.feature_window * self.target_fps)
                            if len(self.audio_buffer) > max_frames:
                                self.audio_buffer = self.audio_buffer[-max_frames:]
                                
                    else:
                        self.audio_status = "No audio frames collected"
                
                except Exception as e:
                    self.audio_status = f"Audio processing error: {str(e)}"
                    print(self.audio_status)
                    time.sleep(0.5)  # Longer delay on serious error

        except Exception as e:
            self.audio_status = f"Failed to open audio device {self.audio_device}: {str(e)}"
            print(self.audio_status)
            
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()
            self.audio_status = "Audio thread stopped"

    def extract_face_features(self, frame):
        """Extract facial features using FaceNet"""
        # Convert from BGR (OpenCV) to RGB (FaceNet)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        try:
            boxes, probs = self.face_detector.detect(rgb_frame, landmarks=False)
            
            # If a face was detected
            if boxes is not None and len(boxes) > 0:
                # Get the box with highest probability
                box = boxes[0]
                
                # Apply padding for better performance
                padding = 30
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                
                # Apply padding with bounds checking
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Draw box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop and extract features
                face_img = rgb_frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = np.moveaxis(face_img, -1, 0)  # HWC to CHW format
                face_img = np.expand_dims(face_img, 0)  # Add batch dimension
                face_img = face_img / 255.0  # Normalize
                
                # Extract embeddings using FaceNet and normalize using our stored stats
                with torch.no_grad():
                    face_tensor = torch.from_numpy(face_img).float()
                    embedding = self.facenet(face_tensor).numpy()
                    
                    # Apply video normalization to the embedding
                    from feature_normalizer import normalize_features
                    embedding_norm = normalize_features(embedding, name="video")
                
                return embedding_norm.reshape(1, -1), (x1, y1, x2, y2)
            
            return None, None
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None, None

    def predict_emotion(self):
        """Predict emotion using the current audio and video buffers"""
        if len(self.audio_buffer) == 0 or len(self.video_buffer) == 0:
            return None
            
        # Prepare inputs for the model
        # Add batch dimension if not present
        video_features = np.expand_dims(self.video_buffer, 0) if self.video_buffer.ndim == 2 else self.video_buffer
        audio_features = np.expand_dims(self.audio_buffer, 0) if self.audio_buffer.ndim == 2 else self.audio_buffer
        
        # Predict emotion - ensuring correct order (video, audio) as in training
        try:
            prediction = self.model.predict(video_features, audio_features)
            # Add to history for smoothing
            self.predictions_history.append(prediction[0])
            
            # Average predictions over window for smoothing
            if len(self.predictions_history) > 0:
                avg_prediction = np.mean(self.predictions_history, axis=0)
                return avg_prediction
            else:
                return prediction[0]
        except Exception as e:
            print(f"Error in emotion prediction: {str(e)}")
            return None

    def display_emotion_probabilities(self, frame, prediction, face_box=None):
        """Display emotion probabilities as a bar chart"""
        if prediction is None:
            # If no prediction, show waiting message
            msg = "Waiting for audio and video data..."
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if face_box:
                x1, y1, x2, y2 = face_box
                cv2.putText(frame, "Waiting", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return
            
        # Get the top emotion
        max_idx = np.argmax(prediction)
        emotion = self.emotions[max_idx]
        score = prediction[max_idx]
        
        # If we have a face, label it with the emotion
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.putText(frame, f"{emotion}: {score:.2f}", (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, EMOTION_COLORS[emotion], 2)
        
        # Draw emotion probabilities as bar chart
        chart_width = 400
        chart_height = 200
        chart_x = 20
        chart_y = frame.shape[0] - chart_height - 20
        
        # Background
        cv2.rectangle(frame, (chart_x, chart_y), 
                    (chart_x + chart_width, chart_y + chart_height), 
                    (32, 32, 32), -1)
                    
        cv2.putText(frame, "Emotion Probabilities", 
                  (chart_x + 10, chart_y + 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw each emotion bar
        bar_height = 25
        gap = 10
        for i, emotion in enumerate(self.emotions):
            # Calculate bar position
            y_pos = chart_y + 50 + i * (bar_height + gap)
            
            # Draw label
            cv2.putText(frame, f"{emotion}", 
                      (chart_x + 10, y_pos + bar_height - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                      
            # Draw background bar (100%)
            cv2.rectangle(frame, 
                        (chart_x + 100, y_pos), 
                        (chart_x + chart_width - 20, y_pos + bar_height), 
                        (100, 100, 100), -1)
            
            # Draw value bar
            bar_width = int((chart_width - 120) * prediction[i])
            cv2.rectangle(frame, 
                        (chart_x + 100, y_pos), 
                        (chart_x + 100 + bar_width, y_pos + bar_height), 
                        EMOTION_COLORS[emotion], -1)
            
            # Show percentage
            cv2.putText(frame, f"{prediction[i]:.2f}", 
                      (chart_x + chart_width - 15, y_pos + bar_height - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA, False)

    def run(self):
        """Run the emotion recognition pipeline"""
        print("Starting real-time emotion recognition pipeline")
        print("Press 'q' or 'ESC' to exit")
        
        global audio_recording
        
        while True:
            # Capture frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break
                
            # Calculate FPS
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = self.new_frame_time
            
            # Extract face features
            face_features, face_box = self.extract_face_features(frame)
            
            if face_features is not None:
                # Update video buffer
                with lock:
                    self.video_buffer = np.vstack([self.video_buffer, face_features])
                    self.video_frames_count = len(self.video_buffer)
                    
                    # Trim buffer to feature window * fps frames
                    max_frames = int(self.feature_window * self.target_fps)
                    if len(self.video_buffer) > max_frames:
                        self.video_buffer = self.video_buffer[-max_frames:]
            
            # Predict emotion if we have enough data
            if len(self.audio_buffer) > 0 and len(self.video_buffer) > 0:
                prediction = self.predict_emotion()
                self.last_prediction = prediction
                self.status_message = "Processing both audio and video"
            else:
                prediction = None
                
                if len(self.audio_buffer) == 0 and len(self.video_buffer) == 0:
                    self.status_message = "Waiting for audio and video data"
                elif len(self.audio_buffer) == 0:
                    self.status_message = "Waiting for audio data"
                else:
                    self.status_message = "Waiting for video data (no face detected)"
            
            # Display status overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                      
            cv2.putText(frame, f"Status: {self.status_message}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                      
            cv2.putText(frame, f"Audio: {self.audio_status}", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                      (0, 255, 0) if self.audio_working else (0, 0, 255), 2)
                      
            # Display buffer information
            cv2.putText(frame, 
                      f"Audio buffer: {len(self.audio_buffer)} frames", 
                      (10, frame.shape[0] - 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                      
            cv2.putText(frame, 
                      f"Video buffer: {len(self.video_buffer)} frames", 
                      (10, frame.shape[0] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Display emotion probabilities
            self.display_emotion_probabilities(frame, self.last_prediction, face_box)
            
            # Show the frame
            cv2.imshow(self.window_name, frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        # Cleanup
        audio_recording = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model', type=str, default='models/dynamic_padding_no_leakage/model_best.h5',
                      help='Path to the emotion recognition model')
    parser.add_argument('--opensmile', type=str, required=False,
                      help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to OpenSMILE config file')
    parser.add_argument('--camera_index', type=int, default=0,
                      help='Camera device index')
    parser.add_argument('--audio_device', type=int, default=None,
                      help='Audio device index (None for auto-detection)')
    parser.add_argument('--fps', type=int, default=15,
                      help='Target FPS for video capture')
    parser.add_argument('--window_size', type=int, default=5,
                      help='Window size for prediction smoothing')
    parser.add_argument('--feature_window', type=int, default=DEFAULT_FEATURE_WINDOW,
                      help='Feature window size in seconds')
    parser.add_argument('--display_width', type=int, default=1200,
                      help='Display window width')
    parser.add_argument('--display_height', type=int, default=700,
                      help='Display window height')
    
    args = parser.parse_args()
    
    # Print audio devices before starting
    devices = audio_device_utils.list_audio_devices()
    print("\n=== Available Audio Input Devices ===")
    for device in devices:
        print(f"Device {device['index']}: {device['name']} ({device['inputs']} inputs)")
    print()
    
    # Run the pipeline
    pipeline = EmotionRecognitionPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    # Conditionally import torch only when needed (for FaceNet)
    import torch
    
    main()
