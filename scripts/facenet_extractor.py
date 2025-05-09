#!/usr/bin/env python3
"""
FaceNet-based face embedding extractor using facenet-pytorch.
This is intended as a replacement for DeepFace to provide better face detection
and higher quality embeddings for emotion recognition tasks.
"""

import cv2
import numpy as np
import torch
import logging
from typing import Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("facenet_extractor.log"),
        logging.StreamHandler()
    ]
)

class FaceNetExtractor:
    """
    Extracts face embeddings using FaceNet (via facenet-pytorch).
    """

    def __init__(self, device=None, keep_all=False, min_face_size=20, thresholds=[0.6, 0.7, 0.7]):
        """
        Initializes the FaceNetExtractor.

        Args:
            device (torch.device, optional): The device to use for computation (CPU or CUDA).
                Defaults to CUDA if available, otherwise CPU.
            keep_all (bool): Whether to keep all detected faces (True) or only the largest (False).
            min_face_size (int): Minimum face size in pixels.
            thresholds (list): MTCNN detection thresholds for the three stages.
        """
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is not installed. Please install it with: "
                "pip install facenet-pytorch"
            )
            
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            keep_all=keep_all,
            device=self.device,
            min_face_size=min_face_size,
            thresholds=thresholds
        )
        
        # Initialize the FaceNet model (InceptionResnetV1)
        self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        
        # Define embedding dimension (512 for FaceNet)
        self.embedding_dim = 512
        logging.info(f"FaceNetExtractor initialized with embedding dimension: {self.embedding_dim}")

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts face embeddings from a single frame.

        Args:
            frame (np.ndarray): The input frame (BGR format, from OpenCV).

        Returns:
            np.ndarray: The face embedding (512-dimensional) if a face is detected,
                otherwise a zero-filled array of the same size.
        """
        try:
            # Convert BGR to RGB (facenet-pytorch expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face boxes
            boxes, _ = self.mtcnn.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:
                # Get face crops (handles alignment automatically)
                faces = self.mtcnn(rgb_frame)
                
                if faces is not None:
                    # If faces is a tensor and 3D, add batch dimension
                    if isinstance(faces, torch.Tensor) and faces.dim() == 3:
                        faces = faces.unsqueeze(0)
                    elif isinstance(faces, list) and len(faces) > 0:
                        # If it's a list of face tensors, stack them
                        faces = torch.stack(faces)
                    
                    # Process face embeddings
                    with torch.no_grad():
                        embeddings = self.model(faces.to(self.device))
                        
                    # Convert to numpy
                    np_embeddings = embeddings.cpu().numpy()
                    
                    # Select primary (largest) face if multiple faces detected
                    if np_embeddings.shape[0] > 1:
                        areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
                        largest_idx = np.argmax(areas)
                        return np_embeddings[largest_idx]
                    else:
                        return np_embeddings[0]
            
            # Return zeros if no face detected
            return np.zeros(self.embedding_dim)
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            # Return zeros on error to maintain consistency with DeepFace approach
            return np.zeros(self.embedding_dim)

    def process_video(self, video_path: str, sample_interval: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire video file and extract embeddings for all frames.

        Args:
            video_path (str): Path to the video file.
            sample_interval (int): Process every Nth frame (default: 1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Array of embeddings, shape (num_frames, embedding_dim)
                - Array of timestamps in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return None, None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Video properties: {frame_count} frames at {fps} fps")
        logging.info(f"Processing every {sample_interval} frames")
        
        # Initialize lists to store features and timestamps
        features = []
        timestamps = []
        
        # Process frames
        frame_idx = 0
        valid_frames = 0
        zero_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame
            if frame_idx % sample_interval == 0:
                # Extract features
                embedding = self.extract_features(frame)
                features.append(embedding)
                timestamps.append(frame_idx / fps)
                
                # Track statistics
                if np.any(embedding != 0):
                    valid_frames += 1
                else:
                    zero_frames += 1
            
            frame_idx += 1
            
            # Print progress every 100 frames
            if frame_idx % 100 == 0:
                logging.info(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        # Convert to numpy arrays
        features_array = np.array(features)
        timestamps_array = np.array(timestamps)
        
        # Log statistics
        total_processed = valid_frames + zero_frames
        if total_processed > 0:
            zero_percentage = (zero_frames / total_processed) * 100
            logging.info(f"Processed {total_processed} frames")
            logging.info(f"Frames with detected faces: {valid_frames} ({100 - zero_percentage:.1f}%)")
            logging.info(f"Frames with no faces: {zero_frames} ({zero_percentage:.1f}%)")
        
        return features_array, timestamps_array


# Simple test function
def test_on_image(image_path):
    """Test the FaceNetExtractor on a single image."""
    # Initialize extractor
    extractor = FaceNetExtractor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Extract features
    features = extractor.extract_features(image)
    
    # Print statistics
    non_zero = np.count_nonzero(features)
    total = features.shape[0]
    print(f"Feature vector shape: {features.shape}")
    print(f"Non-zero elements: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
    print(f"Min: {np.min(features):.4f}, Max: {np.max(features):.4f}")
    print(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FaceNetExtractor on an image or video")
    parser.add_argument("--image", help="Path to an image file for testing")
    parser.add_argument("--video", help="Path to a video file for testing")
    
    args = parser.parse_args()
    
    if args.image:
        test_on_image(args.image)
    elif args.video:
        extractor = FaceNetExtractor()
        features, timestamps = extractor.process_video(args.video)
        
        if features is not None:
            print(f"Extracted features for {len(features)} frames")
            print(f"Feature shape: {features.shape}")
            print(f"Non-zero frames: {np.sum(np.any(features != 0, axis=1))}/{len(features)}")
    else:
        print("Please provide either --image or --video argument")
