#!/usr/bin/env python3
# coding: utf-8
"""
3D CNN Feature Extractor for emotion recognition.
This module implements spatio-temporal feature extraction using 3D CNNs.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNN3DFeatureExtractor:
    """
    Extracts spatio-temporal features from facial videos using a 3D CNN.
    This captures both spatial and temporal dynamics for emotion recognition.
    """

    def __init__(self, input_shape=(16, 112, 112, 3), feature_dim=256,
                 weights_path=None, face_detector_path=None):
        """
        Initialize the 3D CNN feature extractor.

        Args:
            input_shape: Input shape (frames, height, width, channels)
            feature_dim: Dimension of the extracted features
            weights_path: Path to pre-trained weights (optional)
            face_detector_path: Path to face detector model (optional)
        """
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.model = self._build_model()

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded pre-trained weights from {weights_path}")

        # Initialize face detector if provided
        self.face_detector = None
        if face_detector_path and os.path.exists(face_detector_path):
            self.face_detector = cv2.FaceDetectorYN.create(
                face_detector_path, "", (0, 0))

    def _build_model(self):
        """
        Build the 3D CNN model for feature extraction.

        Returns:
            model: Keras model for feature extraction
        """
        inputs = Input(shape=self.input_shape)

        # Block 1
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        x = BatchNormalization()(x)
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool1')(x)

        # Block 2
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv2_2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool2')(x)

        # Block 3
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv3_1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv3_2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool3')(x)

        # Block 4
        x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv4_1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv4_2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool4')(x)

        # Flatten and FC layers
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        features = Dense(self.feature_dim, activation='relu', name='features')(x)

        # For full classification model we would add:
        # output = Dense(6, activation='softmax', name='predictions')(features)

        # Create model for feature extraction
        model = Model(inputs=inputs, outputs=features, name='CNN3D_Features')

        return model

    def extract_features(self, video_path, face_crop=True,
                         normalize_frames=True, save_visualization=False,
                         output_dir=None):
        """
        Extract features from a video file.

        Args:
            video_path: Path to video file
            face_crop: Whether to crop faces from frames
            normalize_frames: Whether to normalize pixel values
            save_visualization: Whether to save visualization of processed frames
            output_dir: Directory to save visualizations

        Returns:
            features: Extracted features (1D numpy array)
        """
        # Get frames from video
        frames = self._extract_frames_from_video(video_path)

        if frames is None or len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        # Preprocess frames
        processed_frames = self._preprocess_frames(frames, face_crop, normalize_frames)

        if processed_frames.shape[0] < self.input_shape[0]:
            # Handle case with fewer frames than expected
            # Pad with zeros or duplicate frames
            padding = np.zeros((self.input_shape[0] - processed_frames.shape[0],
                               *processed_frames.shape[1:]))
            processed_frames = np.vstack([processed_frames, padding])

        # Ensure correct shape for model input
        if len(processed_frames.shape) == 3:
            # Add channel dimension if grayscale
            processed_frames = np.expand_dims(processed_frames, axis=-1)

        # Reshape to model input shape
        model_input = processed_frames[:self.input_shape[0]]
        model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension

        # Save visualization if requested
        if save_visualization and output_dir:
            self._save_visualization(model_input[0], video_path, output_dir)

        # Extract features
        features = self.model.predict(model_input)[0]

        return features

    def _extract_frames_from_video(self, video_path, max_frames=None, fps=None):
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            fps: Frames per second to extract (None for all frames)

        Returns:
            frames: List of extracted frames
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        # Get video FPS for frame skipping
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = 1
        if fps and fps < video_fps:
            skip_frames = int(video_fps / fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                # Convert to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        return np.array(frames)

    def _preprocess_frames(self, frames, face_crop=True, normalize=True):
        """
        Preprocess video frames for the 3D CNN.

        Args:
            frames: List of video frames
            face_crop: Whether to crop faces from frames
            normalize: Whether to normalize pixel values

        Returns:
            processed_frames: Preprocessed frames
        """
        processed_frames = []

        for frame in frames:
            processed = frame.copy()

            # Detect and crop face if requested
            if face_crop and self.face_detector:
                processed = self._crop_face(processed)

            # Resize to expected input dimensions
            processed = cv2.resize(processed, (self.input_shape[1], self.input_shape[2]))

            # Normalize pixel values
            if normalize:
                processed = processed / 255.0

            processed_frames.append(processed)

        return np.array(processed_frames)

    def _crop_face(self, frame):
        """
        Detect and crop face from frame.

        Args:
            frame: Input frame

        Returns:
            face_img: Cropped face or original frame if no face detected
        """
        if self.face_detector is None:
            return frame

        # Set input size for detector
        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))

        # Detect faces
        _, faces = self.face_detector.detect(frame)

        if faces is None or len(faces) == 0:
            return frame

        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])

        # Extract bounding box
        x, y, w, h = largest_face[:4].astype(int)

        # Add margin
        margin = int(max(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)

        # Crop face
        face_img = frame[y:y+h, x:x+w]

        return face_img

    def _save_visualization(self, frames, video_path, output_dir):
        """
        Save visualization of processed frames.

        Args:
            frames: Processed frames
            video_path: Original video path
            output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create filename based on video path
        base_name = os.path.basename(video_path)
        base_name = os.path.splitext(base_name)[0]

        # Create grid of frames
        num_frames = min(frames.shape[0], 16)  # Show up to 16 frames
        grid_size = int(np.ceil(np.sqrt(num_frames)))

        plt.figure(figsize=(12, 12))
        for i in range(num_frames):
            plt.subplot(grid_size, grid_size, i + 1)

            # Handle RGB vs grayscale
            if frames.shape[-1] == 1:
                plt.imshow(frames[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(frames[i])

            plt.title(f"Frame {i+1}")
            plt.axis('off')

        plt.suptitle(f"Processed Frames: {base_name}")
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, f"{base_name}_frames.png"))
        plt.close()

    def batch_extract_features(self, video_paths, output_dir, batch_size=32):
        """
        Extract features from multiple videos and save to disk.

        Args:
            video_paths: List of video file paths
            output_dir: Directory to save extracted features
            batch_size: Batch size for prediction

        Returns:
            results: Dictionary mapping video paths to success/failure
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        for i, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
            try:
                # Extract base filename
                base_name = os.path.basename(video_path)
                base_name = os.path.splitext(base_name)[0]

                # Skip if already processed
                feature_path = os.path.join(output_dir, f"{base_name}.npy")
                if os.path.exists(feature_path):
                    results[video_path] = "Skipped (already exists)"
                    continue

                # Extract features
                features = self.extract_features(video_path)

                if features is None:
                    results[video_path] = "Failed (no features extracted)"
                    continue

                # Save features
                np.save(feature_path, features)
                results[video_path] = "Success"

                # Save visualization for a few samples
                if i % 100 == 0:
                    frames = self._extract_frames_from_video(video_path, max_frames=16)
                    if frames is not None:
                        processed_frames = self._preprocess_frames(frames)
                        self._save_visualization(processed_frames, video_path,
                                               os.path.join(output_dir, "visualizations"))

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                results[video_path] = f"Error: {str(e)}"

        # Save summary
        with open(os.path.join(output_dir, "extraction_results.txt"), "w") as f:
            for path, result in results.items():
                f.write(f"{path}: {result}\n")

        return results


def test_feature_extractor():
    """Test the 3D CNN feature extractor."""
    # Create a feature extractor
    extractor = CNN3DFeatureExtractor()

    # Sample video path
    video_path = "../downsampled_videos/RAVDESS/Actor_01/01-01-06-01-02-01-01.mp4"

    if os.path.exists(video_path):
        # Extract features
        features = extractor.extract_features(video_path, save_visualization=True,
                                           output_dir="./processed_data/visualizations")

        if features is not None:
            print(f"Extracted features shape: {features.shape}")

            # Save features
            os.makedirs("./processed_data/features", exist_ok=True)
            np.save("./processed_data/features/sample.npy", features)
    else:
        print(f"Sample video not found: {video_path}")

        # Try to find any video file
        import glob
        video_files = glob.glob("../downsampled_videos/RAVDESS/**/*.mp4", recursive=True)
        if video_files:
            print(f"Found {len(video_files)} video files. Testing with {video_files[0]}")
            features = extractor.extract_features(video_files[0], save_visualization=True,
                                               output_dir="./processed_data/visualizations")

            if features is not None:
                print(f"Extracted features shape: {features.shape}")


if __name__ == "__main__":
    test_feature_extractor()
