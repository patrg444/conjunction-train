#!/usr/bin/env python3
# coding: utf-8
"""
Multi-Region Attention Model Feature Extractor for emotion recognition.
This module implements a region-based attention mechanism for facial emotion features.
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
import dlib
from pathlib import Path

# Add the parent directory to the path so we can import from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MultiRegionAttentionExtractor:
    """
    Extracts facial features using a multi-region attention mechanism.

    This model divides the face into regions (eyes, nose, mouth, etc.),
    processes each region separately, and uses an attention mechanism to
    weight the importance of each region for emotion recognition.
    """

    def __init__(self, input_shape=(112, 112, 3), feature_dim=256,
                 backbone='vgg16', use_pretrained=True,
                 face_detector_path=None, landmark_predictor_path=None):
        """
        Initialize the multi-region attention feature extractor.

        Args:
            input_shape: Input shape for each region (height, width, channels)
            feature_dim: Dimension of the extracted features
            backbone: Feature extraction backbone ('vgg16' or 'resnet50')
            use_pretrained: Whether to use pretrained weights for the backbone
            face_detector_path: Path to face detector model (optional)
            landmark_predictor_path: Path to facial landmark predictor model (optional)
        """
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.backbone_name = backbone
        self.use_pretrained = use_pretrained

        # Define facial regions
        self.regions = {
            'left_eye': {'landmarks': list(range(36, 42))},
            'right_eye': {'landmarks': list(range(42, 48))},
            'nose': {'landmarks': list(range(27, 36))},
            'mouth': {'landmarks': list(range(48, 68))},
            'left_brow': {'landmarks': list(range(17, 22))},
            'right_brow': {'landmarks': list(range(22, 27))},
            'jawline': {'landmarks': list(range(0, 17))}
        }

        # Initialize face detector and landmark predictor
        self.face_detector = None
        self.landmark_predictor = None

        # Try to use dlib's face detector and predictor if paths are provided
        if face_detector_path and os.path.exists(face_detector_path):
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                print("Using dlib face detector")
            except Exception as e:
                print(f"Error initializing dlib face detector: {e}")

        if landmark_predictor_path and os.path.exists(landmark_predictor_path):
            try:
                self.landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
                print(f"Loaded landmark predictor from {landmark_predictor_path}")
            except Exception as e:
                print(f"Error loading landmark predictor: {e}")

        # If dlib models are not available use OpenCV's face detector
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Using OpenCV face detector")

            # We can't use regions properly without landmarks so we'll
            # use a simpler approach in this case
            if self.landmark_predictor is None:
                print("WARNING: No landmark predictor available. Will use simple grid-based regions.")

        # Build models
        self.region_models = self._build_region_models()
        self.attention_model = self._build_attention_model()

    def _build_backbone(self):
        """
        Build the backbone CNN for feature extraction.

        Returns:
            model: Keras model for the backbone
        """
        weights = 'imagenet' if self.use_pretrained else None

        if self.backbone_name == 'vgg16':
            # Use VGG16 without top layers, smaller input size for regions
            backbone = VGG16(include_top=False, weights=weights,
                          input_shape=self.input_shape)
        elif self.backbone_name == 'resnet50':
            # Use ResNet50 without top layers
            backbone = ResNet50(include_top=False, weights=weights,
                             input_shape=self.input_shape)
        else:
            # Custom CNN backbone
            inputs = Input(shape=self.input_shape)

            # Simple CNN architecture
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            backbone = Model(inputs=inputs, outputs=x, name='custom_backbone')

        # Freeze early layers if using pretrained weights
        if self.use_pretrained:
            for layer in backbone.layers[:10]:
                layer.trainable = False

        return backbone

    def _build_region_models(self):
        """
        Build models for each facial region.

        Returns:
            region_models: Dictionary of models for each region
        """
        backbone = self._build_backbone()
        region_models = {}

        for region_name in self.regions.keys():
            # Create input
            inputs = Input(shape=self.input_shape, name=f"{region_name}_input")

            # Use the same backbone for all regions (shared weights)
            features = backbone(inputs)

            # Add region-specific layers
            x = GlobalAveragePooling2D()(features)
            x = Dense(128, activation='relu', name=f"{region_name}_fc")(x)
            outputs = Dense(64, activation='relu', name=f"{region_name}_features")(x)

            # Create model
            region_models[region_name] = Model(
                inputs=inputs,
                outputs=outputs,
                name=f"{region_name}_model"
            )

        return region_models

    def _build_attention_model(self):
        """
        Build the attention model to combine region features.

        Returns:
            model: Keras model for the attention mechanism
        """
        # Create inputs for each region
        inputs = {
            region_name: Input(shape=(64,), name=f"{region_name}_features_input")
            for region_name in self.regions.keys()
        }

        # Concatenate all region features
        all_features = tf.stack(list(inputs.values()), axis=1)  # [batch, n_regions, features]

        # Self-attention mechanism
        query = Dense(32, activation='relu')(all_features)
        key = Dense(32, activation='relu')(all_features)
        value = Dense(64, activation='relu')(all_features)

        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Apply attention weights
        context_vector = tf.matmul(attention_weights, value)

        # Flatten and combine
        context_flat = Flatten()(context_vector)
        x = Dense(256, activation='relu')(context_flat)
        outputs = Dense(self.feature_dim, activation='relu', name='attention_features')(x)

        # Create model
        return Model(
            inputs=list(inputs.values()),
            outputs=outputs,
            name='attention_model'
        )

    def extract_features(self, video_path, save_visualization=False,
                         output_dir=None, max_frames=None):
        """
        Extract features from a video file.

        Args:
            video_path: Path to video file
            save_visualization: Whether to save visualization of region attention
            output_dir: Directory to save visualizations
            max_frames: Maximum number of frames to process

        Returns:
            features: Extracted features (1D numpy array)
        """
        # Get frames from video
        frames = self._extract_frames_from_video(video_path, max_frames)

        if frames is None or len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        # Process each frame
        all_features = []
        for i, frame in enumerate(tqdm(frames, desc="Processing frames", leave=False)):
            # Extract facial landmarks
            landmarks = self._extract_landmarks(frame)

            if landmarks is None:
                continue

            # Extract regions based on landmarks
            regions = self._extract_regions(frame, landmarks)

            # Extract features for each region
            region_features = {}
            for region_name, region_img in regions.items():
                if region_img is None:
                    continue

                # Preprocess region image
                region_img = self._preprocess_image(region_img)

                # Extract features using the region model
                region_features[region_name] = self.region_models[region_name].predict(
                    np.expand_dims(region_img, axis=0), verbose=0
                )[0]

            # Skip if any region is missing
            if len(region_features) != len(self.regions):
                continue

            # Apply attention mechanism to combine region features
            try:
                combined_features = self.attention_model.predict(
                    [region_features[region] for region in self.regions.keys()], verbose=0
                )[0]

                all_features.append(combined_features)

                # Save visualization if requested
                if save_visualization and output_dir and i % 10 == 0:
                    self._visualize_regions(frame, regions, region_features,
                                         os.path.basename(video_path), i, output_dir)

            except Exception as e:
                print(f"Error in attention model: {e}")
                continue

        if not all_features:
            print(f"No features extracted from {video_path}")
            return None

        # Aggregate features across frames (average)
        features = np.mean(all_features, axis=0)

        return features

    def _extract_frames_from_video(self, video_path, max_frames=None):
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract

        Returns:
            frames: List of extracted frames
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        if len(frames) == 0:
            print(f"No frames extracted from {video_path}")
            return None

        return frames

    def _extract_landmarks(self, frame):
        """
        Extract facial landmarks from a frame.

        Args:
            frame: Input frame

        Returns:
            landmarks: List of facial landmark coordinates
        """
        # Use dlib if available
        if isinstance(self.face_detector, dlib.fhog_object_detector) and self.landmark_predictor:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_detector(gray)
            if not faces:
                return None

            # Get largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # Get facial landmarks
            landmarks = self.landmark_predictor(gray, face)

            # Convert to list of (x, y) coordinates
            landmarks = [(p.x, p.y) for p in landmarks.parts()]

            return landmarks

        else:
            # Using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return None

            # Get largest face
            face = max(faces, key=lambda rect: rect[2] * rect[3])

            # Without a landmark predictor we'll create a simple grid
            # This is a very basic approximation of facial landmarks
            x, y, w, h = face

            # Create a 5x5 grid of points over the face
            landmarks = []
            for i in range(5):
                for j in range(5):
                    landmarks.append((
                        x + j * w // 4,
                        y + i * h // 4
                    ))

            # Add extra points for approximate eye, nose, and mouth positions
            landmarks.extend([
                (x + w // 3, y + h // 3),      # Left eye
                (x + 2 * w // 3, y + h // 3),  # Right eye
                (x + w // 2, y + h // 2),      # Nose
                (x + w // 3, y + 2 * h // 3),  # Left mouth
                (x + w // 2, y + 2 * h // 3),  # Center mouth
                (x + 2 * w // 3, y + 2 * h // 3)  # Right mouth
            ])

            return landmarks

    def _extract_regions(self, frame, landmarks):
        """
        Extract facial regions based on landmarks.

        Args:
            frame: Input frame
            landmarks: Facial landmarks

        Returns:
            regions: Dictionary of facial region images
        """
        h, w = frame.shape[:2]

        # If we have the standard 68 landmarks from dlib
        if len(landmarks) == 68:
            # Process each defined region
            regions = {}
            for region_name, region_info in self.regions.items():
                # Get landmarks for this region
                region_landmarks = [landmarks[i] for i in region_info['landmarks']]

                # Calculate bounding box
                min_x = max(0, int(min(l[0] for l in region_landmarks)) - 5)
                min_y = max(0, int(min(l[1] for l in region_landmarks)) - 5)
                max_x = min(w, int(max(l[0] for l in region_landmarks)) + 5)
                max_y = min(h, int(max(l[1] for l in region_landmarks)) + 5)

                # Ensure minimum size
                if max_x - min_x < 10 or max_y - min_y < 10:
                    regions[region_name] = None
                    continue

                # Crop region
                region_img = frame[min_y:max_y, min_x:max_x]

                # Resize to input shape
                region_img = cv2.resize(region_img, (self.input_shape[0], self.input_shape[1]))

                regions[region_name] = region_img

        else:
            # For a grid or approximate landmarks, create approximate regions
            # This is a fallback when we don't have the standard 68 landmarks

            # Find face bounding box
            if len(landmarks) > 0:
                min_x = max(0, int(min(l[0] for l in landmarks)) - 5)
                min_y = max(0, int(min(l[1] for l in landmarks)) - 5)
                max_x = min(w, int(max(l[0] for l in landmarks)) + 5)
                max_y = min(h, int(max(l[1] for l in landmarks)) + 5)

                face_w = max_x - min_x
                face_h = max_y - min_y

                # Create approximate regions
                regions = {
                    'left_eye': frame[
                        min_y + face_h // 5:min_y + 2 * face_h // 5,
                        min_x + face_w // 5:min_x + 2 * face_w // 5
                    ],
                    'right_eye': frame[
                        min_y + face_h // 5:min_y + 2 * face_h // 5,
                        min_x + 3 * face_w // 5:min_x + 4 * face_w // 5
                    ],
                    'nose': frame[
                        min_y + 2 * face_h // 5:min_y + 3 * face_h // 5,
                        min_x + 2 * face_w // 5:min_x + 3 * face_w // 5
                    ],
                    'mouth': frame[
                        min_y + 3 * face_h // 5:min_y + 4 * face_h // 5,
                        min_x + face_w // 5:min_x + 4 * face_w // 5
                    ],
                    'left_brow': frame[
                        min_y + face_h // 10:min_y + face_h // 5,
                        min_x + face_w // 5:min_x + 2 * face_w // 5
                    ],
                    'right_brow': frame[
                        min_y + face_h // 10:min_y + face_h // 5,
                        min_x + 3 * face_w // 5:min_x + 4 * face_w // 5
                    ],
                    'jawline': frame[
                        min_y + 4 * face_h // 5:max_y,
                        min_x:max_x
                    ]
                }

                # Resize all regions
                for region_name, region_img in regions.items():
                    if region_img.size == 0:
                        regions[region_name] = None
                        continue
                    regions[region_name] = cv2.resize(
                        region_img, (self.input_shape[0], self.input_shape[1])
                    )
            else:
                # No valid landmarks
                regions = {region_name: None for region_name in self.regions.keys()}

        return regions

    def _preprocess_image(self, image):
        """
        Preprocess an image for the CNN.

        Args:
            image: Input image

        Returns:
            processed_img: Preprocessed image
        """
        # Resize if necessary
        if image.shape[:2] != (self.input_shape[0], self.input_shape[1]):
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        return image

    def _visualize_regions(self, frame, regions, region_features, video_name, frame_idx, output_dir):
        """
        Visualize facial regions and their features.

        Args:
            frame: Original frame
            regions: Dictionary of facial region images
            region_features: Dictionary of features for each region
            video_name: Name of the source video
            frame_idx: Frame index
            output_dir: Output directory
        """
        # Create output directory
        viz_dir = os.path.join(output_dir, 'region_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Create a figure with subplots
        n_regions = len(regions)
        n_cols = 3
        n_rows = (n_regions + 2) // n_cols

        plt.figure(figsize=(15, 5 * n_rows))

        # Plot original frame
        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(frame)
        plt.title("Original Frame")
        plt.axis('off')

        # Plot each region
        for i, (region_name, region_img) in enumerate(regions.items(), 2):
            plt.subplot(n_rows, n_cols, i)
            if region_img is not None:
                plt.imshow(region_img)

                # Add average feature value as text
                if region_name in region_features:
                    avg_feature = np.mean(region_features[region_name])
                    plt.text(5, 15, f"Avg: {avg_feature:.2f}",
                           color='white', fontsize=10,
                           bbox=dict(facecolor='black', alpha=0.5))
            else:
                plt.text(0.5, 0.5, "Region not available",
                       horizontalalignment='center', verticalalignment='center')

            plt.title(region_name)
            plt.axis('off')

        # Save figure
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{base_name}_frame{frame_idx:04d}_regions.png"))
        plt.close()

    def batch_extract_features(self, video_paths, output_dir, batch_size=32,
                              visualize=False, resume=False):
        """
        Extract features from multiple videos and save to disk.

        Args:
            video_paths: List of video file paths
            output_dir: Directory to save extracted features
            batch_size: Batch size for processing
            visualize: Whether to save visualizations
            resume: Whether to skip already processed files

        Returns:
            results: Dictionary mapping video paths to success/failure
        """
        features_dir = os.path.join(output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        if visualize:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)

        # Track successes, failures, and skips
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        failures = []

        # Process videos
        for i, video_path in enumerate(tqdm(video_paths, desc="Extracting multi-region features")):
            try:
                # Create output filename
                base_name = os.path.basename(video_path)
                base_name = os.path.splitext(base_name)[0]
                output_path = os.path.join(features_dir, f"{base_name}.npy")

                # Skip if file exists and resume is True
                if resume and os.path.exists(output_path):
                    results['skipped'] += 1
                    continue

                # Extract features
                features = self.extract_features(
                    video_path,
                    save_visualization=(visualize and i % 50 == 0),
                    output_dir=viz_dir if visualize else None
                )

                if features is not None:
                    # Save features
                    np.save(output_path, features)
                    results['success'] += 1

                    # Print progress periodically
                    if (i + 1) % 100 == 0:
                        print(f"Progress: {i+1}/{len(video_paths)} - "
                             f"Success: {results['success']} "
                             f"Failed: {results['failed']} "
                             f"Skipped: {results['skipped']}")
                else:
                    results['failed'] += 1
                    failures.append(video_path)

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                results['failed'] += 1
                failures.append(f"{video_path}: {str(e)}")

        # Save extraction summary
        summary_path = os.path.join(output_dir, "multiregion_extraction_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Multi-Region Feature Extraction Summary:\n")
            f.write(f"-------------------------------------\n")
            f.write(f"Total files: {len(video_paths)}\n")
            f.write(f"Successful extractions: {results['success']}\n")
            f.write(f"Failed extractions: {results['failed']}\n")
            f.write(f"Skipped files: {results['skipped']}\n\n")

            if failures:
                f.write("Failed Files:\n")
                f.write("------------\n")
                for failure in failures:
                    f.write(f"{failure}\n")

        print(f"\nMulti-region extraction complete!")
        print(f"Total: {len(video_paths)} Success: {results['success']} "
             f"Failed: {results['failed']} Skipped: {results['skipped']}")
        print(f"Summary saved to {summary_path}")

        return results


def test_feature_extractor():
    """Test the multi-region attention feature extractor."""
    # Create a feature extractor with a custom (smaller) backbone
    extractor = MultiRegionAttentionExtractor(
        backbone='custom',
        use_pretrained=False
    )

    # Sample video path
    video_path = "./downsampled_videos/RAVDESS/Actor_01/01-01-06-01-02-01-01.mp4"

    if os.path.exists(video_path):
        # Extract features
        features = extractor.extract_features(
            video_path,
            save_visualization=True,
            output_dir="./processed_data/multiregion_visualizations"
        )

        if features is not None:
            print(f"Extracted multi-region features shape: {features.shape}")

            # Save features
            os.makedirs("./processed_data/multiregion_features", exist_ok=True)
            np.save("./processed_data/multiregion_features/sample.npy", features)
    else:
        print(f"Sample video not found: {video_path}")

        # Try to find any video file
        import glob
        video_files = glob.glob("./downsampled_videos/RAVDESS/**/*.mp4", recursive=True)
        if video_files:
            print(f"Found {len(video_files)} video files. Testing with {video_files[0]}")
            features = extractor.extract_features(
                video_files[0],
                save_visualization=True,
                output_dir="./processed_data/multiregion_visualizations"
            )

            if features is not None:
                print(f"Extracted multi-region features shape: {features.shape}")


if __name__ == "__main__":
    test_feature_extractor()
