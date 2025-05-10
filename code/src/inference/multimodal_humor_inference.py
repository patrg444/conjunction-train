#!/usr/bin/env python
# Inference script for multimodal humor detection
# This script demonstrates how to use the trained fusion model for inference

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel, HubertModel, Wav2Vec2Processor
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import MultimodalFusionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HumorDetector:
    """Class for multimodal humor detection using the trained fusion model."""
    
    def __init__(self, model_dir, config_path='configs/model_checkpoint_paths.yaml'):
        """
        Initialize the humor detector.
        
        Args:
            model_dir (str): Directory containing trained model and config
            config_path (str): Path to model checkpoints config
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model config
        model_config_path = os.path.join(model_dir, 'model_config.yaml')
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        # Load global config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = self._load_model(model_dir)
        
        # Load feature extractors
        self.text_model, self.text_tokenizer = self._load_text_model()
        self.audio_model, self.audio_processor = self._load_audio_model()
        self.video_model = self._load_video_model()
    
    def _load_model(self, model_dir):
        """Load the trained fusion model."""
        # Create model with config
        model = MultimodalFusionModel(**self.model_config)
        
        # Load weights
        model_path = os.path.join(model_dir, 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_text_model(self):
        """Load XLM-RoBERTa model for text embedding extraction."""
        model_path = self.config['checkpoints']['xlm_roberta_v3']['model_path']
        
        logger.info(f"Loading XLM-RoBERTa from {model_path}")
        model = XLMRobertaModel.from_pretrained(model_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        
        model.eval()
        model.to(self.device)
        
        return model, tokenizer
    
    def _load_audio_model(self):
        """Load HuBERT model for audio embedding extraction."""
        if 'model_path' in self.config['checkpoints']['hubert_laughter']:
            model_path = self.config['checkpoints']['hubert_laughter']['model_path']
            logger.info(f"Loading custom HuBERT from {model_path}")
            model = HubertModel.from_pretrained(model_path)
            processor = Wav2Vec2Processor.from_pretrained(model_path)
        else:
            base_model = self.config['checkpoints']['hubert_laughter']['base_model']
            logger.info(f"Loading base HuBERT from {base_model}")
            model = HubertModel.from_pretrained(base_model)
            processor = Wav2Vec2Processor.from_pretrained(base_model)
        
        model.eval()
        model.to(self.device)
        
        return model, processor
    
    def _load_video_model(self):
        """Load ResNet18 model for smile detection."""
        model_path = self.config['checkpoints']['smile_resnet18']['model_path']
        
        logger.info(f"Loading Smile ResNet18 from {model_path}")
        model = torch.load(model_path, map_location=self.device)
        
        if hasattr(model, 'state_dict'):
            model = model.model if hasattr(model, 'model') else model
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def extract_text_embedding(self, text, max_length=512):
        """Extract embedding from text using XLM-RoBERTa."""
        inputs = self.text_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use the [CLS] token embedding as the sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def extract_audio_embedding(self, audio_path, sample_rate=16000, max_duration=10):
        """Extract embedding from audio using HuBERT."""
        try:
            # Load audio file
            waveform, orig_sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sample_rate=orig_sr, new_sample_rate=sample_rate)
                waveform = resampler(waveform)
            
            # Convert stereo to mono by averaging channels if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Get number of samples for max_duration
            max_samples = max_duration * sample_rate
            
            # Pad or truncate
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                padding = max_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            # Process input with the model
            inputs = self.audio_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_model(input_values)
                # Use the mean of hidden states as the audio embedding
                embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()[0]
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {str(e)}")
            return np.zeros(self.audio_model.config.hidden_size, dtype=np.float32)
    
    def extract_video_embedding(self, video_path, full_vector=False):
        """Extract smile probability from video frames or image directory."""
        try:
            # Check if video_path is a directory (containing frames) or a file
            if os.path.isdir(video_path):
                # Directory of frames
                frame_files = sorted([f for f in os.listdir(video_path) 
                                    if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                if not frame_files:
                    logger.warning(f"No frames found in {video_path}")
                    if full_vector:
                        return np.zeros(512, dtype=np.float32)
                    else:
                        return np.array([0.0], dtype=np.float32)
                
                # Process the middle frame
                middle_idx = len(frame_files) // 2
                image_path = os.path.join(video_path, frame_files[middle_idx])
                
                # Load and preprocess image
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
                
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(self.device)
                    
                with torch.no_grad():
                    # Forward pass through the model
                    features = self.video_model.backbone(image)
                    if full_vector:
                        # Return the penultimate layer features
                        embedding = features.cpu().numpy()[0]
                    else:
                        # Get smile probability
                        output = self.video_model(image)
                        if isinstance(output, tuple):
                            output = output[0]
                        prob = torch.sigmoid(output).cpu().numpy()[0]
                        embedding = prob
                    
                return embedding
            
            else:
                # Single image file
                if os.path.isfile(video_path) and video_path.endswith(('.jpg', '.jpeg', '.png')):
                    # Load and preprocess image
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
                    ])
                    
                    image = Image.open(video_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(self.device)
                        
                    with torch.no_grad():
                        # Forward pass through the model
                        features = self.video_model.backbone(image)
                        if full_vector:
                            # Return the penultimate layer features
                            embedding = features.cpu().numpy()[0]
                        else:
                            # Get smile probability
                            output = self.video_model(image)
                            if isinstance(output, tuple):
                                output = output[0]
                            prob = torch.sigmoid(output).cpu().numpy()[0]
                            embedding = prob
                        
                    return embedding
                else:
                    # Video file - not supported in this simple version
                    logger.warning(f"Video file processing not implemented for {video_path}")
                    if full_vector:
                        return np.zeros(512, dtype=np.float32)
                    else:
                        return np.array([0.0], dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            if full_vector:
                return np.zeros(512, dtype=np.float32)
            else:
                return np.array([0.0], dtype=np.float32)
    
    def predict(self, text=None, audio_path=None, video_path=None, embeddings=None):
        """
        Predict humor from multimodal inputs.
        
        Args:
            text (str, optional): Text input
            audio_path (str, optional): Path to audio file
            video_path (str, optional): Path to video file or frame directory
            embeddings (dict, optional): Pre-extracted embeddings
            
        Returns:
            dict: Prediction results with probability and class
        """
        # Extract embeddings if not provided
        if embeddings is None:
            embeddings = {}
            
            if text is not None:
                embeddings['text'] = self.extract_text_embedding(text)
            else:
                embeddings['text'] = np.zeros(self.model_config['text_dim'], dtype=np.float32)
                
            if audio_path is not None:
                embeddings['audio'] = self.extract_audio_embedding(audio_path)
            else:
                embeddings['audio'] = np.zeros(self.model_config['audio_dim'], dtype=np.float32)
                
            if video_path is not None:
                embeddings['video'] = self.extract_video_embedding(video_path)
            else:
                embeddings['video'] = np.zeros(self.model_config['video_dim'], dtype=np.float32)
        
        # Convert embeddings to tensors
        text_embedding = torch.FloatTensor(embeddings['text']).unsqueeze(0).to(self.device)
        audio_embedding = torch.FloatTensor(embeddings['audio']).unsqueeze(0).to(self.device)
        video_embedding = torch.FloatTensor(embeddings['video']).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(text_embedding, audio_embedding, video_embedding)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            humor_prob = probabilities[0, 1].item()  # Probability of humor class
        
        return {
            'prediction': prediction,  # 0: not humor, 1: humor
            'humor_probability': humor_prob,
            'is_humor': prediction == 1,
            'confidence': probabilities[0, prediction].item()
        }
    
    def visualize_prediction(self, result, text=None, audio_path=None, video_path=None):
        """
        Visualize the humor prediction result.
        
        Args:
            result (dict): Prediction result from predict()
            text (str, optional): Text input for display
            audio_path (str, optional): Path to audio file for display
            video_path (str, optional): Path to video file or frame for display
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot humor probability
        axes[0].bar(['Not Humor', 'Humor'], [1 - result['humor_probability'], result['humor_probability']])
        axes[0].set_ylim([0, 1])
        axes[0].set_title('Humor Probability')
        axes[0].set_ylabel('Probability')
        
        # Display image if available
        if video_path is not None and os.path.exists(video_path):
            if os.path.isdir(video_path):
                # Find middle frame in directory
                frame_files = sorted([f for f in os.listdir(video_path) 
                                     if f.endswith(('.jpg', '.jpeg', '.png'))])
                if frame_files:
                    middle_idx = len(frame_files) // 2
                    image_path = os.path.join(video_path, frame_files[middle_idx])
                    img = plt.imread(image_path)
                    axes[1].imshow(img)
                    axes[1].set_title('Video Frame')
                    axes[1].axis('off')
            elif video_path.endswith(('.jpg', '.jpeg', '.png')):
                img = plt.imread(video_path)
                axes[1].imshow(img)
                axes[1].set_title('Image')
                axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No image available', ha='center', va='center')
            axes[1].axis('off')
        
        # Display text if available
        if text is not None:
            axes[2].text(0.5, 0.5, text, wrap=True, ha='center', va='center')
            axes[2].set_title('Text Input')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'No text available', ha='center', va='center')
            axes[2].axis('off')
        
        # Add overall result
        if result['is_humor']:
            result_text = f"HUMOR (Confidence: {result['confidence']:.2f})"
            color = 'green'
        else:
            result_text = f"NOT HUMOR (Confidence: {result['confidence']:.2f})"
            color = 'red'
        
        plt.suptitle(result_text, fontsize=16, color=color)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal humor detection inference')
    parser.add_argument('--model_dir', type=str, default='training_logs_humor/multimodal_attention_fusion/final_model',
                        help='Directory containing the trained model')
    parser.add_argument('--config', type=str, default='configs/model_checkpoint_paths.yaml',
                        help='Path to model checkpoints config')
    parser.add_argument('--text', type=str, help='Text input for humor detection')
    parser.add_argument('--audio', type=str, help='Path to audio file for humor detection')
    parser.add_argument('--video', type=str, help='Path to video file/frames for humor detection')
    parser.add_argument('--output', type=str, help='Path to save visualization output')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check inputs
    if not args.text and not args.audio and not args.video:
        logger.error("Please provide at least one input (--text, --audio, or --video)")
        return
    
    # Initialize humor detector
    detector = HumorDetector(args.model_dir, args.config)
    
    # Run prediction
    result = detector.predict(
        text=args.text,
        audio_path=args.audio,
        video_path=args.video
    )
    
    # Print result
    print(f"\nHumor Detection Result:")
    print(f"Prediction: {'HUMOR' if result['is_humor'] else 'NOT HUMOR'}")
    print(f"Humor Probability: {result['humor_probability']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Visualize result
    fig = detector.visualize_prediction(result, args.text, args.audio, args.video)
    
    # Save or display visualization
    if args.output:
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
