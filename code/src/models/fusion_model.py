#!/usr/bin/env python
# Multimodal Fusion Model for Humor Detection
# This module implements several fusion strategies for combining text, audio, and video modalities

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
import yaml
import numpy as np
import logging

logger = logging.getLogger(__name__)

print("\n\n>>> USING LOCAL PROJECT fusion_model.py <<<\n\n")

class AttentionLayer(nn.Module):
    """Cross-modal attention module for focusing on relevant features across modalities."""
    
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, output_dim)
        self.k_proj = nn.Linear(key_dim, output_dim)
        self.v_proj = nn.Linear(value_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        # Project inputs to multi-head queries, keys, and values
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output

class ModalityEncoder(nn.Module):
    """Encoder for a single modality that projects to a common representation space."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3,
                 is_sequential=False, seq_processor_type='lstm', seq_processor_hidden_dim=None,
                 num_seq_processor_layers=1): # Added num_seq_processor_layers
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_sequential = is_sequential
        self.seq_processor = None

        current_processing_dim = input_dim

        if self.is_sequential:
            if seq_processor_hidden_dim is None:
                # Default seq_processor_hidden_dim to be half of the main hidden_dim if not specified,
                # because bidirectional LSTM/GRU will double it.
                seq_processor_hidden_dim = hidden_dim // 2 if hidden_dim else 128 # Ensure it's not zero
                if seq_processor_hidden_dim == 0: seq_processor_hidden_dim = 64 # Fallback if hidden_dim is too small

            if seq_processor_type == 'lstm':
                self.seq_processor = nn.LSTM(input_dim, seq_processor_hidden_dim,
                                             num_layers=num_seq_processor_layers, # Use num_seq_processor_layers
                                             batch_first=True, bidirectional=True, dropout=dropout if num_seq_processor_layers > 1 else 0)
                current_processing_dim = seq_processor_hidden_dim * 2 # Bidirectional
            elif seq_processor_type == 'gru':
                self.seq_processor = nn.GRU(input_dim, seq_processor_hidden_dim,
                                            num_layers=num_seq_processor_layers, # Use num_seq_processor_layers
                                            batch_first=True, bidirectional=True, dropout=dropout if num_seq_processor_layers > 1 else 0)
                current_processing_dim = seq_processor_hidden_dim * 2 # Bidirectional
            # Add other types like 'conv1d' if needed in the future
            else:
                logger.warning(f"Unsupported seq_processor_type: {seq_processor_type}. Defaulting to mean pooling for sequential input.")
                self.is_sequential = False # Fallback to non-sequential path if type is unknown

        self.layers = nn.Sequential(
            nn.Linear(current_processing_dim, hidden_dim), # Input dim depends on seq_processor output or original input_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        if self.is_sequential and self.seq_processor:
            # x shape: (batch_size, seq_len, feature_dim)
            if isinstance(self.seq_processor, (nn.LSTM)):
                # LSTM output: (output, (hn, cn))
                # output shape: (batch_size, seq_len, num_directions * hidden_size)
                # hn shape: (num_layers * num_directions, batch_size, hidden_size)
                _, (hn, _) = self.seq_processor(x)
                # Concatenate hidden states of forward and backward LSTM from the last layer
                # hn is (num_layers*num_directions, batch, hidden_size)
                # We want the last layer's forward and backward hidden states
                x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
            elif isinstance(self.seq_processor, (nn.GRU)):
                # GRU output: (output, hn)
                # output shape: (batch_size, seq_len, num_directions * hidden_size)
                # hn shape: (num_layers * num_directions, batch_size, hidden_size)
                _, hn = self.seq_processor(x)
                x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        elif x.ndim == 3 and not (self.is_sequential and self.seq_processor): # Fallback mean pooling
            logger.debug(f"ModalityEncoder (input_dim={self.input_dim}) received 3D tensor but is_sequential=False or no seq_processor. Applying mean pooling.")
            x = torch.mean(x, dim=1)
        
        return self.layers(x)

class EarlyFusionModel(nn.Module):
    """
    Early fusion model that concatenates all modality embeddings and processes them together.
    Simple but effective approach to multimodal fusion.
    """
    def __init__(
        self,
        text_dim=1024,
        audio_dim=768,
        video_dim=1,
        hidden_dim=512,
        output_dim=128,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, output_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, output_dim, dropout)
        self.video_encoder = ModalityEncoder(32, hidden_dim // 4, output_dim, dropout)
        # Combined dimension after concatenation
        combined_dim = output_dim * 3
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, text_embedding, audio_embedding, video_embedding):
        # Encode each modality
        text_encoded = self.text_encoder(text_embedding)
        audio_encoded = self.audio_encoder(audio_embedding)
        video_encoded = self.video_encoder(video_embedding)
        # Concatenate all encodings
        combined = torch.cat([text_encoded, audio_encoded, video_encoded], dim=1)
        # Classify
        logits = self.classifier(combined)
        return logits

class LateFusionModel(nn.Module):
    """
    Late fusion model that processes each modality separately and combines predictions.
    Good for handling different modalities with different characteristics.
    """
    def __init__(
        self,
        text_dim=1024,
        audio_dim=768,
        video_dim=1,
        hidden_dim=512,
        output_dim=128,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        # Separate encoders for each modality
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, output_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, output_dim, dropout)
        self.video_encoder = ModalityEncoder(32, hidden_dim // 4, output_dim, dropout)
        # Separate classifiers for each modality
        self.text_classifier = nn.Linear(output_dim, num_classes)
        self.audio_classifier = nn.Linear(output_dim, num_classes)
        self.video_classifier = nn.Linear(output_dim, num_classes)
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
    def forward(self, text_embedding, audio_embedding, video_embedding):
        # Encode each modality
        text_encoded = self.text_encoder(text_embedding)
        audio_encoded = self.audio_encoder(audio_embedding)
        video_encoded = self.video_encoder(video_embedding)
        # Get predictions from each modality
        text_logits = self.text_classifier(text_encoded)
        audio_logits = self.audio_classifier(audio_encoded)
        video_logits = self.video_classifier(video_encoded)
        # Normalize fusion weights with softmax
        weights = F.softmax(self.fusion_weights, dim=0)
        # Weighted sum of logits
        combined_logits = (
            weights[0] * text_logits +
            weights[1] * audio_logits +
            weights[2] * video_logits
        )
        return combined_logits

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion model that uses attention to capture interactions between modalities.
    Most sophisticated approach that can learn complex cross-modal relationships.
    """
    def __init__(
        self,
        text_dim=1024,
        audio_dim=768,
        video_dim=512, # Corrected default video_dim
        hidden_dim=512,
        output_dim=128,
        num_classes=2,
        dropout=0.3,
        num_heads=8,
        # New parameters for sequential processing, to be passed from MultimodalFusionModel hparams
        audio_is_sequential=True,
        audio_seq_processor_type='lstm',
        audio_seq_processor_hidden_dim=None, # Will default in ModalityEncoder if None
        audio_num_seq_processor_layers=1,
        video_is_sequential=True,
        video_seq_processor_type='lstm',
        video_seq_processor_hidden_dim=None, # Will default in ModalityEncoder if None
        video_num_seq_processor_layers=1
    ):
        super().__init__()
        # Initial encoders for each modality using dimensions passed to CrossModalAttentionFusion
        # Text features are pre-pooled (e.g., context+punchline)
        self.text_encoder = ModalityEncoder(
            input_dim=text_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            dropout=dropout,
            is_sequential=False # Text is pre-pooled
        )
        # Audio features are sequential
        self.audio_encoder = ModalityEncoder(
            input_dim=audio_dim, # This will be per-frame feature dim for sequential audio
            hidden_dim=hidden_dim, # Hidden dim for the dense layers after LSTM/GRU
            output_dim=output_dim, # Final output dim for attention
            dropout=dropout,
            is_sequential=audio_is_sequential,
            seq_processor_type=audio_seq_processor_type,
            seq_processor_hidden_dim=audio_seq_processor_hidden_dim if audio_seq_processor_hidden_dim is not None else hidden_dim // 2,
            num_seq_processor_layers=audio_num_seq_processor_layers
        )
        # Video features are sequential (e.g., frame-level AUs)
        # For video_encoder, its internal dense layers will use output_dim as their hidden_dim
        self.video_encoder = ModalityEncoder(
            input_dim=video_dim, # This will be per-frame feature dim for sequential video
            hidden_dim=output_dim, # Hidden dim for the dense layers after LSTM/GRU (specific to video encoder in this setup)
            output_dim=output_dim, # Final output dim for attention
            dropout=dropout,
            is_sequential=video_is_sequential,
            seq_processor_type=video_seq_processor_type,
            seq_processor_hidden_dim=video_seq_processor_hidden_dim if video_seq_processor_hidden_dim is not None else output_dim // 2,
            num_seq_processor_layers=video_num_seq_processor_layers
        )
        # Cross-modal attention layers
        self.text_to_audio_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        self.text_to_video_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        self.audio_to_text_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        self.audio_to_video_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        self.video_to_text_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        self.video_to_audio_attn = AttentionLayer(output_dim, output_dim, output_dim, output_dim, num_heads)
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        # Classification head
        self.classifier = nn.Linear(output_dim, num_classes)
    def forward(self, text_embedding, audio_embedding, video_embedding):
        # Initial encoding
        text_encoded = self.text_encoder(text_embedding)
        audio_encoded = self.audio_encoder(audio_embedding)
        video_encoded = self.video_encoder(video_embedding)
        # Reshape for attention
        batch_size = text_encoded.shape[0]
        text_reshaped = text_encoded.view(batch_size, 1, -1)
        audio_reshaped = audio_encoded.view(batch_size, 1, -1)
        video_reshaped = video_encoded.view(batch_size, 1, -1)
        # Cross-modal attention
        text_attends_to_audio = self.text_to_audio_attn(text_reshaped, audio_reshaped, audio_reshaped)
        text_attends_to_video = self.text_to_video_attn(text_reshaped, video_reshaped, video_reshaped)
        audio_attends_to_text = self.audio_to_text_attn(audio_reshaped, text_reshaped, text_reshaped)
        audio_attends_to_video = self.audio_to_video_attn(audio_reshaped, video_reshaped, video_reshaped)
        video_attends_to_text = self.video_to_text_attn(video_reshaped, text_reshaped, text_reshaped)
        video_attends_to_audio = self.video_to_audio_attn(video_reshaped, audio_reshaped, audio_reshaped)
        # Reshape and concatenate attention outputs
        text_attends_to_audio = text_attends_to_audio.view(batch_size, -1)
        text_attends_to_video = text_attends_to_video.view(batch_size, -1)
        audio_attends_to_text = audio_attends_to_text.view(batch_size, -1)
        audio_attends_to_video = audio_attends_to_video.view(batch_size, -1)
        video_attends_to_text = video_attends_to_text.view(batch_size, -1)
        video_attends_to_audio = video_attends_to_audio.view(batch_size, -1)
        # Concatenate all cross-attended features
        cross_modal_features = torch.cat([
            text_attends_to_audio, text_attends_to_video,
            audio_attends_to_text, audio_attends_to_video,
            video_attends_to_text, video_attends_to_audio
        ], dim=1)
        # Fuse features
        fused = self.fusion_layer(cross_modal_features)
        # Classify
        logits = self.classifier(fused)
        return logits

class MultimodalFusionModel(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal fusion with different fusion strategies.
    """
    def __init__(
        self,
        text_dim=1024,
        audio_dim=768,
        video_dim=1,
        hidden_dim=512,
        output_dim=128,
        num_classes=2,
        fusion_strategy='early',
        learning_rate=1e-4,
        weight_decay=1e-5,
        dropout=0.3,
        class_weights=None,
        # Sequential processing hparams for audio
        audio_is_sequential=True, # Default to True for attention strategy
        audio_seq_processor_type='lstm',
        audio_seq_processor_hidden_dim=None, 
        audio_num_seq_processor_layers=1,
        # Sequential processing hparams for video
        video_is_sequential=True, # Default to True for attention strategy
        video_seq_processor_type='lstm',
        video_seq_processor_hidden_dim=None,
        video_num_seq_processor_layers=1
    ):
        super().__init__()
        self.save_hyperparameters() # This will save all constructor args
        # Create model based on fusion strategy
        if fusion_strategy == 'early':
            self.model = EarlyFusionModel(
                text_dim=text_dim,
                audio_dim=audio_dim,
                video_dim=video_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_strategy == 'late':
            self.model = LateFusionModel(
                text_dim=text_dim,
                audio_dim=audio_dim,
                video_dim=video_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_strategy == 'attention':
            self.model = CrossModalAttentionFusion(
                text_dim=self.hparams.text_dim, # Use hparams
                audio_dim=self.hparams.audio_dim, # Use hparams
                video_dim=self.hparams.video_dim, 
                hidden_dim=self.hparams.hidden_dim,
                output_dim=self.hparams.output_dim,
                num_classes=self.hparams.num_classes,
                dropout=self.hparams.dropout,
                # Pass sequential hparams for audio
                audio_is_sequential=self.hparams.audio_is_sequential,
                audio_seq_processor_type=self.hparams.audio_seq_processor_type,
                audio_seq_processor_hidden_dim=self.hparams.audio_seq_processor_hidden_dim,
                audio_num_seq_processor_layers=self.hparams.audio_num_seq_processor_layers,
                # Pass sequential hparams for video
                video_is_sequential=self.hparams.video_is_sequential,
                video_seq_processor_type=self.hparams.video_seq_processor_type,
                video_seq_processor_hidden_dim=self.hparams.video_seq_processor_hidden_dim,
                video_num_seq_processor_layers=self.hparams.video_num_seq_processor_layers
                # num_heads defaults to 8 in CrossModalAttentionFusion
            )
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
        # Loss function
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.train_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        if num_classes == 2:
            self.val_auroc = AUROC(task="binary")
            self.test_auroc = AUROC(task="binary")
            self.val_precision = Precision(task="binary")
            self.val_recall = Recall(task="binary")
            self.test_precision = Precision(task="binary")
            self.test_recall = Recall(task="binary")
    def forward(self, text_embedding, audio_embedding, video_embedding):
        return self.model(text_embedding, audio_embedding, video_embedding)
    def training_step(self, batch, batch_idx):
        text_embedding = batch['text_embedding']
        audio_embedding = batch['audio_embedding']
        video_embedding = batch['video_embedding']
        labels = batch['label']
        # Forward pass
        logits = self(text_embedding, audio_embedding, video_embedding)
        loss = self.criterion(logits, labels)
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        f1 = self.train_f1(preds, labels)
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        text_embedding = batch['text_embedding']
        audio_embedding = batch['audio_embedding']
        video_embedding = batch['video_embedding']
        labels = batch['label']

        # --- BEGIN DEBUG PRINTS ---
        print(f"\n--- Validation Step Debug (batch_idx: {batch_idx}) ---")
        print(f"Text embedding shape: {text_embedding.shape}")
        print(f"Audio embedding shape: {audio_embedding.shape}")
        print(f"Video embedding shape: {video_embedding.shape}")
        print(f"Labels tensor: {labels}")
        print(f"Labels shape: {labels.shape}")
        print(f"--- End Debug Prints ---\n")
        # --- END DEBUG PRINTS ---

        # Forward pass
        logits = self(text_embedding, audio_embedding, video_embedding)
        loss = self.criterion(logits, labels)
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)
        # Binary classification metrics
        if self.hparams.num_classes == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
            auroc = self.val_auroc(probs, labels)
            precision = self.val_precision(preds, labels)
            recall = self.val_recall(preds, labels)
            self.log('val_auroc', auroc, on_epoch=True)
            self.log('val_precision', precision, on_epoch=True)
            self.log('val_recall', recall, on_epoch=True)
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        text_embedding = batch['text_embedding']
        audio_embedding = batch['audio_embedding']
        video_embedding = batch['video_embedding']
        labels = batch['label']
        # Forward pass
        logits = self(text_embedding, audio_embedding, video_embedding)
        loss = self.criterion(logits, labels)
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        f1 = self.test_f1(preds, labels)
        # Binary classification metrics
        if self.hparams.num_classes == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
            auroc = self.test_auroc(probs, labels)
            precision = self.test_precision(preds, labels)
            recall = self.test_recall(preds, labels)
            self.log('test_auroc', auroc)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1', f1)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultimodalFusionModel")
        parser.add_argument('--text_dim', type=int, default=1024)
        parser.add_argument('--audio_dim', type=int, default=768)
        parser.add_argument('--video_dim', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--output_dim', type=int, default=128)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--fusion_strategy', type=str, default='early',
                           choices=['early', 'late', 'attention'])
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--dropout', type=float, default=0.3)

        # Args for sequential audio processing
        parser.add_argument('--audio_is_sequential', type=lambda x: (str(x).lower() == 'true'), default=True, help="Is audio input sequential (for attention strategy)?")
        parser.add_argument('--audio_seq_processor_type', type=str, default='lstm', choices=['lstm', 'gru'], help="Sequence processor type for audio.")
        parser.add_argument('--audio_seq_processor_hidden_dim', type=int, default=None, help="Hidden dim for audio sequence processor (e.g., LSTM hidden size). Defaults to main hidden_dim // 2.")
        parser.add_argument('--audio_num_seq_processor_layers', type=int, default=1, help="Number of layers for audio sequence processor.")

        # Args for sequential video processing
        parser.add_argument('--video_is_sequential', type=lambda x: (str(x).lower() == 'true'), default=True, help="Is video input sequential (for attention strategy)?")
        parser.add_argument('--video_seq_processor_type', type=str, default='lstm', choices=['lstm', 'gru'], help="Sequence processor type for video.")
        parser.add_argument('--video_seq_processor_hidden_dim', type=int, default=None, help="Hidden dim for video sequence processor. Defaults to main output_dim // 2 for video encoder.")
        parser.add_argument('--video_num_seq_processor_layers', type=int, default=1, help="Number of layers for video sequence processor.")
        
        return parent_parser
