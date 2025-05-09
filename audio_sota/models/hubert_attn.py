import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import HubertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
# Removed OneCycleLR import
import torchmetrics # For accuracy calculation
# Removed torchaudio transforms for SpecAugment

# Assuming UAR metric is defined elsewhere, e.g., common.metrics
# from common.metrics import UnweightedAverageRecall 

# Removed AttentiveStatisticalPooling class as we'll use mean/max pooling directly

# Renamed class for clarity, as it no longer uses LSTM/Attention pooling by default
class HubertSequenceClassificationModel(pl.LightningModule):
    """
    PyTorch Lightning module using HuBERT for sequence classification.
    Supports mean/max pooling and optional class weighting/freezing.
    """
    def __init__(self,
                 hubert_model_name: str = "facebook/hubert-base-ls960",
                 num_classes: int = 8,
                 class_weights: torch.Tensor = None, # Optional tensor for weighted loss
                 lr: float = 2e-5,
                 warmup_steps: int = 500,
                 total_training_steps: int = 10000, # Placeholder, should be calculated
                 freeze_encoder_epochs: int = 2,
                 pooling_mode: str = 'mean', # 'mean', 'max'
                 dropout_rate: float = 0.1):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters(ignore=['class_weights']) # Don't save weights tensor

        # Load Pretrained HuBERT model
        self.hubert = HubertModel.from_pretrained(hubert_model_name)

        # Freeze initially, will be handled by on_train_epoch_start
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.hubert.eval()

        hubert_output_dim = self.hubert.config.hidden_size

        # Pooling layer (handled in forward)
        if self.hparams.pooling_mode not in ['mean', 'max']:
             raise ValueError(f"Unsupported pooling_mode: {self.hparams.pooling_mode}")

        # Dropout and Final Classifier
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.classifier = nn.Linear(hubert_output_dim, num_classes)

        # Loss function (optionally weighted)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics - Using torchmetrics.classification for UAR
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # UAR (Unweighted Average Recall) is macro-averaged Recall
        self.val_uar = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_uar = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')


    def forward(self, input_values, attention_mask=None, return_pooled_output=False):
        """
        Args:
            input_values (torch.Tensor): Raw waveform input, shape [batch_size, sequence_length].
            attention_mask (torch.Tensor, optional): Mask for padding tokens, shape [batch_size, sequence_length].
            return_pooled_output (bool): If True, return the pooled hidden state before dropout/classifier.
            attention_mask (torch.Tensor, optional): Mask for padding tokens, shape [batch_size, sequence_length].
        """
        # HuBERT forward pass
        # Ensure model is in correct mode (train/eval handled by PL and freezing logic)
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state # Shape: [batch, seq_len, hidden_size]

        # Apply pooling
        if self.hparams.pooling_mode == 'mean':
            # Mean pool across sequence length, considering attention mask if available
            if attention_mask is not None:
                # Expand mask to match hidden state shape for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                # Sum only non-masked elements
                summed = torch.sum(last_hidden_state * mask_expanded, dim=1)
                # Count non-masked elements per batch item
                summed_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = summed / summed_mask
            else:
                # Simple mean if no mask
                pooled_output = torch.mean(last_hidden_state, dim=1)
        elif self.hparams.pooling_mode == 'max':
             # Max pool across sequence length (masking is trickier here, often ignored or applied before max)
             # Simple max pooling for now:
             pooled_output = torch.max(last_hidden_state, dim=1).values

        # Return pooled output directly if requested (before dropout/classifier)
        if return_pooled_output:
            return pooled_output # Shape: [batch, hidden_size]
        
        # Apply Dropout before classifier
        pooled_output_dropped = self.dropout(pooled_output)

        # Pass through Classifier
        logits = self.classifier(pooled_output_dropped) # Shape: [batch, num_classes]

        return logits

    def _shared_step(self, batch):
        # Assumes batch structure: {'input_values': tensor, 'attention_mask': tensor, 'labels': tensor}
        # Adjust if your DataModule provides a different structure
        input_values = batch.get('input_values')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')

        if input_values is None or labels is None:
             # Handle cases where features/labels might be named differently
             # Fallback for simpler tuple structure (features, labels)
             if isinstance(batch, (tuple, list)) and len(batch) == 2:
                 input_values, labels = batch
                 attention_mask = None # Assume no mask if tuple structure
             else:
                 raise ValueError("Could not extract input_values and labels from batch.")

        logits = self(input_values=input_values, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True) # Less verbose bar
        self.train_accuracy(logits, labels)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(logits, labels)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True) # Less verbose bar
        self.val_uar(logits, labels)
        self.log('val_uar', self.val_uar, on_step=False, on_epoch=True, prog_bar=True, logger=True) # UAR on prog bar

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.test_accuracy(logits, labels)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        self.test_uar(logits, labels)
        self.log('test_uar', self.test_uar, on_step=False, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        """ Configure AdamW optimizer and linear warmup/decay scheduler. """
        # Apply weight decay to all parameters except bias and LayerNorm weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01, # Standard weight decay
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)

        # Get scheduler
        # total_steps needs to be passed accurately during init or estimated carefully
        # Using the value stored in hparams from the training script calculation
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Important: Update scheduler per step
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """ Hook for implementing freezing logic. """
        epoch = self.current_epoch
        freeze_epochs = self.hparams.freeze_encoder_epochs

        if freeze_epochs <= 0:
             # Ensure unfrozen if freeze_epochs is 0 or negative
             if not self.hubert.training:
                 print(f"Epoch {epoch}: Ensuring HuBERT is unfrozen and in train mode.")
                 for param in self.hubert.parameters():
                     param.requires_grad = True
                 self.hubert.train()
             return

        if epoch < freeze_epochs:
            # Keep frozen
            if self.hubert.training: # If it was somehow unfrozen, freeze it back
                 print(f"Epoch {epoch}: Keeping HuBERT frozen and in eval mode.")
                 for param in self.hubert.parameters():
                     param.requires_grad = False
                 self.hubert.eval()
            # else: already frozen, do nothing
        elif epoch == freeze_epochs:
            # Unfreeze at the target epoch
            print(f"Epoch {epoch}: Unfreezing HuBERT backbone and setting to train mode.")
            for param in self.hubert.parameters():
                param.requires_grad = True
            self.hubert.train()
        else: # epoch > freeze_epochs
             # Ensure it stays unfrozen
             if not self.hubert.training:
                 print(f"Epoch {epoch}: Ensuring HuBERT remains unfrozen and in train mode.")
                 for param in self.hubert.parameters():
                     param.requires_grad = True
                 self.hubert.train()


# Example Usage (for testing model structure) - Updated
if __name__ == '__main__':
    print("Testing HubertSequenceClassificationModel structure...")
    
    # Example instantiation with required hparams (placeholders for steps)
    model = HubertSequenceClassificationModel(
        num_classes=8, 
        lr=2e-5, 
        warmup_steps=100, 
        total_training_steps=1000, 
        freeze_encoder_epochs=2,
        pooling_mode='mean'
    )
    
    # Create a dummy batch of raw audio data (as if loaded by DataModule)
    # Shape: [batch_size, sequence_length]
    dummy_input_values = torch.randn(4, 16000 * 4) # Batch of 4, 4 seconds at 16kHz
    dummy_attention_mask = torch.ones(4, 16000 * 4) # Simple mask (no padding)
    
    # Perform a forward pass
    logits = model(input_values=dummy_input_values, attention_mask=dummy_attention_mask)
    
    print("Model instantiated.")
    print("Input waveform shape:", dummy_input_values.shape)
    print("Output logits shape:", logits.shape) # Should be [batch_size, num_classes] -> [4, 8]
    
    # Print model summary (optional, requires torchinfo)
    try:
        from torchinfo import summary
        # Provide input size matching the forward pass input
        # Need to provide both input_values and attention_mask if model expects it
        summary(model, input_data={'input_values': dummy_input_values, 'attention_mask': dummy_attention_mask})
    except ImportError:
        print("\nInstall torchinfo for model summary: pip install torchinfo")
    except Exception as e:
        print(f"\nCould not generate summary: {e}")
