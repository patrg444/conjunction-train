import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import get_linear_schedule_with_warmup

# Assuming the new dataset and model are in these locations
from dataloaders.humor_multimodal_dataset_v2 import HumorMultimodalDatasetV2, AU_INTENSITY_COLUMNS
from models.humor_fusion_model_v2 import HumorFusionModelV2

# Set up logging (duplicates from other scripts, consider a shared utility)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class HumorFusionLightningModule(pl.LightningModule):
    def __init__(self, model_config, training_config, audio_feature_dim, text_feature_dim, video_au_feature_dim):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ arguments to hparams
        self.model = HumorFusionModelV2(
            audio_input_dim=audio_feature_dim,
            text_input_dim=text_feature_dim,
            video_au_input_dim=video_au_feature_dim,
            audio_lstm_hidden_dim=model_config.get('audio_lstm_hidden_dim', 128),
            video_au_lstm_hidden_dim=model_config.get('video_au_lstm_hidden_dim', 64),
            text_fc_dim=model_config.get('text_fc_dim', 256),
            fusion_dim=model_config.get('fusion_dim', 256),
            num_classes=model_config.get('num_classes', 2),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
        self.training_config = training_config
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(training_config['class_weights'], dtype=torch.float) if training_config.get('class_weights') else None)

    def forward(self, audio_features, text_features, video_au_features):
        return self.model(audio_features, text_features, video_au_features)

    def training_step(self, batch, batch_idx):
        audio, text, video_au, labels = batch['audio_features'], batch['text_features'], batch['video_au_features'], batch['label']
        logits = self(audio, text, video_au)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, text, video_au, labels = batch['audio_features'], batch['text_features'], batch['video_au_features'], batch['label']
        logits = self(audio, text, video_au)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def test_step(self, batch, batch_idx):
        audio, text, video_au, labels = batch['audio_features'], batch['text_features'], batch['video_au_features'], batch['label']
        logits = self(audio, text, video_au)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds, 'labels': labels, 'probs': probs}

    def _epoch_end_metrics(self, outputs, stage_name):
        labels = torch.cat([x['labels'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])
        
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average=self.training_config.get('f1_average', 'binary'), zero_division=0)
        precision = precision_score(labels.cpu(), preds.cpu(), average=self.training_config.get('f1_average', 'binary'), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), average=self.training_config.get('f1_average', 'binary'), zero_division=0)
        
        self.log(f'{stage_name}_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage_name}_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage_name}_precision', precision, on_epoch=True, logger=True)
        self.log(f'{stage_name}_recall', recall, on_epoch=True, logger=True)

        if stage_name != 'train' and self.model.num_classes == 2: # AUROC only for binary classification and val/test
            probs = torch.cat([x['probs'] for x in outputs])
            try:
                auroc = roc_auc_score(labels.cpu(), probs.cpu()[:, 1]) # Prob of positive class
                self.log(f'{stage_name}_auroc', auroc, on_epoch=True, logger=True)
            except ValueError as e:
                 logger.warning(f"Could not compute AUROC for {stage_name}: {e}")
                 self.log(f'{stage_name}_auroc', 0.0, on_epoch=True, logger=True)


    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking: # Skip sanity check for aggregating metrics
            outputs = self.trainer.callback_metrics # Lightning 2.0+ way
            # For older versions, you might need to collect outputs manually
            # For simplicity, assuming outputs are directly available or handled by Lightning's internal mechanism for logging
            # This part might need adjustment based on the exact Lightning version and how outputs are passed.
            # The provided `validation_step` returns a dict, which Lightning should aggregate.
            # If direct access to aggregated outputs is needed, it's usually via `self.validation_step_outputs`
            # and then clearing it with `self.validation_step_outputs.clear()`
            # However, relying on `self.trainer.callback_metrics` for logged values is often sufficient.
            # For explicit calculation:
            # aggregated_outputs = [] # This would need to be populated by collecting from validation_step
            # self._epoch_end_metrics(aggregated_outputs, 'val')
            pass # Metrics are logged per step/epoch via self.log

    def on_test_epoch_end(self):
        # aggregated_outputs = [] # Similar to validation, this would need manual collection if not relying on logged metrics
        # self._epoch_end_metrics(aggregated_outputs, 'test')
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config.get('weight_decay', 0.01))
        
        if self.training_config.get("scheduler"):
            scheduler_params = self.training_config["scheduler"]
            if scheduler_params["type"] == "linear_with_warmup":
                num_training_steps = self.trainer.estimated_stepping_batches
                num_warmup_steps = int(scheduler_params["warmup_ratio"] * num_training_steps)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer


def main(args):
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config['model_params']
    data_params = config['data_params']
    training_params = config['training_params']

    # Determine feature dimensions (these should be known from your feature extraction)
    # Example: audio_feature_dim = 1024 (WavLM)
    # text_feature_dim = 1024 (XLM-R Large)
    # video_au_feature_dim = 17 (OpenFace AUs)
    # These should ideally be part of the config or inferred more robustly
    audio_feature_dim = model_params.get('audio_input_dim', 1024) 
    text_feature_dim = model_params.get('text_input_dim', 1024)
    video_au_feature_dim = model_params.get('video_au_input_dim', len(AU_INTENSITY_COLUMNS))


    # Datasets and DataLoaders
    train_dataset = HumorMultimodalDatasetV2(
        manifest_path=data_params['train_manifest_path'],
        max_audio_len=data_params.get('max_audio_len'),
        max_video_au_len=data_params.get('max_video_au_len')
    )
    val_dataset = HumorMultimodalDatasetV2(
        manifest_path=data_params['val_manifest_path'],
        max_audio_len=data_params.get('max_audio_len'),
        max_video_au_len=data_params.get('max_video_au_len')
    )
    test_dataset = HumorMultimodalDatasetV2(
        manifest_path=data_params['test_manifest_path'],
        max_audio_len=data_params.get('max_audio_len'),
        max_video_au_len=data_params.get('max_video_au_len')
    )

    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=data_params.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=data_params.get('num_workers', 4), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=data_params.get('num_workers', 4), pin_memory=True)

    # Model
    lightning_model = HumorFusionLightningModule(model_params, training_params, audio_feature_dim, text_feature_dim, video_au_feature_dim)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
        save_top_k=training_params.get('save_top_k', 1),
        monitor=training_params.get('monitor_metric', 'val_f1'),
        mode=training_params.get('monitor_mode', 'max')
    )
    early_stop_callback = EarlyStopping(
        monitor=training_params.get('monitor_metric', 'val_f1'),
        patience=training_params.get('early_stopping_patience', 5),
        verbose=True,
        mode=training_params.get('monitor_mode', 'max')
    )
    
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=args.output_dir, name='lightning_logs')

    # Trainer
    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=training_params['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        deterministic=training_params.get('deterministic', False),
        # precision=16 if training_params.get('use_amp', False) else 32 # For mixed precision
    )

    logger.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    
    logger.info("Training finished. Starting testing...")
    # Load best model for testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} for testing.")
        lightning_model = HumorFusionLightningModule.load_from_checkpoint(
            best_model_path,
            model_config=model_params, 
            training_config=training_params,
            audio_feature_dim=audio_feature_dim,
            text_feature_dim=text_feature_dim,
            video_au_feature_dim=video_au_feature_dim
            )
    else:
        logger.warning("No best model checkpoint found. Testing with the last model state.")

    test_results = trainer.test(lightning_model, test_loader)
    logger.info("Testing results:")
    for key, value in test_results[0].items():
        logger.info(f"  {key}: {value}")

    # Save test results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(os.path.join(args.output_dir, "test_results.csv"), index=False)
    logger.info(f"Test results saved to {os.path.join(args.output_dir, 'test_results.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Humor Multimodal Fusion Model V2")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use (if available).")
    
    cli_args = parser.parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)
    main(cli_args)
