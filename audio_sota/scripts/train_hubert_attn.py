import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torchmetrics # Added for UAR
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Adjust imports based on actual project structure if needed
# Assumes common and models are siblings to scripts or accessible via PYTHONPATH
import sys
# Add parent directory (audio_sota) to sys.path to allow imports like common.datamodule
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from common.datamodule import AudioDataModule # Assuming this handles data loading/sampling
from models.hubert_attn import HubertSequenceClassificationModel # Corrected class name
from common.augment import build_augmentations # Assuming this exists
from common.metrics import UnweightedAverageRecall # Assuming this exists

def main(args):
    pl.seed_everything(args.seed, workers=True)

    # --- Data Module ---
    # Build augmentation pipeline if requested
    train_transforms = build_augmentations(args.sample_rate) if args.use_augmentations else None
    val_transforms = None   # No augmentation for validation/test typically
    test_transforms = None

    dm = AudioDataModule(
        dataset_names=args.dataset_names,
        data_dir=args.data_dir,
        feature_type=args.feature_type, # This likely needs to change if not using pre-extracted features
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
        use_balanced_sampler=args.use_balanced_sampler, # Pass sampler flag
        max_length_seconds=args.max_length_seconds, # Pass max length
        sample_rate=args.sample_rate # Pass sample rate
    )
    dm.prepare_data() # Call prepare_data explicitly if needed
    dm.setup()        # Call setup explicitly

    # --- Model ---
    # Get num_classes and class weights from datamodule after setup
    num_classes = dm.num_classes
    class_weights = dm.compute_class_weights() if args.use_class_weights else None

    # Calculate total steps for scheduler
    # Ensure dm.train_dataset is accessible after setup()
    if hasattr(dm, 'train_dataset') and dm.train_dataset is not None:
         # Calculate steps per epoch carefully based on actual dataset size and batching
        effective_batch_size = args.batch_size * args.gpus * args.gradient_accumulation_steps
        # Use len(dm.train_dataloader()) if sampler affects length, else len(dm.train_dataset)
        # This might require dm.setup() to be called first. Let's assume dm provides train_dataloader length
        steps_per_epoch = len(dm.train_dataloader()) # Need to ensure this is correct after setup
        total_training_steps = steps_per_epoch * args.max_epochs
        warmup_steps = args.warmup_steps # Use direct steps from args
        print(f"Calculated steps_per_epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")
        print(f"Warmup steps: {warmup_steps}")
    else:
        print("Warning: Could not determine train_dataset length before Trainer.fit(). Scheduler steps might be inaccurate.")
        total_training_steps = 10000 # Placeholder, will be recalculated by model if needed
        warmup_steps = args.warmup_steps # Use direct steps from args


    model = HubertSequenceClassificationModel( # Corrected class name
        hubert_model_name=args.hubert_model,
        num_classes=num_classes,
        class_weights=class_weights, # Pass class weights
        lr=args.lr, # Use single LR
        warmup_steps=warmup_steps, # Pass calculated warmup steps
        total_training_steps=total_training_steps, # Pass calculated total steps
        freeze_encoder_epochs=args.freeze_encoder_epochs, # Pass freeze duration
        pooling_mode=args.pooling_mode, # Pass pooling mode
        dropout_rate=args.dropout_rate, # Pass dropout rate
        # Remove LSTM specific args if not using LSTM head
        # lstm_hidden_dim=args.lstm_hidden,
        # lstm_layers=args.lstm_layers,
        # Remove specaugment args if handled by HF model directly
        # use_specaugment=args.use_specaugment,
        # time_mask_param=args.time_mask_param,
        # freq_mask_param=args.freq_mask_param,
    )

    # --- Training ---
    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.run_name, 'checkpoints'),
        filename='{epoch}-{step}-{val_uar:.3f}', # Include step, use val_uar
        monitor='val_uar', # Monitor validation UAR
        mode='max',        # Save the model with the highest validation UAR
        save_top_k=1,
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_uar', # Monitor validation UAR
        patience=args.early_stopping_patience, # Use arg for patience
        verbose=True,
        mode='max'
    )
    
    # --- Logger ---
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.run_name,
        version=0 # Start versioning from 0
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        deterministic=True,
        precision='16-mixed' if args.use_amp else 32, # Use '16-mixed' for PyTorch >= 1.6
        # strategy='ddp' if args.gpus > 1 else None, # Add DDP later if needed
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_interval,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.gradient_accumulation_steps # Add gradient accumulation
    )

    # --- Run Training ---
    print(f"\nStarting training run: {args.run_name}")
    print(f"Datasets: {', '.join(args.dataset_names)}")
    print(f"Model: {args.hubert_model}, Pooling: {args.pooling_mode}")
    print(f"  - LR: {args.lr}, Warmup Steps: {warmup_steps}")
    print(f"  - Freeze Encoder Epochs: {args.freeze_encoder_epochs}")
    print(f"  - Batch Size: {args.batch_size} (Effective: {effective_batch_size})")
    print(f"  - Max Epochs: {args.max_epochs}, Early Stopping Patience: {args.early_stopping_patience}")
    print(f"  - Augmentations: {args.use_augmentations}, Balanced Sampler: {args.use_balanced_sampler}")
    print(f"  - Class Weights: {args.use_class_weights}, Dropout: {args.dropout_rate}")
    print(f"  - Precision: {'16-mixed' if args.use_amp else 32}, Grad Clip: {args.gradient_clip_val}")
    print(f"Output directory: {os.path.join(args.output_dir, args.run_name)}")

    trainer.fit(model, datamodule=dm)

    # --- Run Testing ---
    if checkpoint_callback.best_model_path:
        print(f"\nStarting testing using best checkpoint: {checkpoint_callback.best_model_path}")
        # trainer.test() automatically uses the best checkpoint if ckpt_path='best' and load_best_model_at_end=True in Trainer
        # However, explicitly loading might be safer depending on PL version and exact setup.
        # Let's rely on ckpt_path='best' for now.
        test_results = trainer.test(model=model, datamodule=dm, ckpt_path='best')
        print("Test Results:", test_results)
    else:
        print("\nNo best model checkpoint found. Skipping testing.")

    print(f"\nTraining and testing finished for run: {args.run_name}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best model score (val_uar): {checkpoint_callback.best_model_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HuBERT-Attention model for Speech Emotion Recognition")

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Train HuBERT model for Speech Emotion Recognition using PyTorch Lightning")

    # Data Args
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, choices=['crema_d', 'ravdess'], help='List of dataset names (e.g., ravdess crema_d)')
    parser.add_argument('--data_dir', type=str, default='data/datasets_processed', help='Base directory containing processed dataset CSVs (train.csv, val.csv, test.csv)')
    parser.add_argument('--audio_dir', type=str, default='data/datasets_raw', help='Base directory containing raw audio files referenced in CSVs')
    parser.add_argument('--output_dir', type=str, default='audio_sota/outputs', help='Directory to save logs and checkpoints')
    parser.add_argument('--run_name', type=str, default='hubert_ser_run', help='Specific name for this training run')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--max_length_seconds', type=float, default=5.0, help='Maximum audio length in seconds (longer clips are truncated)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate for audio loading')
    parser.add_argument('--feature_type', type=str, default='raw', help='Feature type (e.g., raw, hubert_large). "raw" means load audio.') # Keep for datamodule, but model uses raw audio

    # Model Args
    parser.add_argument('--hubert_model', type=str, default='facebook/hubert-base-ls960', help='Pretrained HuBERT model name from Hugging Face')
    parser.add_argument('--pooling_mode', type=str, default='mean', choices=['mean', 'max', 'attention'], help='Pooling strategy for HuBERT outputs')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate before the final classifier')

    # Training Args
    parser.add_argument('--lr', type=float, default=2e-5, help='Peak learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients over')
    parser.add_argument('--max_epochs', type=int, default=15, help='Maximum number of training epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (set to 0 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience (epochs) for early stopping based on val_uar')
    parser.add_argument('--log_interval', type=int, default=10, help='Log training metrics every N steps')
    parser.add_argument('--val_interval', type=float, default=1.0, help='Run validation every N epochs (float) or steps (int)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (FP16)')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of linear warmup steps for the learning rate scheduler')

    # Fine-tuning Args
    parser.add_argument('--freeze_encoder_epochs', type=int, default=2, help='Number of initial epochs to keep the HuBERT encoder frozen')
    # Removed partial unfreeze for simplicity, aligning with HF Trainer recipe

    # Loss & Sampling Args
    parser.add_argument('--use_class_weights', action='store_true', help='Use inverse frequency class weighting for the loss')
    parser.add_argument('--use_balanced_sampler', action='store_true', help='Use a weighted random sampler for the training dataloader')

    # Augmentation Args
    parser.add_argument('--use_augmentations', action='store_true', help='Enable audio augmentations (noise, pitch, stretch)')
    # Removed SpecAugment args, assuming it's handled within the HF model if applicable or not used in base recipe

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, args.run_name, 'checkpoints'), exist_ok=True)

    main(args)
