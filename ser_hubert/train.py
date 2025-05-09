import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import math
import argparse
import torch # Added for loading weights
import os    # Added for checking file existence

# Import the modules we created
from hubert_ser_module import HubertSER
from data_module import CREMADRAVDESSDataModule, LABEL_MAP

def train(args):
    # --- Configuration ---
    NUM_CLASSES = len(LABEL_MAP) # Determine from the imported map
    MODEL_NAME = args.model_name
    FREEZE_EPOCHS = args.freeze_epochs
    MAX_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size
    ACCUMULATE_GRAD_BATCHES = args.accumulate_grad_batches
    LEARNING_RATE = args.learning_rate
    WARMUP_STEPS = args.warmup_steps
    POOLING_MODE = args.pooling_mode
    DROPOUT = args.dropout
    PRECISION = args.precision
    DATA_DIR = args.data_dir
    NUM_WORKERS = args.num_workers
    PATIENCE = args.early_stopping_patience

    # --- DataModule ---
    print("Initializing DataModule...")
    data_module = CREMADRAVDESSDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        model_name=MODEL_NAME,
        label_map=LABEL_MAP, # Pass the map
        data_root="/Users/patrickgloria/conjunction-train/SMILE/" # Set data_root
    )
    # Call setup to determine dataset size for total_steps calculation
    print("Setting up DataModule to calculate total steps...")
    data_module.setup('fit')
    print("DataModule setup complete.")

    # Calculate total training steps for LR scheduler
    # Ensure train_dataloader is accessible after setup
    train_loader = data_module.train_dataloader()
    if hasattr(train_loader.dataset, '__len__'):
         # Estimate steps based on dataset size, batch size, accumulation, and epochs
         # This might need adjustment if using iterable datasets without a fixed length
         steps_per_epoch = math.ceil(len(train_loader.dataset) / (BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))
         total_steps = steps_per_epoch * MAX_EPOCHS
         print(f"Calculated total_steps: {total_steps}")
    else:
         # Fallback or error if length cannot be determined
         print("Warning: Could not determine dataset length. Using default total_steps=10000 for scheduler.")
         total_steps = 10000 # Default fallback

    # --- Load Class Weights ---
    class_weights = None
    weights_path = "ser_hubert/class_weights.pt"
    if os.path.exists(weights_path):
        print(f"Loading class weights from {weights_path}...")
        try:
            class_weights = torch.load(weights_path)
            # Optional: Move weights to the correct device if needed, though PL might handle this
            # class_weights = class_weights.to(device) # Replace 'device' with actual device
            print(f"Successfully loaded class weights: {class_weights}")
        except Exception as e:
            print(f"Warning: Failed to load class weights from {weights_path}: {e}. Proceeding without weights.")
            class_weights = None
    else:
        print(f"Class weights file not found at {weights_path}. Proceeding without weights.")


    # --- Model ---
    print("Initializing Model...")
    model = HubertSER(
        hubert_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        class_weights=class_weights, # Pass loaded weights
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps, # Pass calculated total steps
        freeze_epochs=FREEZE_EPOCHS,
        pooling=POOLING_MODE,
        dropout=DROPOUT
        # class_weights can be added here if needed
    )
    print("Model initialized.")

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_uar", # Monitor Unweighted Average Recall on validation set
        mode="max",
        filename='hubert-ser-{epoch:02d}-{val_uar:.2f}',
        save_top_k=1,
        verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_uar",
        patience=PATIENCE,
        mode="max",
        verbose=True
    )

    # --- Trainer ---
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        precision=PRECISION,
        gradient_clip_val=1.0, # As mentioned in the minimal script example
        callbacks=[checkpoint_callback, early_stopping_callback],
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        # Add other trainer args like accelerator='gpu', devices=1 if needed
        # accelerator='mps' could be used for Apple Silicon GPUs
        # accelerator='cpu' # Default if no GPU
        # logger=... # Add TensorBoardLogger or other loggers if desired
    )
    print("Trainer initialized.")

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("Training finished.")

    # --- Testing (Optional) ---
    # print("Starting testing...")
    # trainer.test(model, datamodule=data_module) # Loads best checkpoint by default
    # print("Testing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HuBERT SER Model")

    # Model args
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960", help="Pretrained HuBERT model name")
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "max"], help="Pooling strategy")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the classifier")

    # Training args
    parser.add_argument("--max_epochs", type=int, default=15, help="Maximum number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=2, help="Number of epochs to keep HuBERT frozen")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps for LR scheduler")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision (e.g., '32', '16-mixed', 'bf16-mixed')")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping based on val_uar")

    # Data args
    parser.add_argument("--data_dir", type=str, default="splits", help="Directory containing train/val/test CSV files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Add GPU/Accelerator args if needed
    # parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (cpu, gpu, mps, etc.)")
    # parser.add_argument("--devices", type=int, default=1, help="Number of devices")

    args = parser.parse_args()
    train(args)
