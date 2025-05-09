import torch
import numpy as np
import pytorch_lightning as pl
from hubert_ser_module import HubertSER
# Import both DataModules
from data_module import CREMADRAVDESSDataModule, ManifestDataModule, LABEL_MAP
import argparse
from tqdm import tqdm
import pandas as pd # Needed for ManifestDataModule path check potentially
import os # Needed for path checks

def generate_embeddings(args):
    # --- Configuration ---
    NUM_CLASSES = len(LABEL_MAP)
    MODEL_NAME = args.model_name # Should match the trained model
    POOLING_MODE = args.pooling_mode
    CHECKPOINT_PATH = args.checkpoint_path # Optional now
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    OUTPUT_FILE = args.output_file
    DATA_SPLIT = args.data_split

    # --- Load Model ---
    device = torch.device('cuda' if torch.cuda.is_available() and args.accelerator != 'cpu' else 'cpu')

    if CHECKPOINT_PATH:
        print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
        # Load using the class, strict=False if some head weights might be missing/different
        # Pass necessary hyperparameters if they are not saved in the checkpoint or need override
        model = HubertSER.load_from_checkpoint(
            CHECKPOINT_PATH,
            hubert_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            pooling=POOLING_MODE,
            # Add other necessary hparams if load_from_checkpoint requires them
            # e.g., lr, warmup_steps, total_steps might be needed if strict=True
            map_location=device # Load directly to the target device
        )
        print(f"Model loaded from checkpoint.")
    else:
        print(f"No checkpoint provided. Initializing HubertSER with pre-trained weights for {MODEL_NAME}...")
        # Instantiate directly, will load pre-trained HuBERT weights
        # Note: The classification head will be randomly initialized, but we only need the base HuBERT features.
        model = HubertSER(
            hubert_name=MODEL_NAME,
            num_classes=NUM_CLASSES, # Still need num_classes for model init
            pooling=POOLING_MODE
            # No need for lr, warmup, total_steps etc. for inference
        )
        print(f"Model initialized with pre-trained weights.")

    model.eval() # Set to evaluation mode
    model.to(device) # Move model to appropriate device
    print(f"Model ready on device: {device}.")


    # --- DataModule ---
    # Use the same DataModule setup as training to ensure consistency
    # --- DataModule & DataLoader ---
    print("Initializing DataModule...")
    if args.manifest_path:
        print(f"Using manifest file: {args.manifest_path}")
        if not os.path.exists(args.manifest_path):
             raise FileNotFoundError(f"Manifest file not found: {args.manifest_path}")
        data_module = ManifestDataModule(
            manifest_path=args.manifest_path,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            model_name=MODEL_NAME,
            max_duration_s=args.max_duration_s,
            label_map=LABEL_MAP,
            label_col=args.label_col,
            path_col=args.path_col,
            data_root=args.data_root # Pass data_root if provided
        )
        print("Setting up ManifestDataModule...")
        data_module.setup()
        print("ManifestDataModule setup complete.")
        dataloader = data_module.manifest_dataloader()
        data_description = f"manifest {os.path.basename(args.manifest_path)}"
    elif DATA_DIR:
        print(f"Using data splits from directory: {DATA_DIR}")
        data_module = CREMADRAVDESSDataModule(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            model_name=MODEL_NAME,
            max_duration_s=args.max_duration_s,
            label_map=LABEL_MAP,
            add_noise=False # No noise augmentation for embedding generation
        )
        print("Setting up CREMADRAVDESSDataModule...")
        data_module.setup()
        print("CREMADRAVDESSDataModule setup complete.")
        # Select the appropriate dataloader based on DATA_SPLIT
        if DATA_SPLIT == 'train':
            dataloader = data_module.train_dataloader()
        elif DATA_SPLIT == 'val':
            dataloader = data_module.val_dataloader()
        elif DATA_SPLIT == 'test':
            dataloader = data_module.test_dataloader()
        else:
            raise ValueError(f"Invalid data_split: {DATA_SPLIT}. Choose 'train', 'val', or 'test' when using --data_dir.")
        data_description = f"'{DATA_SPLIT}' split"
    else:
        raise ValueError("Either --manifest_path or --data_dir must be provided.")


    # --- Generate Embeddings ---
    all_embeddings = []
    all_labels = []
    print(f"Generating embeddings for {data_description}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {data_description} batches"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'] # Keep labels on CPU

            # Get hidden states from the model's forward pass
            # Modify HubertSER forward if it doesn't return hidden states directly
            # Option 1: Modify forward to return hidden_states
            # outputs = model.hubert(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
            # hidden_states = outputs.last_hidden_state

            # Option 2: Access hubert directly (if accessible)
            outputs = model.hubert(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state # [Batch, SeqLen, HiddenDim]

            # Apply the same pooling used during training
            pooled_embeddings = model._pool(hidden_states, attention_mask) # [Batch, HiddenDim]

            all_embeddings.append(pooled_embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate results from all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Generated labels shape: {all_labels.shape}")

    # --- Save Embeddings ---
    print(f"Saving embeddings to {OUTPUT_FILE}...")
    # Save embeddings and labels together for easier use later
    np.savez(OUTPUT_FILE, embeddings=all_embeddings, labels=all_labels)
    # Or just embeddings: np.save(OUTPUT_FILE, all_embeddings)
    print("Embeddings saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HuBERT Embeddings from a Trained Model")

    # Model args
    parser.add_argument("--model_name", type=str, default="facebook/hubert-base-ls960", help="Pretrained HuBERT model name")
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "max"], help="Pooling strategy for embeddings")
    parser.add_argument("--checkpoint_path", type=str, default=None, required=False, help="Optional path to a trained model checkpoint (.ckpt) to load fine-tuned weights")

    # Data args
    # Data args (Split-based - keep for backward compatibility or alternative use)
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing train/val/test CSV files (used if --manifest_path is not provided)")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"], help="Which data split to use (used if --manifest_path is not provided)")

    # Data args (Manifest-based - new arguments)
    parser.add_argument("--manifest_path", type=str, default=None, help="Path to a single manifest CSV file (overrides --data_dir and --data_split)")
    parser.add_argument("--path_col", type=str, default="path", help="Column name for file paths in the manifest CSV")
    parser.add_argument("--label_col", type=str, default="emotion", help="Column name for labels in the manifest CSV")
    parser.add_argument("--data_root", type=str, default=None, help="Optional root directory for resolving relative paths in the manifest")

    # Common data args
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference") # Can often be larger than training bs
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--max_duration_s", type=float, default=5.0, help="Maximum audio duration in seconds for truncation/padding")


    # Output args
    parser.add_argument("--output_file", type=str, default="audio_embeddings.npz", help="Path to save the generated embeddings (.npz file)")

    # Device args
    parser.add_argument("--accelerator", type=str, default="auto", help="Device to use ('cpu', 'gpu', 'mps', 'auto')")

    args = parser.parse_args()
    generate_embeddings(args)
