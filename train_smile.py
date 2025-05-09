import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import argparse
from datasets.image_dataset import ImageDataset # Assuming image_dataset.py is in datasets/
import os
import torchmetrics # For accuracy calculation

class SmileClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for training a ResNet-18 smile classifier on CelebA.
    """
    def __init__(self, learning_rate=1e-4, max_epochs=10):
        super().__init__()
        self.save_hyperparameters() # Saves learning_rate, max_epochs

        # Load pre-trained ResNet-18 model
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)

        # Replace the final fully connected layer for binary classification (smile/no smile)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1) # Output single logit for BCEWithLogitsLoss

        # Use BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Use torchmetrics for accuracy
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")

        # Store transforms needed by the model (from weights)
        self.preprocess = weights.transforms()
        print("Using standard ImageNet transforms for ResNet18:")
        print(self.preprocess)


    def forward(self, x):
        # Apply preprocessing within forward if not done in dataset
        # x = self.preprocess(x) # Assuming transforms are applied in Dataset
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1) # Remove trailing dimension for BCE loss
        loss = self.criterion(logits, y.float()) # Ensure target is float for BCE
        
        # Calculate accuracy
        acc = self.train_accuracy(torch.sigmoid(logits), y) # Use sigmoid for probability

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        
        # Calculate accuracy
        acc = self.val_accuracy(torch.sigmoid(logits), y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # Optional: Add test_step if you have a separate test manifest
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x).squeeze(1)
    #     loss = self.criterion(logits, y.float())
    #     acc = self.accuracy(torch.sigmoid(logits), y)
    #     self.log('test_loss', loss, on_epoch=True, logger=True)
    #     self.log('test_acc', acc, on_epoch=True, logger=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Cosine Annealing LR Scheduler
        # Consider a smaller T_max if using early stopping
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def main(args):
    pl.seed_everything(42, workers=True) # for reproducibility

    # --- Transforms ---
    # Use the transforms recommended for the ResNet18 weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    train_transform = weights.transforms()
    val_transform = weights.transforms() # Usually same transforms for validation

    print("Applying Transforms:")
    print(f"Train: {train_transform}")
    print(f"Val: {val_transform}")

    # --- Data Loaders ---
    train_dataset = ImageDataset(manifest_path=args.train_manifest, transform=train_transform)
    val_dataset = ImageDataset(manifest_path=args.val_manifest, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    # --- Model ---
    model = SmileClassifier(learning_rate=args.learning_rate, max_epochs=args.epochs)

    # --- Logging & Checkpointing ---
    log_dir = os.path.join(args.log_dir, args.exp_name)
    logger = TensorBoardLogger(save_dir=log_dir, name="lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename='smile-resnet18-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc', # Monitor validation accuracy
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(
        monitor='val_acc', # Stop if validation accuracy doesn't improve
        patience=5,        # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='max'
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        deterministic=False, # Deterministic can slow down training, set False for speed
        log_every_n_steps=50 # Log more frequently
    )

    # --- Training ---
    print(f"Starting smile classification training for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")

    # --- Optional: Test after training ---
    # if args.test_manifest:
    #     print("Running test phase...")
    #     test_dataset = ImageDataset(manifest_path=args.test_manifest, transform=val_transform)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #     trainer.test(model, dataloaders=test_loader, ckpt_path='best') # Load best checkpoint for testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-18 Smile Classifier on CelebA')

    # Data args
    parser.add_argument('--train_manifest', type=str, required=True, help='Path to the training manifest CSV file (e.g., datasets/manifests/humor/train_smile.csv)')
    parser.add_argument('--val_manifest', type=str, required=True, help='Path to the validation manifest CSV file (e.g., datasets/manifests/humor/val_smile.csv)')
    # parser.add_argument('--test_manifest', type=str, default=None, help='Optional path to the test manifest CSV file') # Add if testing needed

    # Training args
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')


    # Logging args
    parser.add_argument('--log_dir', type=str, default='training_logs_smile', help='Directory to save logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='smile_resnet18', help='Experiment name for logging')

    args = parser.parse_args()
    main(args)
