import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse
from datasets.video_dataset import VideoDataset # Assuming video_dataset.py is in datasets/
import os

class R3D18Classifier(pl.LightningModule):
    """
    PyTorch Lightning module for training an R3D-18 classifier.
    """
    def __init__(self, num_classes, learning_rate=1e-4, max_epochs=10):
        super().__init__()
        self.save_hyperparameters() # Saves num_classes, learning_rate, max_epochs

        # Load pre-trained R3D-18 model
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)

        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Cosine Annealing LR Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

def main(args):
    pl.seed_everything(42, workers=True) # for reproducibility

    # --- Data Loaders ---
    train_dataset = VideoDataset(manifest_path=args.train_manifest, clip_length=16, target_size=112)
    val_dataset = VideoDataset(manifest_path=args.val_manifest, clip_length=16, target_size=112) # Use same transforms for val usually

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model = R3D18Classifier(num_classes=args.num_classes, learning_rate=args.learning_rate, max_epochs=args.epochs)

    # --- Logging & Checkpointing ---
    log_dir = os.path.join(args.log_dir, args.exp_name)
    logger = TensorBoardLogger(save_dir=log_dir, name="lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename='r3d18-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # Adjust for multi-GPU if needed
        deterministic=True, # For reproducibility
        log_every_n_steps=10
    )

    # --- Training ---
    print(f"Starting training for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train R3D-18 Classifier')

    # Data args
    parser.add_argument('--train_manifest', type=str, required=True, help='Path to the training manifest CSV file')
    parser.add_argument('--val_manifest', type=str, required=True, help='Path to the validation manifest CSV file')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of target classes')

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')

    # Logging args
    parser.add_argument('--log_dir', type=str, default='training_logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='r3d18_experiment', help='Experiment name for logging')

    args = parser.parse_args()
    main(args)
