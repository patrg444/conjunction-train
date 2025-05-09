import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

# Import the Lightning Module and Dataset defined in other files
from train_r3d18 import R3D18Classifier # Assumes train_r3d18.py is in the same directory or accessible
from datasets.video_dataset import VideoDataset # Assumes video_dataset.py is in datasets/

def evaluate_model(args):
    """Loads a checkpoint and evaluates the R3D-18 model on the test set."""

    # --- Load Model from Checkpoint ---
    # Ensure num_classes is provided if loading manually, or rely on saved hyperparameters
    # Using load_from_checkpoint handles hyperparameter loading automatically
    try:
        model = R3D18Classifier.load_from_checkpoint(args.checkpoint_path)
        model.eval() # Set model to evaluation mode
        model.freeze() # Freeze weights
        print(f"Loaded model from checkpoint: {args.checkpoint_path}")
        print(f"Model hyperparameters: {model.hparams}")
        num_classes = model.hparams.num_classes # Get num_classes from loaded model hparams
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Ensure the checkpoint path is correct and the R3D18Classifier class definition is available.")
        return

    # --- Data Loader ---
    test_dataset = VideoDataset(manifest_path=args.test_manifest, clip_length=16, target_size=112)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded test data from: {args.test_manifest}")

    # --- Evaluation Loop ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            clips, labels = batch
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(cm)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save metrics to a file
        results_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
        with open(results_path, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Test Manifest: {args.test_manifest}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
            f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
        print(f"Evaluation metrics saved to {results_path}")

        # Save confusion matrix plot
        if args.save_cm_plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            cm_plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            plt.savefig(cm_plot_path)
            print(f"Confusion matrix plot saved to {cm_plot_path}")
            # plt.show() # Optionally display the plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate R3D-18 Classifier')

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.ckpt file)')
    parser.add_argument('--test_manifest', type=str, required=True, help='Path to the test manifest CSV file')
    # num_classes is now inferred from the checkpoint's hyperparameters
    # parser.add_argument('--num_classes', type=int, required=True, help='Number of target classes (must match trained model)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results (metrics, plots)')
    parser.add_argument('--save_cm_plot', action='store_true', help='Save the confusion matrix as a PNG file')

    args = parser.parse_args()
    evaluate_model(args)
