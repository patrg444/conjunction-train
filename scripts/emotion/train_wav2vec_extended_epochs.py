#!/usr/bin/env python3
# coding: utf-8
"""
Extended training for wav2vec audio emotion recognition model
This script continues training from a checkpoint for additional epochs
and includes options for hyperparameter adjustments
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, Callback
)
import matplotlib.pyplot as plt
from datetime import datetime

# Add custom layer import to ensure model loads properly
from tensorflow.keras.layers import Layer
import sys

# Import our fixed model and utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try importing SelfAttentionLayer from multiple potential locations
try:
    from fixed_v4_script import SelfAttentionLayer, improved_load_features, load_emotion_data, NaNLossDetector
except ImportError:
    from scripts.train_wav2vec_audio_only_fixed_v4 import SelfAttentionLayer, improved_load_features, load_emotion_data, NaNLossDetector

# Fixed learning rate scheduler with .value() issue corrected
class WarmUpReduceLROnPlateau(Callback):
    def __init__(
        self,
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-8,
        warmup_epochs=3,
        learning_rate_base=0.001,
        **kwargs
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.learning_rate_base = learning_rate_base

        self.monitor_op = None
        self.best = None
        self.cooldown_counter = 0
        self.wait = 0

        if mode == 'min' or (mode == 'auto' and 'loss' in monitor):
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best = -float('inf')

    def on_epoch_begin(self, epoch, logs=None):
        # Apply learning rate warm-up
        if epoch < self.warmup_epochs:
            # Ensure learning_rate is float32 for consistency
            learning_rate = tf.cast(self.learning_rate_base * (epoch + 1) / self.warmup_epochs, dtype=tf.float32)
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau setting learning rate to {learning_rate.numpy()}.')
            # Directly assign the learning rate value
            self.model.optimizer.learning_rate.assign(learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if epoch < self.warmup_epochs:
            return  # Skip LR reduction during warm-up

        if current is None:
            if self.verbose > 0:
                print(f'\nWarmUpReduceLROnPlateau conditioned on metric `{self.monitor}` which is not available. Available metrics are: {", ".join(list(logs.keys()))}')
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                # Get current LR value correctly - FIXED for TF compatibility
                # Using float() instead of value() for ResourceVariable access
                old_lr = float(self.model.optimizer.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    # Ensure new_lr is float32
                    new_lr = tf.cast(max(new_lr, self.min_lr), dtype=tf.float32)
                    # Assign the new learning rate value
                    self.model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau reducing learning rate to {new_lr.numpy()}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

def continue_training(args):
    """Continue training the wav2vec emotion recognition model from a checkpoint"""
    # Parameters
    data_dir = args.data_dir
    model_weights = args.weights_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    dropout_rate = args.dropout
    extended_model_name = f"{args.model_name}_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and prepare data
    print(f"Loading data from {data_dir}")
    X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length = load_emotion_data(
        data_dir, test_size=0.1, use_cache=args.use_cache
    )

    # Load model architecture from the fixed script
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Register custom objects for model loading
    tf.keras.utils.get_custom_objects()['SelfAttentionLayer'] = SelfAttentionLayer

    # Custom model loading approach
    if os.path.exists(model_weights):
        # Build the model architecture first
        from fixed_v4_script import build_model
        model = build_model(input_shape, num_classes, dropout_rate)
        
        # Load weights
        print(f"Loading weights from {model_weights}")
        model.load_weights(model_weights)
    else:
        print(f"Error: Model weights file {model_weights} not found")
        return

    # Setup optimizer with gradient clipping
    optimizer = SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0
    )

    # Recompile model - with label smoothing for stability
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{extended_model_name}_best.weights.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{extended_model_name}_final.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch'
        ),
        TensorBoard(
            log_dir=os.path.join("logs", extended_model_name),
            histogram_freq=1,
            update_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        WarmUpReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            warmup_epochs=3,  # Shorter warm-up for continued training
            learning_rate_base=learning_rate
        ),
        NaNLossDetector()
    ]

    # Train model
    print(f"Starting extended training: {extended_model_name}")
    print(f"Input shape: {input_shape}, Classes: {num_classes}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save history for later analysis
    history_file = os.path.join(checkpoint_dir, f"{extended_model_name}_history.json")
    with open(history_file, 'w') as f:
        history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
        json.dump(history_dict, f)

    # Plot training curves
    if not args.no_plots:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, f"{extended_model_name}_training_curves.png"))

    # Final evaluation
    print("\nEvaluating model on validation set:")
    results = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)
    print(f"Validation loss: {results[0]:.4f}, Validation accuracy: {results[1]:.4f}")

    # Save validation results in a summary file
    with open(os.path.join(checkpoint_dir, f"{extended_model_name}_validation_summary.csv"), 'w') as f:
        f.write("metric,value\n")
        f.write(f"val_loss,{results[0]}\n")
        f.write(f"val_accuracy,{results[1]:.4f}\n")

    # Also save as a json file
    with open(os.path.join(checkpoint_dir, f"{extended_model_name}_validation_accuracy.json"), 'w') as f:
        json.dump({"val_accuracy": float(results[1])}, f)

    print(f"Extended training complete. Best weights saved to {checkpoint_dir}/{extended_model_name}_best.weights.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training wav2vec-based emotion recognition model")
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/audio_emotion/models/wav2vec",
                        help="Directory containing wav2vec features")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to model weights file (.weights.h5)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Additional training epochs")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Initial learning rate (lower than original)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--model_name", type=str, default="wav2vec_extended",
                        help="Custom model name prefix")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable training curve plots")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cached pre-processed numpy arrays")
    
    args = parser.parse_args()
    continue_training(args)
