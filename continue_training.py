#!/usr/bin/env python
# continue_training.py - Script to continue training models from checkpoints
# This script should be placed on the AWS instance to enable continued training

import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Continue training emotion recognition models')
    parser.add_argument('--model', type=str, required=True, help='Model name to continue training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of additional epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directory containing training and validation data')
    return parser.parse_args()

def load_latest_checkpoint(model_name, checkpoint_dir):
    """Load the latest checkpoint for the specified model"""
    model_checkpoints = os.path.join(checkpoint_dir, model_name)
    
    if not os.path.exists(model_checkpoints):
        raise ValueError(f"No checkpoints found for model {model_name} in {checkpoint_dir}")
    
    checkpoints = [cp for cp in os.listdir(model_checkpoints) if cp.endswith('.h5')]
    if not checkpoints:
        raise ValueError(f"No .h5 checkpoint files found for model {model_name}")
    
    # Sort checkpoints by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(model_checkpoints, x)), reverse=True)
    latest_checkpoint = os.path.join(model_checkpoints, checkpoints[0])
    
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    return load_model(latest_checkpoint, compile=True)

def load_dataset(model_name, data_dir):
    """Load training and validation datasets for continued training"""
    # For simplicity, we'll assume the datasets are NPZ files
    # In a real implementation, this would load the appropriate preprocessed datasets
    
    train_data_path = os.path.join(data_dir, f"{model_name}_train.npz")
    val_data_path = os.path.join(data_dir, f"{model_name}_val.npz")
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        # If model-specific data doesn't exist, fall back to common datasets
        train_data_path = os.path.join(data_dir, "emotion_train.npz")
        val_data_path = os.path.join(data_dir, "emotion_val.npz")
    
    print(f"Loading training data from: {train_data_path}")
    print(f"Loading validation data from: {val_data_path}")
    
    # Load data
    train_data = np.load(train_data_path)
    val_data = np.load(val_data_path)
    
    # Extract features and labels
    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = val_data['features'], val_data['labels']
    
    return (X_train, y_train), (X_val, y_val)

def continue_training(model, train_data, val_data, model_name, epochs, batch_size, checkpoint_dir):
    """Continue training the model with additional epochs"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Setup checkpoint callback
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_checkpoint_dir, f"{model_name}_continued_{{epoch:02d}}.h5"),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # Setup early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Setup TensorBoard callback
    log_dir = f"./logs/{model_name}_continued_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Continue training
    print(f"Continuing training of {model_name} for {epochs} additional epochs")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback, early_stopping, tensorboard_callback],
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(model_checkpoint_dir, f"{model_name}_final_continued.h5")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return history

def main():
    args = parse_args()
    
    # Load the model from checkpoint
    model = load_latest_checkpoint(args.model, args.checkpoint_dir)
    
    # Load the datasets
    train_data, val_data = load_dataset(args.model, args.data_dir)
    
    # Continue training
    history = continue_training(
        model, 
        train_data, 
        val_data, 
        args.model, 
        args.epochs, 
        args.batch_size, 
        args.checkpoint_dir
    )
    
    # Print final results
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final validation accuracy after continuation: {final_val_acc:.4f}")
    print(f"Final validation loss after continuation: {final_val_loss:.4f}")

if __name__ == "__main__":
    main()
