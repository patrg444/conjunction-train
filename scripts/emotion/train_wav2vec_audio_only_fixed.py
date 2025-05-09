#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Audio-only emotion recognition model using wav2vec2 embeddings.
Implements aggressive numerical stability safeguards to prevent NaN/Inf values.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Enable operation determinism for reproducibility
tf.config.experimental.enable_op_determinism()

# Explicitly set to float32 for numerical stability
tf.keras.mixed_precision.set_global_policy("float32")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import Masking, GlobalAveragePooling1D, LayerNormalization, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, Callback
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
import json
import math
from tqdm import tqdm
import sklearn.metrics as metrics
from datetime import datetime

# Import our custom fixed loader
# This module should be in the same directory
from scripts import wav2vec_fixed_loader as loader

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class F1ScoreCallback(Callback):
    """
    Callback to calculate F1 score after each epoch
    """
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.val_f1s = []
        
    def on_epoch_end(self, epoch, logs={}):
        # Get predictions
        val_x, val_y = self.validation_data
        val_predict = np.argmax(self.model.predict(val_x), axis=1)
        val_true = np.argmax(val_y, axis=1)
        
        # Calculate F1 score (macro avg)
        _val_f1 = metrics.f1_score(val_true, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        logs['val_f1'] = _val_f1
        print(f" — val_f1: {_val_f1:.4f}")

class NaNLossDetector(Callback):
    """
    Callback to detect NaN losses and stop training
    """
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(f'NaN/Inf loss detected: {loss} at batch {batch}, stopping training.')
            self.model.stop_training = True
            
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Current learning rate: {float(self.model.optimizer.learning_rate.numpy()):.8f}")

class WarmUpCosineDecayScheduler(Callback):
    """ 
    Implements warm-up with cosine decay learning rate scheduling 
    """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-7):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []
    
    def _set_lr(self, learning_rate):
        """Safely set learning rate for different optimizer types"""
        opt = self.model.optimizer
        # Handle both tf.Variable and plain Python types
        if isinstance(opt.learning_rate, tf.Variable):
            tf.keras.backend.set_value(opt.learning_rate, learning_rate)
        else:
            # For plain Python types, direct assignment
            opt.learning_rate = learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
            
        if epoch < self.warmup_epochs:
            # Linear warm-up phase
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay phase
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        
        # Ensure numeric stability - no too small or NaN values
        learning_rate = max(learning_rate, self.min_learning_rate)
        if math.isnan(learning_rate) or math.isinf(learning_rate):
            print(f"Warning: Invalid learning rate {learning_rate}, using minimum value")
            learning_rate = self.min_learning_rate
            
        # Set learning rate using the safe method
        self._set_lr(learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.8f}")

class AttentionPooling(tf.keras.layers.Layer):
    """
    Attention-based pooling layer that respects masking.
    Numerically stabilized implementation to avoid NaN/Inf.
    """
    def __init__(self, units=64, epsilon=1e-7):
        super(AttentionPooling, self).__init__()
        self.units = units
        self.epsilon = epsilon
        self.attention_weights = None  # Store for visualization
    
    def build(self, input_shape):
        self.w = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            constraint=MaxNorm(3.0)  # Constrain weight magnitude
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            constraint=MaxNorm(3.0)  # Constrain weight magnitude
        )
        super(AttentionPooling, self).build(input_shape)
    
    def compute_mask(self, inputs, mask=None):
        # Output does not need a mask
        return None
    
    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, features)
        return (input_shape[0], input_shape[2])
        
    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, features)
        dtype = inputs.dtype
        
        # Ensure inputs are finite (safety check)
        inputs = tf.debugging.check_numerics(inputs, "NaN/Inf found in attention inputs")
        
        # Linear projection with tanh activation
        uit = tf.tensordot(inputs, self.w, axes=1) + self.b  # (batch_size, time_steps, units)
        uit = tf.nn.tanh(uit)
        uit = tf.debugging.check_numerics(uit, "NaN/Inf in attention projection")
        
        # Compute attention scores
        scores = tf.tensordot(uit, self.u, axes=1)  # (batch_size, time_steps, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch_size, time_steps)
        scores = tf.debugging.check_numerics(scores, "NaN/Inf in attention scores")
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask from boolean to float with matching dtype
            mask = tf.cast(mask, dtype=dtype)
            # Set attention scores for masked timesteps to large negative value
            scores = scores + (1.0 - mask) * tf.constant(-1e9, dtype=dtype)
        
        # Compute attention weights with softmax (safe from overflow/underflow)
        scores_max = tf.reduce_max(scores, axis=1, keepdims=True)
        scores_shifted = scores - scores_max  # Subtract max for stability
        attention_weights = tf.nn.softmax(scores_shifted, axis=1)  # (batch_size, time_steps)
        attention_weights = tf.debugging.check_numerics(attention_weights, "NaN/Inf in attention weights")
        
        self.attention_weights = attention_weights  # Store for later inspection
        
        # Apply attention weights to input sequence
        context = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)  # (batch_size, features)
        context = tf.debugging.check_numerics(context, "NaN/Inf in attention context")
        
        return context

def create_audio_only_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Create a numerically stable audio-only model for emotion recognition
    using a simplified architecture with aggressive regularization.

    Args:
        input_shape: Shape of the audio embedding input (time_steps, embedding_dim)
        num_classes: Number of emotion classes to predict
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='audio_input')
    
    # Masking layer to handle variable-length sequences
    x = Masking(mask_value=0.0)(inputs)
    
    # Use LayerNormalization instead of BatchNormalization (no running stats)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # First Bi-LSTM layer with return sequences
    # Smaller size than original to reduce potential for overflow
    x = Bidirectional(LSTM(64, 
                         return_sequences=True, 
                         dropout=0.0,  # Apply dropout separately 
                         recurrent_dropout=0.0,  # Avoid recurrent dropout
                         kernel_constraint=MaxNorm(3.0),
                         recurrent_constraint=MaxNorm(3.0),
                         kernel_initializer='glorot_uniform'))(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Second LSTM layer (non-bidirectional for simplicity)
    x = LSTM(64, 
           return_sequences=True,
           dropout=0.0,
           recurrent_dropout=0.0,
           kernel_constraint=MaxNorm(3.0),
           recurrent_constraint=MaxNorm(3.0),
           kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Attention-based pooling - reduced size for stability
    x = AttentionPooling(units=32)(x)
    
    # Dense layer with LayerNormalization
    x = Dense(32, 
             activation='relu', 
             kernel_initializer='he_normal',
             kernel_constraint=MaxNorm(3.0))(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(dropout_rate)(x)
    
    # Clip logits before softmax for numerical stability
    x = Dense(num_classes, kernel_initializer='glorot_uniform')(x)
    x = Lambda(lambda z: tf.clip_by_value(z, -15.0, 15.0), name='logit_clipping')(x)
    
    # Output layer with softmax activation
    outputs = Lambda(lambda z: tf.nn.softmax(z), name='softmax_stable')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train robust audio-only emotion recognition model')
    
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing wav2vec feature files')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to pre-computed mean values for normalization')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to pre-computed std values for normalization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate (will be decreased for warmup)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Model checkpoint directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra logging')
    parser.add_argument('--clip_value', type=float, default=5.0,
                        help='Clip input features to this value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    global RANDOM_SEED
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("Robust Audio-only Wav2Vec Emotion Recognition Training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Enable operation determinism for debugging (helps with reproducing NaNs)
    tf.config.experimental.enable_op_determinism()
    
    # Enable numeric check if in debug mode
    if args.debug:
        tf.debugging.enable_check_numerics()
        print("Numeric checking enabled - will detect NaN/Inf during execution")
    
    # Create output directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load normalization statistics
    if not args.mean_path or not args.std_path:
        print("Error: You must provide mean_path and std_path for normalization.")
        sys.exit(1)
        
    if not os.path.exists(args.mean_path) or not os.path.exists(args.std_path):
        print(f"Error: Normalization files not found at {args.mean_path} or {args.std_path}")
        sys.exit(1)
    
    print(f"Loading normalization stats from {args.mean_path} and {args.std_path}")
    mean = np.load(args.mean_path)
    std = np.load(args.std_path)
    
    # Load dataset files using our robust loader
    audio_files, labels = loader.load_dataset_files(args.features_dir)
    
    if len(audio_files) == 0:
        print("Error: No audio files found. Check the features_dir path.")
        sys.exit(1)
    
    # Create dataset with robust loader
    train_ds, val_ds, num_classes, class_weights = loader.create_tf_dataset(
        audio_files, labels, mean, std, 
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_len=None  # Use dynamic length
    )
    
    # Create model - get input shape from the mean array
    model = create_audio_only_model(
        input_shape=(None, mean.shape[0]),
        num_classes=num_classes,
        dropout_rate=0.5
    )
    
    # Use SGD with momentum for stability
    learning_rate = args.lr / 100  # Start with 100x smaller learning rate
    
    optimizer = SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0,  # Clip gradients by norm
        clipvalue=0.5  # Also clip by value
    )
    
    # Compile with label smoothing for numerical stability
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Test run with random data to check for potential issues
    print("\nTesting forward pass with random data...")
    batch_shape = (args.batch_size, 100, mean.shape[0])
    test_input = tf.random.normal(batch_shape, mean=0.0, stddev=0.1)
    test_labels = tf.one_hot(tf.random.uniform(
        shape=(args.batch_size,), 
        minval=0, 
        maxval=num_classes-1, 
        dtype=tf.int32
    ), depth=num_classes)
    
    # Try forward and backward pass
    try:
        with tf.GradientTape() as tape:
            test_output = model(test_input, training=True)
            test_loss = model.loss(test_labels, test_output)
            
        print(f"  Forward pass successful: Loss={test_loss.numpy():.6f}")
        print(f"  Predictions shape: {test_output.shape}")
        
        # Check gradients
        grads = tape.gradient(test_loss, model.trainable_variables)
        grads_ok = True
        for g in grads:
            if g is not None and (tf.math.reduce_any(tf.math.is_nan(g)) or tf.math.reduce_any(tf.math.is_inf(g))):
                grads_ok = False
                break
                
        print(f"  Gradients OK: {grads_ok}")
        
        if not grads_ok:
            print("Warning: NaNs detected in gradients during test. Model may be numerically unstable.")
    except Exception as e:
        print(f"Error during test forward pass: {e}")
        print("Proceeding with training anyway but watch for numerical issues.")
    
    # Model summary
    model.summary()
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_fixed_{timestamp}_best.weights.h5")
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Increased patience
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(args.log_dir, f"wav2vec_audio_only_fixed_{timestamp}"),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Aggressive reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Warm-up LR schedule
        WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_epochs=args.epochs,
            warmup_epochs=10,  # Extended warm-up period
            min_learning_rate=1e-7
        ),
        # NaN detector
        NaNLossDetector(),
    ]
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    # Get a single batch for the F1 callback
    for x_val, y_val in val_ds.take(1):
        val_data = (x_val, y_val)
        callbacks.append(F1ScoreCallback(val_data))
        break
    
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=dict(enumerate(class_weights)),
        verbose=1
    )
    
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes.")
    
    # Save model - both weights and architecture
    model_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_fixed_{timestamp}_final")
    model.save(model_path, save_format='tf')
    print(f"Final model saved to {model_path}")
    
    # Save weights separately for flexibility
    final_weights_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_fixed_{timestamp}_final.weights.h5")
    model.save_weights(final_weights_path)
    print(f"Final weights saved to {final_weights_path}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_fixed_{timestamp}_history.json")
    with open(history_path, 'w') as f:
        history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
        json.dump(history_dict, f, indent=4)
    
    print(f"Training history saved to {history_path}")
    print("\nRun TensorBoard with:")
    print(f"tensorboard --logdir={args.log_dir}")
    
    # Final evaluation on validation set
    print("\nEvaluating best model on validation set...")
    model.load_weights(checkpoint_path)
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
