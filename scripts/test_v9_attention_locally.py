#!/usr/bin/env python3
"""
Local testing version of the attention model with mocked data
"""

import os
import numpy as np
import tensorflow as tf
import tempfile
import argparse
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Bidirectional, LSTM, LayerNormalization,
    Attention, GlobalAveragePooling1D, Masking
)

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

class CosineDecayWithWarmup(Callback):
    """
    Combined learning rate scheduler with:
    1. Linear warmup for initial epochs
    2. ReduceLROnPlateau for middle epochs
    3. Cosine decay for final epochs
    """
    def __init__(
        self,
        min_lr=5e-6,
        max_lr=3e-4,
        warmup_epochs=8,
        cosine_decay_start=20,
        total_epochs=100,
        verbose=1
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_decay_start = cosine_decay_start
        self.total_epochs = total_epochs
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        # Start with minimum learning rate
        self.model.optimizer.learning_rate.assign(self.min_lr)
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: CosineDecayWithWarmup setting learning rate to {warmup_lr:.6f} (warmup phase).")


def generate_mock_data(num_samples=20, sequence_length=100, feature_dim=768, num_classes=6):
    """
    Generate synthetic data for testing the model
    """
    print(f"Generating {num_samples} mock samples with shape ({sequence_length}, {feature_dim})")
    
    # Create features with varying sequence lengths
    features = []
    max_seq_len = sequence_length * 2  # Some sequences will be longer than our target padding
    for i in range(num_samples):
        # Random sequence length between 10 and max_seq_len
        seq_len = np.random.randint(10, max_seq_len)
        # Create random feature vectors
        feature = np.random.randn(seq_len, feature_dim).astype(np.float32)
        features.append(feature)
    
    # Create random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    return features, labels


def pad_sequences(features, max_length=None):
    """
    Pad the sequences to a fixed length
    """
    if max_length is None:
        # Use the 98th percentile length
        lengths = [len(f) for f in features]
        max_length = int(np.percentile(lengths, 98))
    
    print(f"Padding sequences to length {max_length}")
    
    # Get feature dimension
    feature_dim = features[0].shape[1]
    
    # Initialize output array
    padded_features = np.zeros((len(features), max_length, feature_dim))
    
    # Fill with actual data (truncate if needed)
    for i, feature in enumerate(features):
        seq_length = min(len(feature), max_length)
        padded_features[i, :seq_length, :] = feature[:seq_length]
    
    return padded_features, max_length


def build_model(input_shape, num_classes):
    """
    Build the model with attention mechanism
    """
    print(f"Building model with {num_classes} output classes...")
    
    # Input layer with masking to handle variable-length sequences
    input_layer = Input(shape=input_shape, name="input_layer")
    masked_input = Masking(mask_value=0.0)(input_layer)
    
    # Bidirectional LSTM with recurrent dropout for regularization
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25))(masked_input)
    
    # Add Layer Normalization for more stable training
    x = LayerNormalization()(x)
    
    # Self-attention mechanism to focus on important frames
    context_vector = Attention(use_scale=True)([x, x])
    
    # Global pooling to convert attention output to fixed-length representation
    x = GlobalAveragePooling1D()(context_vector)
    
    # Dense layers with stronger regularization
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Starting with min LR
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    return model


def test_forward_pass(model, input_data, labels):
    """
    Run a single forward pass to verify model compiles and runs
    """
    print("Testing forward pass...")
    
    # Make a prediction
    predictions = model.predict(input_data[:1])
    
    # Run a single training step
    history = model.fit(
        input_data[:2], 
        labels[:2],
        epochs=1,
        batch_size=2,
        verbose=1
    )
    
    print("Forward pass successful!")
    return True


def save_model_test(model, temp_dir):
    """
    Test model saving functionality
    """
    print(f"Testing model save to {temp_dir}...")
    model_path = os.path.join(temp_dir, "test_model.h5")
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Local testing for WAV2VEC attention model')
    parser.add_argument('--samples', type=int, default=20, help='Number of mock samples to generate')
    parser.add_argument('--seq_length', type=int, default=100, help='Target sequence length')
    parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension')
    parser.add_argument('--classes', type=int, default=6, help='Number of emotion classes')
    
    args = parser.parse_args()
    
    # Create a temporary directory for model saving tests
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Generate mock data
        features, labels = generate_mock_data(
            num_samples=args.samples,
            sequence_length=args.seq_length,
            feature_dim=args.feature_dim,
            num_classes=args.classes
        )
        
        # Pad sequences
        X, max_length = pad_sequences(features)
        
        # Print shapes
        print(f"Input shape: {X.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Build model
        input_shape = (max_length, args.feature_dim)
        model = build_model(input_shape, args.classes)
        
        # Test forward pass
        test_forward_pass(model, X, labels)
        
        # Test model saving
        save_model_test(model, temp_dir)
        
        print("\n===== All tests passed! =====")
        print("The model architecture is valid and can be deployed.")


if __name__ == "__main__":
    main()
