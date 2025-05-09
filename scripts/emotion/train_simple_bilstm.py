#!/usr/bin/env python3
"""
Improved wav2vec emotion recognition using BiLSTM.
This builds on the original logistic regression baseline but maintains
temporal structure of wav2vec features using BiLSTM.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import argparse
import itertools

# Suppress TF debug logs
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
MAX_SEQ_LENGTH = 200  # Pad/truncate sequences to this length
EMBED_DIM = 768      # wav2vec feature dimension
BATCH_SIZE = 16      # Smaller batch for more stable gradients
EPOCHS = 30          # More epochs with early stopping
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.5   # Increased dropout for better regularization
L2_REG = 0.001       # L2 regularization strength
LEARNING_RATE = 5e-4 # Slightly lower learning rate

def load_data(data_dirs, class_map=None):
    """Load wav2vec feature files and prepare for training."""
    
    # Default class mapping
    if class_map is None:
        class_map = {0: "angry", 1: "disgust", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad"}
    
    # Inverse mapping
    label_to_id = {v: k for k, v in class_map.items()}
    
    # Find all npz files
    files = list(itertools.chain.from_iterable(
        glob.glob(os.path.join(d, "**", "*.npz"), recursive=True) 
        for d in data_dirs
    ))
    
    features, labels = [], []
    sequence_lengths = []
    
    for file_path in files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Skip files without wav2vec features
            if "wav2vec_features" not in data:
                continue
                
            # Get features
            wav_features = data["wav2vec_features"].astype(np.float32)
            sequence_lengths.append(len(wav_features))
            
            # Extract label
            label = None
            if "emotion" in data:
                # String label
                label_str = str(data["emotion"])
                if label_str in label_to_id:
                    label = label_to_id[label_str]
            elif "emotion_class" in data:
                # Integer label
                label = int(data["emotion_class"])
            elif "label" in data and data["label"].size == len(class_map):
                # One-hot encoded label
                label = int(data["label"].argmax())
                
            # Skip if label not found or not in our class map
            if label is None or label not in class_map:
                continue
                
            features.append(wav_features)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(features)} valid samples")
    print(f"Label distribution: {collections.Counter([class_map[l] for l in labels])}")
    
    # Analyze sequence lengths
    seq_lengths = np.array(sequence_lengths)
    print(f"Sequence length stats: min={seq_lengths.min()}, "
          f"max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}, "
          f"median={np.median(seq_lengths):.1f}, "
          f"95th={np.percentile(seq_lengths, 95):.1f}")
    
    return features, np.array(labels), class_map

def pad_sequences(sequences, max_length=MAX_SEQ_LENGTH):
    """Pad or truncate sequences to fixed length."""
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) > max_length:
            # Truncate
            padded_seq = seq[:max_length]
        else:
            # Pad with zeros
            padding = np.zeros((max_length - len(seq), seq.shape[1]), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
            
        padded_sequences.append(padded_seq)
        
    return np.array(padded_sequences)

class AttentionLayer(keras.layers.Layer):
    """Custom attention layer for sequence models."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(1,),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Alignment scores
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Get attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Weighted sum (context vector)
        context = x * a
        context = tf.reduce_sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model(num_classes):
    """Build BiLSTM model with attention for sequence classification."""
    # Input layer 
    inputs = keras.layers.Input(shape=(MAX_SEQ_LENGTH, EMBED_DIM))
    
    # Normalize feature dimensions
    x = keras.layers.LayerNormalization(axis=-1)(inputs)
    
    # Initial feature transformation
    x = keras.layers.Conv1D(
        filters=EMBED_DIM, 
        kernel_size=3, 
        padding='same',
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(L2_REG)
    )(x)
    
    # Bidirectional LSTM with return_sequences=True for attention
    lstm_out = keras.layers.Bidirectional(
        keras.layers.LSTM(
            LSTM_UNITS, 
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=keras.regularizers.l2(L2_REG)
        )
    )(x)
    
    # Apply attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Regularization and dense layers
    x = keras.layers.Dropout(DROPOUT_RATE)(attention_out)
    x = keras.layers.Dense(
        DENSE_UNITS, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(L2_REG)
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('bilstm_confusion_matrix.png')
    print(f"Saved confusion matrix to bilstm_confusion_matrix.png")

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bilstm_training_history.png')
    print(f"Saved training history to bilstm_training_history.png")

def train_and_evaluate():
    """Main function to train and evaluate the model."""
    parser = argparse.ArgumentParser(description='Train BiLSTM on wav2vec features')
    parser.add_argument('--data_dirs', nargs='+', default=['crema_d_features'],
                       help='Directories containing wav2vec feature files')
    args = parser.parse_args()
    
    print(f"Loading data from: {args.data_dirs}")
    
    # Load data
    features, labels, class_map = load_data(args.data_dirs)
    
    if len(features) < 10:
        print("Not enough samples to train. Exiting.")
        return
    
    # Convert to one-hot encoding for class weights
    unique_labels = sorted(list(set(labels)))
    class_names = [class_map[i] for i in unique_labels]
    num_classes = len(unique_labels)
    
    # Compute class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    class_weight_dict = {i: w for i, w in zip(unique_labels, class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Pad sequences
    print(f"Padding sequences to length {MAX_SEQ_LENGTH}")
    padded_features = pad_sequences(features)
    print(f"Padded features shape: {padded_features.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Build and train model
    model = build_model(num_classes)
    model.summary()
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train
    print("\nTraining BiLSTM model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=[class_map[i] for i in unique_labels],
                               digits=3))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Training history
    plot_training_history(history)
    
    # Compare to baseline
    print("\nPerformance comparison:")
    print("Previous baseline (LogReg on mean features): ~0.35 accuracy")
    print(f"BiLSTM model (sequence features): {test_acc:.4f} accuracy")
    print(f"Improvement: {(test_acc - 0.35) * 100:.1f}%")
    
    # Save model if good performance
    if test_acc > 0.40:
        model.save('wav2vec_bilstm_model.keras')
        print("Model saved to wav2vec_bilstm_model.keras")

if __name__ == "__main__":
    train_and_evaluate()
