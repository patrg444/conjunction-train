#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Attention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def load_wav2vec_data(data_dirs, split_ratio=0.8):
    """Load wav2vec features from specified directories."""
    all_data = []
    all_labels = []
    emotion_mapping = {
        'ANG': 0,  # Angry
        'DIS': 1,  # Disgust
        'FEA': 2,  # Fear
        'HAP': 3,  # Happy
        'SAD': 4,  # Sad
        'NEU': 5,  # Neutral
        'CAL': 5   # Calm (mapped to same as Neutral)
    }
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']

    print(f"Loading data from directories: {data_dirs}")

    for data_dir in data_dirs:
        npz_files = glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
        print(f"Found {len(npz_files)} files in {data_dir}")

        if not npz_files:
            print(f"WARNING: No .npz files found in {data_dir}")
            continue

        for npz_file in npz_files:
            try:
                filename = os.path.basename(npz_file)
                
                # Try to determine emotion from filename for different datasets
                emotion_code = None
                if 'cremad_' in filename and '_' in filename:
                    # Format: cremad_XXXX_XXX_EMO_XX.npz
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        emotion_code = parts[3]
                elif 'ravdess_' in filename:
                    # Format: ravdess_XX-XX-XX-XX-XX-XX-XX.npz
                    # where the third number is emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust)
                    parts = filename.replace('.npz', '').split('-')
                    if len(parts) >= 7:
                        ravdess_mapping = {'01': 5, '02': 5, '03': 3, '04': 4, '05': 0, '06': 2, '07': 1}
                        emotion_code = ravdess_mapping.get(parts[2], None)
                    
                # Now load the data
                data = np.load(npz_file, allow_pickle=True)
                
                # Try different feature keys
                feature_key = None
                for key in data.keys():
                    if 'feature' in key.lower() or 'embedding' in key.lower() or key == 'wav2vec':
                        feature_key = key
                        break
                
                if feature_key is None:
                    # Try common keys
                    if 'wav2vec_features' in data.keys():
                        feature_key = 'wav2vec_features'
                    elif 'x' in data.keys():
                        feature_key = 'x'
                    elif 'features' in data.keys():
                        feature_key = 'features'
                    elif len(data.keys()) == 1:
                        # If there's only one key, use it
                        feature_key = list(data.keys())[0]
                
                if feature_key is None:
                    print(f"Skipping {filename} - could not find feature data in keys: {data.keys()}")
                    continue
                
                features = data[feature_key]
                
                # If emotion_code is None, try to get it from data
                if emotion_code is None:
                    if 'emotion_class' in data:
                        emotion_code = data['emotion_class'].item() if hasattr(data['emotion_class'], 'item') else data['emotion_class']
                    elif 'emotion' in data:
                        emotion_str = str(data['emotion'])
                        if emotion_str not in emotion_mapping:
                            # Skip if we don't recognize this emotion
                            print(f"Skipping {filename} - unknown emotion format: {emotion_str}")
                            continue
                        emotion_code = emotion_mapping[emotion_str]
                
                # Get label from emotion_code
                if isinstance(emotion_code, str):
                    label = emotion_mapping.get(emotion_code, None)
                else:
                    label = emotion_code
                
                if label is not None:
                    all_data.append(features)
                    all_labels.append(label)
                else:
                    print(f"Skipping {filename} - could not determine emotion class")
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")

    if not all_data:
        raise ValueError("No valid data was loaded")

    # Convert to arrays
    X = all_data
    y = np.array(all_labels)

    # Print data stats
    print(f"Loaded {len(X)} samples with {len(set(y))} emotion classes")
    for emotion, count in zip(*np.unique(y, return_counts=True)):
        print(f"  Class {emotion}: {count} samples")

    # Split into train/validation sets
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(indices) * split_ratio)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Create training and validation sets
    X_train = [X[i] for i in train_indices]
    y_train = y[train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = y[val_indices]

    # Pad sequences for batch processing
    # Find the maximum sequence length
    max_length = max(x.shape[0] for x in X)
    print(f"Maximum sequence length: {max_length}")

    # Pad all sequences to the same length
    X_train_padded = np.zeros((len(X_train), max_length, X_train[0].shape[1]))
    X_val_padded = np.zeros((len(X_val), max_length, X_val[0].shape[1]))

    for i, x in enumerate(X_train):
        X_train_padded[i, :x.shape[0], :] = x

    for i, x in enumerate(X_val):
        X_val_padded[i, :x.shape[0], :] = x

    # Convert labels to one-hot encoding
    num_classes = len(set(y))
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)

    return (X_train_padded, y_train_onehot), (X_val_padded, y_val_onehot), num_classes


def create_attention_crnn_model(input_shape, num_classes):
    """Create a CNN+RNN+Attention model for wav2vec features."""
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN layers for feature extraction
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)

    # Bidirectional LSTM for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Self-attention mechanism
    attention_output = LSTM(64, return_sequences=True)(x)
    query = Dense(64)(attention_output)
    key = Dense(64)(attention_output)
    value = Dense(64)(attention_output)

    # Attention weights
    score = tf.matmul(query, key, transpose_b=True)
    score = tf.nn.softmax(score, axis=-1)

    # Context vector
    context = tf.matmul(score, value)
    context = tf.reduce_mean(context, axis=1)

    # Classification layers
    x = Dense(128, activation='relu')(context)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('attn_crnn_confusion_matrix.png')
    print(f"Confusion matrix saved to attn_crnn_confusion_matrix.png")


def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('attn_crnn_training_history.png')
    print(f"Training history saved to attn_crnn_training_history.png")


def main():
    parser = argparse.ArgumentParser(description='Train an Attention-based CRNN model for wav2vec features')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='Directories containing wav2vec feature files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    print("Starting training with arguments:", args)

    # Load data
    (X_train, y_train), (X_val, y_val), num_classes = load_wav2vec_data(args.data_dirs)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_attention_crnn_model(input_shape, num_classes)
    model.summary()

    # Setup callbacks
    callbacks = [
        ModelCheckpoint('best_attn_crnn_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=args.patience, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_accuracy'),
        TensorBoard(log_dir=f'logs/attn_crnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")

    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Generate classification report
    class_names = [str(i) for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

    # Plot training history
    plot_training_history(history)

    print("Training complete!")


if __name__ == "__main__":
    main()
