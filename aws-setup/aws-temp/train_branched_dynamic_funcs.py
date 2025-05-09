#!/usr/bin/env python3
"""
Additional functions for the branched LSTM model with dynamic sequence length.
These functions support the dynamic batch-wise padding implemented in 
sequence_data_generator.py.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model_with_generator(model, test_generator, class_names=None, output_dir='model_evaluation/branched_6class'):
    """Evaluate the model on the test set using a data generator.

    Args:
        model: Trained Keras model.
        test_generator: Data generator for test data.
        class_names: List of class names for reporting.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate the model
    logging.info("Evaluating model on test set...")
    metrics = model.evaluate(test_generator, verbose=1)

    # Get metric names
    metric_names = model.metrics_names

    # Create dictionary of metrics
    metrics_dict = {name: value for name, value in zip(metric_names, metrics)}

    # Print metrics
    for name, value in metrics_dict.items():
        logging.info(f"Test {name}: {value:.4f}")

    # Save metrics to file
    metrics_text = "\n".join([f"{name}: {value:.4f}" for name, value in metrics_dict.items()])
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(metrics_text)

    # Generate predictions
    logging.info("Generating predictions for the test set...")
    # Need to reset the generator before prediction
    test_generator.on_epoch_end()
    
    # Collect all batches of true labels and predictions
    y_true_all = []
    y_pred_all = []
    
    for i in range(len(test_generator)):
        batch_inputs, batch_labels = test_generator[i]
        batch_preds = model.predict(batch_inputs, verbose=0)
        
        y_true_all.append(batch_labels)
        y_pred_all.append(batch_preds)
    
    # Concatenate all batches
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Generate confusion matrix and classification report
    try:
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Generate classification report
        if class_names:
            report = classification_report(y_true_classes, y_pred_classes,
                                          target_names=class_names, zero_division=0)
        else:
            report = classification_report(y_true_classes, y_pred_classes, zero_division=0)

        # Save classification report
        report_file = os.path.join(output_dir, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)

        # Plot confusion matrix
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 8))

            if class_names:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()

            cm_file = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_file)
            plt.close()
        except Exception as e:
            logging.warning(f"Could not create confusion matrix plot: {str(e)}")

    except Exception as e:
        logging.warning(f"Could not generate classification metrics: {str(e)}")

    return metrics_dict

def train_model_with_generator(model, train_generator, val_generator, model_dir='models/branched_6class',
                epochs=100, class_names=None, class_weights=None):
    """Train the model using data generators with dynamic padding.

    Args:
        model: Compiled Keras model.
        train_generator: Data generator for training data.
        val_generator: Data generator for validation data.
        model_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        class_names: List of class names for reporting.
        class_weights: Dictionary of class weights for handling imbalance.

    Returns:
        Tuple of (trained_model, history).
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Define callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]

    # Train the model using generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Save training history
    try:
        import matplotlib.pyplot as plt

        # Create plots directory
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        history_path = os.path.join(plots_dir, 'training_history.png')
        plt.savefig(history_path)
        plt.close()
        logging.info(f"Training history plot saved to {history_path}")
        
        # Plot F1 Score if available
        if 'f1_score' in history.history and 'val_f1_score' in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['f1_score'])
            plt.plot(history.history['val_f1_score'])
            plt.title('Model F1 Score')
            plt.ylabel('F1 Score')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            f1_history_path = os.path.join(plots_dir, 'f1_score_history.png')
            plt.savefig(f1_history_path)
            plt.close()
            logging.info(f"F1 score history plot saved to {f1_history_path}")
            
    except Exception as e:
        logging.warning(f"Could not create training history plot: {str(e)}")

    return model, history
