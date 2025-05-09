"""
Visualization utilities for emotion comparison framework.
Provides functions to generate confusion matrices, attention heatmaps,
and consolidated HTML reports.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import base64
from io import BytesIO
import pandas as pd
import json
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """
    Plot and save a confusion matrix visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title for the plot
    
    Returns:
        fig: The matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    return fig

def plot_attention_heatmap(image, attention_weights, title='Attention Heatmap'):
    """
    Plot attention weights as a heatmap overlaid on an image.
    
    Args:
        image: The input image
        attention_weights: Attention weights for regions
        title: Title for the plot
    
    Returns:
        fig: The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the original image
    ax.imshow(image)
    
    # Overlay attention heatmap
    attention_reshaped = attention_weights.reshape(image.shape[0], image.shape[1])
    ax.imshow(attention_reshaped, cmap='jet', alpha=0.5)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance, feature_names, title='Feature Importance'):
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        title: Title for the plot
    
    Returns:
        fig: The matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.bar(range(len(indices)), feature_importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, min(20, len(indices))])  # Show top 20 features
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    
    return fig

def plot_accuracy_comparison(results_dict, title='Method Comparison'):
    """
    Plot comparison of accuracies across methods.
    
    Args:
        results_dict: Dictionary with method names as keys and accuracy values as values
        title: Title for the plot
    
    Returns:
        fig: The matplotlib figure
    """
    methods = list(results_dict.keys())
    accuracies = [results_dict[method] for method in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = plt.bar(methods, accuracies, color='skyblue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.ylim(0, max(accuracies) * 1.15)  # Add some space at the top
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.tight_layout()
    
    return fig

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_html_report(output_dir, dataset_name='RAVDESS'):
    """
    Generate a comprehensive HTML report for a dataset comparison.
    
    Args:
        output_dir: Directory with results
        dataset_name: Name of the dataset
    
    Returns:
        report_path: Path to the generated HTML report
    """
    # Load summary data
    summary_path = os.path.join(output_dir, 'comparison_summary.txt')
    
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found at {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        summary_content = f.read()
    
    # Extract results data
    methods = []
    method_results = {}
    
    # Look for feature directories
    for item in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, item)) and "_features" in item:
            method_name = item.replace("_features", "")
            methods.append(method_name)
    
    # Find results for each method
    for method in methods:
        method_results[method] = {}
        for item in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, item)) and item.startswith(f"{method}_") and "_results" in item:
                classifier = item.replace(f"{method}_", "").replace("_results", "")
                
                # Look for accuracy in result files
                accuracy_file = os.path.join(output_dir, item, "accuracy.txt")
                if os.path.exists(accuracy_file):
                    with open(accuracy_file, 'r') as f:
                        accuracy_str = f.read().strip()
                        try:
                            accuracy = float(accuracy_str.replace('%', '')) / 100
                            method_results[method][classifier] = accuracy
                        except ValueError:
                            print(f"Warning: Could not parse accuracy from {accuracy_file}")
    
    # Generate comparison chart
    best_results = {}
    for method in method_results:
        if method_results[method]:
            best_results[method] = max(method_results[method].values())
    
    if best_results:
        comparison_fig = plot_accuracy_comparison(best_results, f"{dataset_name} - Method Comparison")
        comparison_img = fig_to_base64(comparison_fig)
    else:
        comparison_img = None
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Recognition Comparison - {dataset_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .results-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }}
            .result-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                width: 45%;
                min-width: 300px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .method-title {{
                font-weight: bold;
                font-size: 1.2em;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            pre {{
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .timestamp {{
                color: #888;
                font-size: 0.9em;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>Emotion Recognition Method Comparison</h1>
        <h2>Dataset: {dataset_name}</h2>
        
        <div class="chart-container">
            <h3>Accuracy Comparison</h3>
            {f'<img src="data:image/png;base64,{comparison_img}" alt="Method Comparison" />' if comparison_img else '<p>No comparison image available</p>'}
        </div>
        
        <h3>Summary</h3>
        <pre>{summary_content}</pre>
        
        <h3>Method Details</h3>
        <div class="results-container">
    """
    
    # Add section for each method
    for method in methods:
        html_content += f"""
        <div class="result-card">
            <div class="method-title">{method.upper()} Method</div>
            <table>
                <tr>
                    <th>Classifier</th>
                    <th>Accuracy</th>
                </tr>
        """
        
        if method in method_results:
            for classifier, accuracy in method_results[method].items():
                html_content += f"""
                <tr>
                    <td>{classifier}</td>
                    <td>{accuracy:.2%}</td>
                </tr>
                """
        
        html_content += """
            </table>
        </div>
        """
    
    # Close HTML
    html_content += f"""
        </div>
        
        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, f"{dataset_name}_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def generate_confusion_matrix_reports(output_dir, dataset_name='RAVDESS'):
    """
    Generate confusion matrices for all method-classifier combinations.
    
    Args:
        output_dir: Directory with results
        dataset_name: Name of the dataset
    
    Returns:
        Path to the directory containing confusion matrices
    """
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    # Look for prediction files
    for item in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, item)) and "_results" in item:
            parts = item.split('_')
            if len(parts) >= 2:
                method = parts[0]
                classifier = item.replace(f"{method}_", "").replace("_results", "")
                
                # Look for predictions and true labels
                pred_file = os.path.join(output_dir, item, "predictions.npy")
                true_file = os.path.join(output_dir, item, "true_labels.npy")
                classes_file = os.path.join(output_dir, item, "class_names.json")
                
                if os.path.exists(pred_file) and os.path.exists(true_file) and os.path.exists(classes_file):
                    y_pred = np.load(pred_file)
                    y_true = np.load(true_file)
                    
                    with open(classes_file, 'r') as f:
                        class_names = json.load(f)
                    
                    # Plot confusion matrix
                    title = f"{dataset_name} - {method.upper()} method with {classifier} classifier"
                    fig = plot_confusion_matrix(y_true, y_pred, class_names, title)
                    
                    # Save figure
                    output_file = os.path.join(cm_dir, f"{method}_{classifier}_confusion_matrix.png")
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Generate classification report
                    # Make sure we only use class names for classes that are actually present
                    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
                    used_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
                    
                    # Generate the report with the correct labels
                    try:
                        report = classification_report(y_true, y_pred, target_names=used_class_names)
                        report_file = os.path.join(cm_dir, f"{method}_{classifier}_classification_report.txt")
                        with open(report_file, 'w') as f:
                            f.write(report)
                    except ValueError as e:
                        print(f"Warning: Could not generate classification report for {method}_{classifier}: {str(e)}")
                        # Generate a basic report without target_names as fallback
                        report = classification_report(y_true, y_pred)
                        report_file = os.path.join(cm_dir, f"{method}_{classifier}_classification_report.txt")
                        with open(report_file, 'w') as f:
                            f.write(f"Note: Using numeric labels due to mismatch\n\n{report}")
    
    return cm_dir
