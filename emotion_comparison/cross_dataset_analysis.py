#!/usr/bin/env python3
"""
Cross-dataset analysis script for emotion recognition comparison framework.
Compares results between different datasets (e.g., RAVDESS vs CREMA-D).
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.visualization import fig_to_base64

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-dataset Analysis')
    parser.add_argument('--ravdess_dir', type=str, required=True, help='RAVDESS results directory')
    parser.add_argument('--cremad_dir', type=str, required=True, help='CREMA-D results directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for combined results')
    
    return parser.parse_args()

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_results(results_dir):
    """Load results from a dataset directory."""
    results = {}
    
    # Check if the directory exists
    if not os.path.isdir(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return results
    
    # Load summary data
    summary_path = os.path.join(results_dir, 'comparison_summary.txt')
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found: {summary_path}")
        return results
    
    # Extract method directories
    methods = []
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and "_features" in item:
            method = item.replace("_features", "")
            methods.append(method)
    
    # Load results for each method and classifier
    for method in methods:
        results[method] = {}
        for item in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, item)) and item.startswith(f"{method}_") and "_results" in item:
                classifier = item.replace(f"{method}_", "").replace("_results", "")
                
                # Load accuracy
                accuracy_file = os.path.join(results_dir, item, "accuracy.txt")
                if os.path.exists(accuracy_file):
                    with open(accuracy_file, 'r') as f:
                        accuracy_str = f.read().strip()
                        try:
                            accuracy = float(accuracy_str.replace('%', '')) / 100
                            results[method][classifier] = accuracy
                        except ValueError:
                            print(f"Warning: Could not parse accuracy from {accuracy_file}")
    
    return results

def find_best_methods(results):
    """Find the best method and classifier for each dataset."""
    best_methods = {}
    
    for dataset, dataset_results in results.items():
        best_accuracy = 0
        best_method = None
        best_classifier = None
        
        for method, method_results in dataset_results.items():
            for classifier, accuracy in method_results.items():
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_method = method
                    best_classifier = classifier
        
        best_methods[dataset] = {
            'method': best_method,
            'classifier': best_classifier,
            'accuracy': best_accuracy
        }
    
    return best_methods

def create_cross_dataset_tables(ravdess_results, cremad_results):
    """Create tables comparing results across datasets."""
    # Find common methods and classifiers
    ravdess_methods = set(ravdess_results.keys())
    cremad_methods = set(cremad_results.keys())
    common_methods = ravdess_methods.intersection(cremad_methods)
    
    tables = {}
    
    # For each common method, create a table of classifiers
    for method in common_methods:
        ravdess_classifiers = set(ravdess_results[method].keys())
        cremad_classifiers = set(cremad_results[method].keys())
        common_classifiers = ravdess_classifiers.intersection(cremad_classifiers)
        
        if common_classifiers:
            table_data = []
            for classifier in common_classifiers:
                ravdess_acc = ravdess_results[method].get(classifier, 0)
                cremad_acc = cremad_results[method].get(classifier, 0)
                diff = cremad_acc - ravdess_acc
                
                table_data.append({
                    'Classifier': classifier,
                    'RAVDESS Accuracy': f"{ravdess_acc:.2%}",
                    'CREMA-D Accuracy': f"{cremad_acc:.2%}",
                    'Difference': f"{diff:.2%}",
                    'Numeric Diff': diff
                })
            
            # Sort by difference
            table_data = sorted(table_data, key=lambda x: abs(x['Numeric Diff']), reverse=True)
            tables[method] = table_data
    
    return tables

def plot_cross_dataset_comparison(ravdess_results, cremad_results, title="Cross-Dataset Comparison"):
    """Create a bar chart comparing results across datasets."""
    # Find common methods and classifiers
    ravdess_methods = set(ravdess_results.keys())
    cremad_methods = set(cremad_results.keys())
    common_methods = ravdess_methods.intersection(cremad_methods)
    
    # Prepare data for plotting
    methods = []
    ravdess_accs = []
    cremad_accs = []
    
    for method in common_methods:
        ravdess_best = max(ravdess_results[method].values()) if ravdess_results[method] else 0
        cremad_best = max(cremad_results[method].values()) if cremad_results[method] else 0
        
        methods.append(method.upper())
        ravdess_accs.append(ravdess_best)
        cremad_accs.append(cremad_best)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ravdess_accs, width, label='RAVDESS')
    rects2 = ax.bar(x + width/2, cremad_accs, width, label='CREMA-D')
    
    # Add labels and values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add title and labels
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_cross_dataset_report(ravdess_results, cremad_results, tables, output_dir):
    """Create HTML report comparing results across datasets."""
    # Create comparison chart
    fig = plot_cross_dataset_comparison(ravdess_results, cremad_results)
    comparison_img = fig_to_base64(fig)
    
    # Generate HTML for tables
    tables_html = ""
    for method, table_data in tables.items():
        tables_html += f"""
        <div class="result-card">
            <div class="method-title">{method.upper()} Method</div>
            <table>
                <tr>
                    <th>Classifier</th>
                    <th>RAVDESS Accuracy</th>
                    <th>CREMA-D Accuracy</th>
                    <th>Difference</th>
                </tr>
        """
        
        for row in table_data:
            # Determine color based on difference
            diff_class = ""
            if row['Numeric Diff'] > 0.05:
                diff_class = "better-cremad"
            elif row['Numeric Diff'] < -0.05:
                diff_class = "better-ravdess"
            
            tables_html += f"""
                <tr>
                    <td>{row['Classifier']}</td>
                    <td>{row['RAVDESS Accuracy']}</td>
                    <td>{row['CREMA-D Accuracy']}</td>
                    <td class="{diff_class}">{row['Difference']}</td>
                </tr>
            """
        
        tables_html += """
            </table>
        </div>
        """
    
    # Find best methods
    best_methods = find_best_methods({'RAVDESS': ravdess_results, 'CREMAD': cremad_results})
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cross-Dataset Comparison: RAVDESS vs CREMA-D</title>
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
            .summary-card {{
                background-color: #f9f9f9;
                border-left: 4px solid #2c3e50;
                padding: 15px;
                margin: 20px 0;
            }}
            .better-cremad {{
                color: green;
            }}
            .better-ravdess {{
                color: blue;
            }}
            .timestamp {{
                color: #888;
                font-size: 0.9em;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>Cross-Dataset Comparison: RAVDESS vs CREMA-D</h1>
        
        <div class="chart-container">
            <h3>Accuracy Comparison Across Datasets</h3>
            <img src="data:image/png;base64,{comparison_img}" alt="Cross-Dataset Comparison" />
        </div>
        
        <h3>Best Methods by Dataset</h3>
        <div class="summary-card">
            <p><strong>RAVDESS Best Method:</strong> {best_methods['RAVDESS']['method']} with {best_methods['RAVDESS']['classifier']} classifier ({best_methods['RAVDESS']['accuracy']:.2%} accuracy)</p>
            <p><strong>CREMA-D Best Method:</strong> {best_methods['CREMAD']['method']} with {best_methods['CREMAD']['classifier']} classifier ({best_methods['CREMAD']['accuracy']:.2%} accuracy)</p>
        </div>
        
        <h3>Cross-Dataset Performance</h3>
        <div class="results-container">
            {tables_html}
        </div>
        
        <h3>Key Findings</h3>
        <div class="summary-card">
            <p>Performance differences between datasets may indicate:</p>
            <ul>
                <li>Dataset-specific biases or characteristics</li>
                <li>Differences in emotional expression clarity</li>
                <li>Feature extractor sensitivity to recording conditions</li>
                <li>Generalization capabilities of different approaches</li>
            </ul>
            <p>Methods with consistent performance across datasets are more likely to generalize well to new data.</p>
        </div>
        
        <div class="timestamp">
            Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, "cross_dataset_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def main():
    args = parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Load results from both datasets
    print("Loading RAVDESS results...")
    ravdess_results = load_results(args.ravdess_dir)
    
    print("Loading CREMA-D results...")
    cremad_results = load_results(args.cremad_dir)
    
    # Create cross-dataset tables
    print("Creating cross-dataset comparison tables...")
    tables = create_cross_dataset_tables(ravdess_results, cremad_results)
    
    # Generate HTML report
    print("Generating cross-dataset report...")
    report_path = create_cross_dataset_report(ravdess_results, cremad_results, tables, args.output_dir)
    
    print(f"Cross-dataset analysis complete!")
    print(f"Report saved to: {report_path}")
    
    # Also create PNG file for the comparison chart
    fig = plot_cross_dataset_comparison(ravdess_results, cremad_results)
    chart_path = os.path.join(args.output_dir, "cross_dataset_comparison.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
