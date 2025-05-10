import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

# Training data from command output
training_data = """
Model: hybrid_attention_training
296/296 [==============================] - 31s 105ms/step - loss: 0.1566 - accuracy: 0.8551 - val_loss: 0.6189 - val_accuracy: 0.6468
296/296 [==============================] - 31s 104ms/step - loss: 0.1178 - accuracy: 0.8841 - val_loss: 0.6854 - val_accuracy: 0.6299
296/296 [==============================] - 32s 107ms/step - loss: 0.0982 - accuracy: 0.9002 - val_loss: 0.7089 - val_accuracy: 0.6322
296/296 [==============================] - 31s 106ms/step - loss: 0.0880 - accuracy: 0.9092 - val_loss: 0.8040 - val_accuracy: 0.6119
296/296 [==============================] - 32s 107ms/step - loss: 0.0977 - accuracy: 0.9047 - val_loss: 0.6576 - val_accuracy: 0.6552

Model: branched_cross_attention
296/296 [==============================] - 113s 380ms/step - loss: 0.0337 - accuracy: 0.9904 - val_loss: 0.7277 - val_accuracy: 0.8318
296/296 [==============================] - 102s 343ms/step - loss: 0.0298 - accuracy: 0.9927 - val_loss: 0.7197 - val_accuracy: 0.8330
296/296 [==============================] - 102s 345ms/step - loss: 0.0310 - accuracy: 0.9901 - val_loss: 0.7057 - val_accuracy: 0.8414
296/296 [==============================] - 94s 319ms/step - loss: 0.0276 - accuracy: 0.9927 - val_loss: 0.7061 - val_accuracy: 0.8335
296/296 [==============================] - 86s 291ms/step - loss: 0.0310 - accuracy: 0.9913 - val_loss: 0.7093 - val_accuracy: 0.8380

Model: branched_focal_loss
296/296 [==============================] - 134s 452ms/step - loss: 0.0410 - accuracy: 0.9573 - val_loss: 0.3403 - val_accuracy: 0.8183
296/296 [==============================] - 141s 477ms/step - loss: 0.0396 - accuracy: 0.9587 - val_loss: 0.3511 - val_accuracy: 0.8200
296/296 [==============================] - 142s 478ms/step - loss: 0.0376 - accuracy: 0.9572 - val_loss: 0.3536 - val_accuracy: 0.8217
296/296 [==============================] - 140s 472ms/step - loss: 0.0378 - accuracy: 0.9576 - val_loss: 0.3480 - val_accuracy: 0.8144
296/296 [==============================] - 126s 424ms/step - loss: 0.0362 - accuracy: 0.9610 - val_loss: 0.3568 - val_accuracy: 0.8211

Model: branched_optimizer
296/296 [==============================] - 149s 503ms/step - loss: 0.6926 - accuracy: 0.7485 - val_loss: 0.7153 - val_accuracy: 0.7435
296/296 [==============================] - 130s 438ms/step - loss: 0.6707 - accuracy: 0.7595 - val_loss: 0.7570 - val_accuracy: 0.7177
296/296 [==============================] - 140s 472ms/step - loss: 0.6633 - accuracy: 0.7594 - val_loss: 0.7379 - val_accuracy: 0.7272
296/296 [==============================] - 139s 469ms/step - loss: 0.6201 - accuracy: 0.7720 - val_loss: 0.6990 - val_accuracy: 0.7430
296/296 [==============================] - 139s 468ms/step - loss: 0.6060 - accuracy: 0.7839 - val_loss: 0.6752 - val_accuracy: 0.7610

Model: branched_regularization
296/296 [==============================] - 137s 463ms/step - loss: 0.2226 - accuracy: 0.9589 - val_loss: 0.6754 - val_accuracy: 0.8375
296/296 [==============================] - 130s 438ms/step - loss: 0.1779 - accuracy: 0.9737 - val_loss: 0.6912 - val_accuracy: 0.8346
296/296 [==============================] - 103s 346ms/step - loss: 0.1612 - accuracy: 0.9770 - val_loss: 0.7301 - val_accuracy: 0.8403
296/296 [==============================] - 101s 341ms/step - loss: 0.1574 - accuracy: 0.9779 - val_loss: 0.7772 - val_accuracy: 0.8290
296/296 [==============================] - 102s 343ms/step - loss: 0.1612 - accuracy: 0.9727 - val_loss: 0.8280 - val_accuracy: 0.8166

Model: branched_self_attention
296/296 [==============================] - 96s 326ms/step - loss: 0.0989 - accuracy: 0.9672 - val_loss: 0.6211 - val_accuracy: 0.8335
296/296 [==============================] - 92s 312ms/step - loss: 0.0950 - accuracy: 0.9686 - val_loss: 0.6276 - val_accuracy: 0.8330
296/296 [==============================] - 96s 325ms/step - loss: 0.0893 - accuracy: 0.9732 - val_loss: 0.6346 - val_accuracy: 0.8318
296/296 [==============================] - 90s 303ms/step - loss: 0.0768 - accuracy: 0.9765 - val_loss: 0.6424 - val_accuracy: 0.8268
296/296 [==============================] - 97s 326ms/step - loss: 0.0748 - accuracy: 0.9770 - val_loss: 0.6296 - val_accuracy: 0.8285

Model: branched_sync_aug
592/592 [==============================] - 151s 255ms/step - loss: 0.0125 - accuracy: 0.9968 - val_loss: 0.8418 - val_accuracy: 0.8448
592/592 [==============================] - 151s 255ms/step - loss: 0.0119 - accuracy: 0.9970 - val_loss: 0.8528 - val_accuracy: 0.8431
592/592 [==============================] - 152s 256ms/step - loss: 0.0095 - accuracy: 0.9975 - val_loss: 0.8572 - val_accuracy: 0.8425
592/592 [==============================] - 151s 255ms/step - loss: 0.0107 - accuracy: 0.9974 - val_loss: 0.8461 - val_accuracy: 0.8442
592/592 [==============================] - 151s 255ms/step - loss: 0.0089 - accuracy: 0.9977 - val_loss: 0.8424 - val_accuracy: 0.8453

Model: branched_tcn
296/296 [==============================] - 135s 455ms/step - loss: 0.1491 - accuracy: 0.9486 - val_loss: 0.5747 - val_accuracy: 0.8301
296/296 [==============================] - 144s 486ms/step - loss: 0.1525 - accuracy: 0.9503 - val_loss: 0.5314 - val_accuracy: 0.8420
296/296 [==============================] - 133s 451ms/step - loss: 0.1455 - accuracy: 0.9499 - val_loss: 0.5465 - val_accuracy: 0.8436
296/296 [==============================] - 130s 440ms/step - loss: 0.1335 - accuracy: 0.9555 - val_loss: 0.5391 - val_accuracy: 0.8436
296/296 [==============================] - 143s 483ms/step - loss: 0.1375 - accuracy: 0.9565 - val_loss: 0.5594 - val_accuracy: 0.8408

Model: lstm_attention_no_aug
296/296 [==============================] - 399s 1s/step - loss: 0.0061 - accuracy: 0.9935 - val_loss: 0.6355 - val_accuracy: 0.7531
296/296 [==============================] - 375s 1s/step - loss: 0.0048 - accuracy: 0.9954 - val_loss: 0.6393 - val_accuracy: 0.7576
296/296 [==============================] - 376s 1s/step - loss: 0.0042 - accuracy: 0.9968 - val_loss: 0.6455 - val_accuracy: 0.7553
296/296 [==============================] - 374s 1s/step - loss: 0.0043 - accuracy: 0.9968 - val_loss: 0.6404 - val_accuracy: 0.7570
296/296 [==============================] - 542s 2s/step - loss: 0.0044 - accuracy: 0.9965 - val_loss: 0.6526 - val_accuracy: 0.7553

Model: no_leakage_verification
296/296 [==============================] - 77s 261ms/step - loss: 0.0324 - accuracy: 0.9911 - val_loss: 0.6917 - val_accuracy: 0.8414
296/296 [==============================] - 78s 262ms/step - loss: 0.0293 - accuracy: 0.9931 - val_loss: 0.6998 - val_accuracy: 0.8386
296/296 [==============================] - 78s 263ms/step - loss: 0.0297 - accuracy: 0.9924 - val_loss: 0.6946 - val_accuracy: 0.8408
296/296 [==============================] - 87s 292ms/step - loss: 0.0276 - accuracy: 0.9934 - val_loss: 0.7113 - val_accuracy: 0.8403
296/296 [==============================] - 86s 291ms/step - loss: 0.0302 - accuracy: 0.9910 - val_loss: 0.6997 - val_accuracy: 0.8408

Model: rl_model
296/296 [==============================] - 86s 289ms/step - loss: 1.5266 - accuracy: 0.3885 - val_loss: 1.8733 - val_accuracy: 0.1586
296/296 [==============================] - 85s 289ms/step - loss: 1.5106 - accuracy: 0.3941 - val_loss: 1.8969 - val_accuracy: 0.1676
296/296 [==============================] - 86s 292ms/step - loss: 1.5064 - accuracy: 0.4017 - val_loss: 1.9366 - val_accuracy: 0.1569
296/296 [==============================] - 85s 288ms/step - loss: 1.4934 - accuracy: 0.4085 - val_loss: 2.0366 - val_accuracy: 0.1710
296/296 [==============================] - 87s 293ms/step - loss: 1.7846 - accuracy: 0.2171 - val_loss: 1.8407 - val_accuracy: 0.1704
"""

# Function to parse validation accuracy
def parse_training_data(data):
    model_data = {}
    current_model = None
    
    for line in data.strip().split('\n'):
        if line.startswith('Model:'):
            current_model = line.split('Model:')[1].strip()
            model_data[current_model] = []
        elif 'val_accuracy:' in line and current_model is not None:
            # Extract validation accuracy
            val_acc_match = re.search(r'val_accuracy: ([\d\.]+)', line)
            if val_acc_match:
                val_acc = float(val_acc_match.group(1))
                model_data[current_model].append(val_acc)
    
    return model_data

# Calculate trend and performance metrics
def analyze_trends(model_data):
    trend_results = {}
    
    for model, acc_values in model_data.items():
        if len(acc_values) < 3:
            continue
            
        # Calculate different metrics
        max_val_acc = max(acc_values)
        avg_val_acc = sum(acc_values) / len(acc_values)
        
        # Calculate trend in last 3 epochs
        last_three = acc_values[-3:]
        slope = np.polyfit(range(len(last_three)), last_three, 1)[0]
        
        # Measure variance/stability
        variance = np.var(acc_values)
        
        # Calculate final improvement over starting value
        improvement = acc_values[-1] - acc_values[0]
        
        # Calculate diff between highest and latest accuracy
        diff_from_max = acc_values[-1] - max_val_acc
        
        trend_results[model] = {
            'values': acc_values,
            'max_acc': max_val_acc,
            'avg_acc': avg_val_acc,
            'final_acc': acc_values[-1],
            'recent_trend': slope,
            'variance': variance,
            'improvement': improvement,
            'diff_from_max': diff_from_max
        }
    
    return trend_results

# Create visualizations
def create_visualizations(trend_results, output_dir="analysis_results"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trend visualization
    plt.figure(figsize=(12, 8))
    for model, data in trend_results.items():
        plt.plot(data['values'], marker='o', label=f"{model} (final: {data['final_acc']:.4f})")
    
    plt.title('Validation Accuracy Over Final Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_accuracy_trends.png'), dpi=300)
    
    # Create bar chart of recent trends
    models = list(trend_results.keys())
    recent_trends = [data['recent_trend'] for data in trend_results.values()]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(models, recent_trends, color=['g' if t > 0 else 'r' for t in recent_trends])
    plt.title('Recent Trend in Validation Accuracy (Slope of Last 3 Epochs)')
    plt.xlabel('Model')
    plt.ylabel('Trend (Positive is Improving)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + (0.001 if height > 0 else -0.004),
                 f'{height:.4f}',
                 ha='center', va='bottom' if height > 0 else 'top', 
                 rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recent_trends.png'), dpi=300)
    
    # Create a figure for final accuracy comparison
    plt.figure(figsize=(14, 6))
    final_accs = [(model, data['final_acc']) for model, data in trend_results.items()]
    final_accs.sort(key=lambda x: x[1], reverse=True)
    
    models_sorted = [x[0] for x in final_accs]
    accuracies = [x[1] for x in final_accs]
    
    bars = plt.bar(models_sorted, accuracies)
    plt.title('Final Validation Accuracy by Model')
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + 0.005,
                 f'{height:.4f}',
                 ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_accuracy_comparison.png'), dpi=300)
    
    return

# Generate recommendations based on trend analysis
def generate_recommendations(trend_results):
    recommendations = {
        'continue_training': [],
        'plateau_reached': [],
        'declining_performance': []
    }
    
    for model, data in trend_results.items():
        recent_trend = data['recent_trend']
        final_acc = data['final_acc']
        max_acc = data['max_acc']
        
        # Check if model is still improving
        if recent_trend > 0.001:
            recommendations['continue_training'].append((model, recent_trend, final_acc))
        # Check if model is at plateau
        elif abs(recent_trend) <= 0.001:
            recommendations['plateau_reached'].append((model, recent_trend, final_acc))
        # Check if model is declining
        else:
            recommendations['declining_performance'].append((model, recent_trend, final_acc))
    
    # Sort by trend strength and accuracy
    recommendations['continue_training'].sort(key=lambda x: (x[1], x[2]), reverse=True)
    recommendations['plateau_reached'].sort(key=lambda x: x[2], reverse=True)
    recommendations['declining_performance'].sort(key=lambda x: x[2], reverse=True)
    
    return recommendations

# Create detailed report
def create_report(trend_results, recommendations, output_dir="analysis_results"):
    report_path = os.path.join(output_dir, "training_recommendations.md")
    
    with open(report_path, 'w') as f:
        f.write("# Model Training Analysis and Recommendations\n\n")
        
        f.write("## Models That Should Continue Training\n\n")
        if recommendations['continue_training']:
            f.write("These models show a positive trend in validation accuracy and would likely benefit from additional training epochs:\n\n")
            f.write("| Model | Trend Slope | Final Val Accuracy | Max Val Accuracy | Improvement |\n")
            f.write("|-------|------------|-------------------|-----------------|------------|\n")
            for model, trend, acc in recommendations['continue_training']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} | {data['improvement']:.4f} |\n")
        else:
            f.write("No models are showing significant positive trends in validation accuracy.\n\n")
        
        f.write("\n## Models at Plateau\n\n")
        if recommendations['plateau_reached']:
            f.write("These models show minimal change in recent validation accuracy and may have reached a plateau:\n\n")
            f.write("| Model | Trend Slope | Final Val Accuracy | Max Val Accuracy |\n")
            f.write("|-------|------------|-------------------|-----------------|------------|\n")
            for model, trend, acc in recommendations['plateau_reached']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} |\n")
        else:
            f.write("No models are showing plateau in validation accuracy.\n\n")
        
        f.write("\n## Models with Declining Performance\n\n")
        if recommendations['declining_performance']:
            f.write("These models show a negative trend in validation accuracy, suggesting potential overfitting:\n\n")
            f.write("| Model | Trend Slope | Final Val Accuracy | Max Val Accuracy | Diff from Max |\n")
            f.write("|-------|------------|-------------------|-----------------|------------|\n")
            for model, trend, acc in recommendations['declining_performance']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} | {data['diff_from_max']:.4f} |\n")
        else:
            f.write("No models are showing declining performance in validation accuracy.\n\n")
        
        f.write("\n## Individual Model Analysis\n\n")
        for model, data in trend_results.items():
            f.write(f"### {model}\n\n")
            f.write(f"- **Final Validation Accuracy**: {data['final_acc']:.4f}\n")
            f.write(f"- **Maximum Validation Accuracy**: {data['max_acc']:.4f}\n")
            f.write(f"- **Recent Trend (Slope)**: {data['recent_trend']:.4f}\n")
            f.write(f"- **Stability (Variance)**: {data['variance']:.6f}\n")
            f.write(f"- **Overall Improvement**: {data['improvement']:.4f}\n")
            f.write(f"- **All Validation Accuracies**: {', '.join(f'{v:.4f}' for v in data['values'])}\n\n")
    
    return report_path

# Main execution
def main():
    # Parse data
    model_data = parse_training_data(training_data)
    
    # Analyze trends
    trend_results = analyze_trends(model_data)
    
    # Generate recommendations
    recommendations = generate_recommendations(trend_results)
    
    # Create visualizations
    create_visualizations(trend_results)
    
    # Create report
    report_path = create_report(trend_results, recommendations)
    
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
