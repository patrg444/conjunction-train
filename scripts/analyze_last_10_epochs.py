import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
from collections import defaultdict

def fetch_training_data(ssh_key, ec2_instance, num_epochs=10):
    """
    Fetch the last 10 epochs of training data from EC2 instance logs
    """
    print(f"Fetching the last {num_epochs} epochs for each model from EC2 instance...")
    
    # Command to fetch epochs from the logs
    cmd = [
        "ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", ec2_instance,
        f"for log in ~/emotion_training/*.log; do model=$(basename $log .log | sed 's/training_//g'); "
        f"echo -e \"\\nModel: $model\"; grep -a 'val_accuracy:' $log | tail -{num_epochs}; done"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error fetching training data: {e}")
        print(f"Error output: {e.stderr}")
        return None

def parse_training_data(data):
    """Parse validation accuracy from training logs"""
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

def analyze_trends(model_data):
    """Calculate trend metrics for each model focusing on the last 10 epochs"""
    trend_results = {}
    
    for model, acc_values in model_data.items():
        if len(acc_values) < 3:
            continue
            
        # Calculate metrics
        max_val_acc = max(acc_values)
        avg_val_acc = sum(acc_values) / len(acc_values)
        final_acc = acc_values[-1]
        
        # Calculate trend over all epochs provided (should be 10)
        epochs = range(len(acc_values))
        overall_trend = np.polyfit(epochs, acc_values, 1)[0]
        
        # Calculate trend in the last 3 epochs for recent direction
        if len(acc_values) >= 3:
            last_three_trend = np.polyfit(range(3), acc_values[-3:], 1)[0]
        else:
            last_three_trend = None
            
        # Calculate improvement over the full 10 epochs
        improvement_overall = acc_values[-1] - acc_values[0]
        
        # Calculate improvement in second half 
        halfway = len(acc_values) // 2
        improvement_second_half = acc_values[-1] - acc_values[halfway]
        
        # Measure stability
        variance = np.var(acc_values)
        
        # Calculate difference from max
        diff_from_max = final_acc - max_val_acc
        
        trend_results[model] = {
            'values': acc_values,
            'max_acc': max_val_acc,
            'avg_acc': avg_val_acc,
            'final_acc': final_acc,
            'overall_trend': overall_trend,
            'last_three_trend': last_three_trend,
            'improvement_overall': improvement_overall,
            'improvement_second_half': improvement_second_half,
            'variance': variance,
            'diff_from_max': diff_from_max
        }
    
    return trend_results

def generate_recommendations(trend_results):
    """Generate recommendations based on the last 10 epochs trend"""
    recommendations = {
        'continue_training': [],
        'consider_continue': [], 
        'plateau': [],
        'declining': []
    }
    
    for model, data in trend_results.items():
        overall_trend = data['overall_trend']
        final_acc = data['final_acc']
        improvement = data['improvement_overall']
        
        # Categorize models based on 10-epoch trend
        if overall_trend > 0.003:
            # Strong positive trend
            recommendations['continue_training'].append((model, overall_trend, final_acc, improvement))
        elif overall_trend > 0.0005:
            # Slight positive trend
            recommendations['consider_continue'].append((model, overall_trend, final_acc, improvement))
        elif overall_trend > -0.0005:
            # Plateau
            recommendations['plateau'].append((model, overall_trend, final_acc))
        else:
            # Declining
            recommendations['declining'].append((model, overall_trend, final_acc, data['diff_from_max']))
    
    # Sort by trend strength and accuracy
    for category in recommendations:
        if category in ['continue_training', 'consider_continue']:
            recommendations[category].sort(key=lambda x: (x[1], x[2]), reverse=True)
        elif category == 'plateau':
            recommendations[category].sort(key=lambda x: x[2], reverse=True)
        else:
            recommendations[category].sort(key=lambda x: (x[1], x[2]))
    
    return recommendations

def create_visualizations(trend_results, output_dir="analysis_results"):
    """Create visualizations focusing on 10-epoch trends"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trend visualization for all models
    plt.figure(figsize=(14, 8))
    for model, data in trend_results.items():
        plt.plot(data['values'], marker='o', label=f"{model} (final: {data['final_acc']:.4f})")
    
    plt.title('Validation Accuracy Over Last 10 Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'last_10_epochs_trends.png'), dpi=300)
    
    # Create bar chart of trends
    plt.figure(figsize=(14, 6))
    models = list(trend_results.keys())
    trends = [data['overall_trend'] for data in trend_results.values()]
    colors = ['g' if t > 0 else 'r' for t in trends]
    
    bars = plt.bar(models, trends, color=colors)
    plt.title('10-Epoch Trend in Validation Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Trend (Slope)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + (0.001 if height > 0 else -0.004),
                f'{height:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', 
                rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_epoch_trend_comparison.png'), dpi=300)
    
    # Create improvement visualization
    plt.figure(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.35
    
    improvements = [data['improvement_overall'] for data in trend_results.values()]
    half_improvements = [data['improvement_second_half'] for data in trend_results.values()]
    
    plt.bar(x - width/2, improvements, width, label='10-Epoch Improvement')
    plt.bar(x + width/2, half_improvements, width, label='5-Epoch Improvement')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Model')
    plt.ylabel('Improvement (Val Accuracy)')
    plt.title('Improvement in Validation Accuracy')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_epoch_improvements.png'), dpi=300)
    
    return

def create_report(trend_results, recommendations, output_dir="analysis_results"):
    """Create report based on 10-epoch analysis"""
    report_path = os.path.join(output_dir, "10_epoch_training_recommendations.md")
    
    with open(report_path, 'w') as f:
        f.write("# Model Training Analysis - Last 10 Epochs\n\n")
        
        f.write("## Models That Should Continue Training\n\n")
        if recommendations['continue_training']:
            f.write("These models show a strong positive trend over the last 10 epochs:\n\n")
            f.write("| Model | 10-Epoch Trend | Final Val Accuracy | Max Val Accuracy | Improvement |\n")
            f.write("|-------|---------------|-------------------|-----------------|------------|\n")
            for model, trend, acc, improvement in recommendations['continue_training']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} | {improvement:.4f} |\n")
        else:
            f.write("No models are showing strong positive trends over the last 10 epochs.\n\n")
        
        f.write("\n## Models to Consider Continuing Training\n\n")
        if recommendations['consider_continue']:
            f.write("These models show a slight positive trend over the last 10 epochs:\n\n")
            f.write("| Model | 10-Epoch Trend | Final Val Accuracy | Max Val Accuracy | Improvement |\n")
            f.write("|-------|---------------|-------------------|-----------------|------------|\n")
            for model, trend, acc, improvement in recommendations['consider_continue']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} | {improvement:.4f} |\n")
        else:
            f.write("No models are showing moderate improvement over the last 10 epochs.\n\n")
        
        f.write("\n## Models at Plateau\n\n")
        if recommendations['plateau']:
            f.write("These models show minimal change in validation accuracy over the last 10 epochs:\n\n")
            f.write("| Model | 10-Epoch Trend | Final Val Accuracy | Max Val Accuracy |\n")
            f.write("|-------|---------------|-------------------|------------------|\n")
            for model, trend, acc in recommendations['plateau']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} |\n")
        else:
            f.write("No models are showing plateau in validation accuracy.\n\n")
        
        f.write("\n## Models with Declining Performance\n\n")
        if recommendations['declining']:
            f.write("These models show a negative trend in validation accuracy over the last 10 epochs:\n\n")
            f.write("| Model | 10-Epoch Trend | Final Val Accuracy | Max Val Accuracy | Diff from Max |\n")
            f.write("|-------|---------------|-------------------|-----------------|---------------|\n")
            for model, trend, acc, diff in recommendations['declining']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.4f} | {acc:.4f} | {data['max_acc']:.4f} | {diff:.4f} |\n")
        else:
            f.write("No models are showing declining performance in validation accuracy.\n\n")
        
        f.write("\n## Individual Model Analysis\n\n")
        for model, data in trend_results.items():
            f.write(f"### {model}\n\n")
            f.write(f"- **Final Validation Accuracy**: {data['final_acc']:.4f}\n")
            f.write(f"- **Maximum Validation Accuracy**: {data['max_acc']:.4f}\n")
            f.write(f"- **10-Epoch Trend**: {data['overall_trend']:.4f}\n")
            f.write(f"- **Recent Trend (Last 3 Epochs)**: {data['last_three_trend']:.4f}\n")
            f.write(f"- **10-Epoch Improvement**: {data['improvement_overall']:.4f}\n")
            f.write(f"- **Improvement in Last 5 Epochs**: {data['improvement_second_half']:.4f}\n")
            f.write(f"- **Stability (Variance)**: {data['variance']:.6f}\n")
            f.write(f"- **All Validation Accuracies**: {', '.join(f'{v:.4f}' for v in data['values'])}\n\n")
    
    return report_path

def create_continuation_script(recommendations, script_path):
    """Create a bash script to continue training for recommended models based on 10-epoch analysis"""
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script to continue training for models showing positive trends\n")
        f.write("# Based on last 10 epochs trend analysis\n\n")
        
        f.write('echo "=========================================================="\n')
        f.write('echo "10-Epoch Training Continuation Script"\n')
        f.write('echo "=========================================================="\n')
        f.write('echo\n\n')
        
        # Define variables
        f.write("# Settings for continued training\n")
        f.write("ADDITIONAL_EPOCHS=30\n")
        f.write('AWS_INSTANCE="ec2-user@3.235.76.0"\n')
        f.write('SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"\n\n')
        
        # Define models to train
        f.write("# Models with strong positive trends\n")
        f.write("CONTINUE_MODELS=(\n")
        for model, _, _, _ in recommendations['continue_training']:
            f.write(f'  "{model}"\n')
        f.write(")\n\n")
        
        f.write("# Models with slight positive trends\n")
        f.write("CONSIDER_MODELS=(\n")
        for model, _, _, _ in recommendations['consider_continue']:
            f.write(f'  "{model}"\n')
        f.write(")\n\n")
        
        # Create function to continue training
        f.write("# Function to continue training for a specific model\n")
        f.write("continue_training() {\n")
        f.write("  local model=$1\n")
        f.write("  local epochs=$2\n")
        f.write('  echo "=========================================================="\n')
        f.write('  echo "Continuing training for model: $model ($epochs epochs)"\n')
        f.write('  echo "=========================================================="\n')
        f.write("  \n")
        f.write("  # Construct the SSH command to continue training\n")
        f.write('  ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE "cd ~/emotion_training && \\\n')
        f.write('    echo \'Continuing training for $model for $epochs more epochs\' && \\\n')
        f.write('    python continue_training.py --model $model --epochs $epochs \\\n')
        f.write('    >> training_${model}.log 2>&1"\n')
        f.write("  \n")
        f.write('  echo "Requested continuation of training for $model"\n')
        f.write('  echo "Training logs will be appended to ~/emotion_training/training_${model}.log"\n')
        f.write('  echo\n')
        f.write("}\n\n")
        
        # Execute training continuation
        f.write('echo "Models that showed strong positive trends over the last 10 epochs:"\n')
        f.write('for model in "${CONTINUE_MODELS[@]}"; do\n')
        f.write('  echo "- $model"\n')
        f.write('done\n')
        f.write('echo\n\n')
        
        f.write('echo "Models that showed slight positive trends over the last 10 epochs:"\n')
        f.write('for model in "${CONSIDER_MODELS[@]}"; do\n')
        f.write('  echo "- $model"\n')
        f.write('done\n')
        f.write('echo\n\n')
        
        f.write('# Execute training continuation for recommended models\n')
        f.write('for model in "${CONTINUE_MODELS[@]}"; do\n')
        f.write('  continue_training "$model" $ADDITIONAL_EPOCHS\n')
        f.write('done\n\n')
        
        f.write('# Ask if user wants to train the "consider" models too\n')
        f.write('read -p "Would you like to continue training for the models with slight positive trends? (y/n): " train_consider\n')
        f.write('if [[ "$train_consider" == "y" ]]; then\n')
        f.write('  for model in "${CONSIDER_MODELS[@]}"; do\n')
        f.write('    continue_training "$model" $ADDITIONAL_EPOCHS\n')
        f.write('  done\n')
        f.write('fi\n\n')
        
        f.write('echo "=========================================================="\n')
        f.write('echo "Monitoring training progress"\n')
        f.write('echo "=========================================================="\n')
        f.write('echo "To monitor training progress, use:"\n')
        f.write('echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \\"tail -f ~/emotion_training/training_MODEL_NAME.log\\""\n')
        f.write('echo\n')
        f.write('echo "To check validation accuracy after training:"\n')
        f.write('echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \\"grep -a \'val_accuracy:\' ~/emotion_training/training_MODEL_NAME.log | tail -10\\""\n')
        f.write('echo\n')
        f.write('echo "Script completed."\n')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Created 10-epoch training continuation script: {script_path}")

def main():
    # Configuration
    ssh_key = "aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
    ec2_instance = "ec2-user@3.235.76.0"
    
    # Fetch training data (last 10 epochs)
    training_data = fetch_training_data(ssh_key, ec2_instance, 10)
    
    if not training_data:
        print("Failed to fetch training data. Using example data for analysis...")
        # Use existing data as fallback
        with open("analyze_training_trends.py", "r") as f:
            content = f.read()
            training_data_match = re.search(r'training_data = """(.*?)"""', content, re.DOTALL)
            if training_data_match:
                training_data = training_data_match.group(1)
            else:
                print("Could not find training data in analyze_training_trends.py")
                return
    
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
    
    print(f"10-epoch analysis complete. Report saved to {report_path}")
    
    # Generate a script for training continuation
    create_continuation_script(recommendations, "last_10_epochs_continuation.sh")

if __name__ == "__main__":
    main()
