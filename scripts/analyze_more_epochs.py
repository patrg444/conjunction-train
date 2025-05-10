import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
from collections import defaultdict

def fetch_extended_training_data(ssh_key, ec2_instance, num_epochs=20):
    """
    Fetch more epochs of training data from EC2 instance logs
    """
    print(f"Fetching the last {num_epochs} epochs for each model from EC2 instance...")
    
    # Command to fetch more epochs from the logs
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
    """Calculate trend metrics for each model"""
    trend_results = {}
    
    for model, acc_values in model_data.items():
        if len(acc_values) < 3:
            continue
            
        # Calculate metrics
        max_val_acc = max(acc_values)
        avg_val_acc = sum(acc_values) / len(acc_values)
        
        # Calculate different trend periods
        if len(acc_values) >= 10:
            last_ten = acc_values[-10:]
            trend_last_ten = np.polyfit(range(len(last_ten)), last_ten, 1)[0]
        else:
            trend_last_ten = None
            
        if len(acc_values) >= 5:
            last_five = acc_values[-5:]
            trend_last_five = np.polyfit(range(len(last_five)), last_five, 1)[0]
        else:
            trend_last_five = None
            
        last_three = acc_values[-3:]
        trend_last_three = np.polyfit(range(len(last_three)), last_three, 1)[0]
        
        # Calculate overall trend
        overall_trend = np.polyfit(range(len(acc_values)), acc_values, 1)[0]
        
        # Measure variance/stability
        variance = np.var(acc_values)
        
        # Calculate improvements
        improvement_overall = acc_values[-1] - acc_values[0]
        improvement_half = acc_values[-1] - acc_values[len(acc_values)//2] if len(acc_values) > 1 else 0
        
        # Calculate diff between highest and latest accuracy
        diff_from_max = acc_values[-1] - max_val_acc
        
        trend_results[model] = {
            'values': acc_values,
            'max_acc': max_val_acc,
            'avg_acc': avg_val_acc,
            'final_acc': acc_values[-1],
            'trend_last_three': trend_last_three,
            'trend_last_five': trend_last_five,
            'trend_last_ten': trend_last_ten,
            'trend_overall': overall_trend,
            'variance': variance,
            'improvement_overall': improvement_overall,
            'improvement_half': improvement_half,
            'diff_from_max': diff_from_max
        }
    
    return trend_results

def create_visualizations(trend_results, output_dir="analysis_results"):
    """Create visualization plots for the trends"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full training history visualization
    plt.figure(figsize=(14, 10))
    for model, data in trend_results.items():
        plt.plot(data['values'], marker='o', label=f"{model} (final: {data['final_acc']:.4f})")
    
    plt.title('Validation Accuracy Over Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_validation_history.png'), dpi=300)
    
    # Create trend visualization for different time periods
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot for overall trend
    models = list(trend_results.keys())
    overall_trends = [data['trend_overall'] for data in trend_results.values()]
    colors = ['g' if t > 0 else 'r' for t in overall_trends]
    
    axs[0, 0].bar(models, overall_trends, color=colors)
    axs[0, 0].set_title('Overall Training Trend')
    axs[0, 0].set_xlabel('Model')
    axs[0, 0].set_ylabel('Slope (Trend)')
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot for last 10 epochs
    last_ten_trends = []
    valid_models = []
    for model, data in trend_results.items():
        if data['trend_last_ten'] is not None:
            last_ten_trends.append(data['trend_last_ten'])
            valid_models.append(model)
    
    if last_ten_trends:
        colors = ['g' if t > 0 else 'r' for t in last_ten_trends]
        axs[0, 1].bar(valid_models, last_ten_trends, color=colors)
        axs[0, 1].set_title('Trend in Last 10 Epochs')
        axs[0, 1].set_xlabel('Model')
        axs[0, 1].set_ylabel('Slope (Trend)')
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot for last 5 epochs
    last_five_trends = []
    valid_models = []
    for model, data in trend_results.items():
        if data['trend_last_five'] is not None:
            last_five_trends.append(data['trend_last_five'])
            valid_models.append(model)
    
    if last_five_trends:
        colors = ['g' if t > 0 else 'r' for t in last_five_trends]
        axs[1, 0].bar(valid_models, last_five_trends, color=colors)
        axs[1, 0].set_title('Trend in Last 5 Epochs')
        axs[1, 0].set_xlabel('Model')
        axs[1, 0].set_ylabel('Slope (Trend)')
        axs[1, 0].tick_params(axis='x', rotation=45)
        axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot for last 3 epochs
    last_three_trends = [data['trend_last_three'] for data in trend_results.values()]
    colors = ['g' if t > 0 else 'r' for t in last_three_trends]
    
    axs[1, 1].bar(models, last_three_trends, color=colors)
    axs[1, 1].set_title('Trend in Last 3 Epochs')
    axs[1, 1].set_xlabel('Model')
    axs[1, 1].set_ylabel('Slope (Trend)')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_comparison.png'), dpi=300)
    
    # Create visualization for improvement metrics
    plt.figure(figsize=(14, 10))
    models = list(trend_results.keys())
    x = np.arange(len(models))
    width = 0.35
    
    improvement_overall = [data['improvement_overall'] for data in trend_results.values()]
    improvement_half = [data['improvement_half'] for data in trend_results.values()]
    
    plt.bar(x - width/2, improvement_overall, width, label='Overall Improvement')
    plt.bar(x + width/2, improvement_half, width, label='Second-Half Improvement')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Model')
    plt.ylabel('Improvement (Validation Accuracy)')
    plt.title('Training Improvement Metrics')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_metrics.png'), dpi=300)
    
    return

def generate_recommendations(trend_results):
    """Generate training continuation recommendations based on different trend metrics"""
    recommendations = {
        'definite_continue': [],
        'consider_continue': [],
        'plateau': [],
        'declining': []
    }
    
    for model, data in trend_results.items():
        # Calculate a combined trend score that considers different trends with weights
        weights = {
            'last_three': 0.3,
            'last_five': 0.3 if data['trend_last_five'] is not None else 0,
            'last_ten': 0.2 if data['trend_last_ten'] is not None else 0,
            'overall': 0.2
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Calculate combined score
        combined_trend = 0
        if weights['last_three'] > 0:
            combined_trend += weights['last_three'] * data['trend_last_three']
        if weights['last_five'] > 0:
            combined_trend += weights['last_five'] * data['trend_last_five']
        if weights['last_ten'] > 0:
            combined_trend += weights['last_ten'] * data['trend_last_ten']
        if weights['overall'] > 0:
            combined_trend += weights['overall'] * data['trend_overall']
        
        # The final threshold needs to be tuned based on your specific models,
        # but these are reasonable starting values
        if combined_trend > 0.005:
            # Strongly positive trend
            recommendations['definite_continue'].append((model, combined_trend, data['final_acc']))
        elif combined_trend > 0.0005:
            # Slightly positive trend
            recommendations['consider_continue'].append((model, combined_trend, data['final_acc']))
        elif combined_trend > -0.0005:
            # Plateau (neither significantly improving nor declining)
            recommendations['plateau'].append((model, combined_trend, data['final_acc']))
        else:
            # Declining performance
            recommendations['declining'].append((model, combined_trend, data['final_acc']))
    
    # Sort by trend strength and accuracy
    for category in recommendations:
        recommendations[category].sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    return recommendations

def create_report(trend_results, recommendations, output_dir="analysis_results"):
    """Create a detailed markdown report with recommendations"""
    report_path = os.path.join(output_dir, "extended_training_recommendations.md")
    
    with open(report_path, 'w') as f:
        f.write("# Extended Model Training Analysis and Recommendations\n\n")
        
        f.write("## Models That Should Definitely Continue Training\n\n")
        if recommendations['definite_continue']:
            f.write("These models show a strong positive trend in validation accuracy across multiple time frames:\n\n")
            f.write("| Model | Combined Trend | Final Val Accuracy | Max Val Accuracy | Overall Improvement | Recent Improvement |\n")
            f.write("|-------|---------------|-------------------|-----------------|---------------------|--------------------|\n")
            for model, trend, acc in recommendations['definite_continue']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.6f} | {acc:.4f} | {data['max_acc']:.4f} | " 
                        f"{data['improvement_overall']:.4f} | {data['improvement_half']:.4f} |\n")
        else:
            f.write("No models are showing strong positive trends in validation accuracy.\n\n")
        
        f.write("\n## Models to Consider Continuing Training\n\n")
        if recommendations['consider_continue']:
            f.write("These models show a slight positive trend in validation accuracy:\n\n")
            f.write("| Model | Combined Trend | Final Val Accuracy | Max Val Accuracy | Overall Improvement | Recent Improvement |\n")
            f.write("|-------|---------------|-------------------|-----------------|---------------------|--------------------|\n")
            for model, trend, acc in recommendations['consider_continue']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.6f} | {acc:.4f} | {data['max_acc']:.4f} | " 
                        f"{data['improvement_overall']:.4f} | {data['improvement_half']:.4f} |\n")
        else:
            f.write("No models are showing slight positive trends in validation accuracy.\n\n")
        
        f.write("\n## Models at Plateau\n\n")
        if recommendations['plateau']:
            f.write("These models show minimal change in recent validation accuracy and may have reached a plateau:\n\n")
            f.write("| Model | Combined Trend | Final Val Accuracy | Max Val Accuracy | Overall Improvement |\n")
            f.write("|-------|---------------|-------------------|-----------------|---------------------|\n")
            for model, trend, acc in recommendations['plateau']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.6f} | {acc:.4f} | {data['max_acc']:.4f} | {data['improvement_overall']:.4f} |\n")
        else:
            f.write("No models are showing plateau in validation accuracy.\n\n")
        
        f.write("\n## Models with Declining Performance\n\n")
        if recommendations['declining']:
            f.write("These models show a negative trend in validation accuracy, suggesting potential overfitting:\n\n")
            f.write("| Model | Combined Trend | Final Val Accuracy | Max Val Accuracy | Diff from Max |\n")
            f.write("|-------|---------------|-------------------|-----------------|---------------|\n")
            for model, trend, acc in recommendations['declining']:
                data = trend_results[model]
                f.write(f"| {model} | {trend:.6f} | {acc:.4f} | {data['max_acc']:.4f} | {data['diff_from_max']:.4f} |\n")
        else:
            f.write("No models are showing declining performance in validation accuracy.\n\n")
        
        f.write("\n## Detailed Trend Analysis\n\n")
        f.write("Trend analysis was performed using multiple time frames to get a comprehensive picture of model training dynamics:\n\n")
        f.write("- **Last 3 epochs**: Short-term trend showing very recent direction\n")
        f.write("- **Last 5 epochs**: Medium-term trend\n")
        f.write("- **Last 10 epochs**: Longer-term trend (when available)\n")
        f.write("- **Overall trend**: Trend calculated across all available training data\n\n")
        
        f.write("The combined trend score weights these different time frames to prioritize recent performance while still considering the overall training trajectory.\n\n")
        
        f.write("\n## Individual Model Analysis\n\n")
        for model, data in trend_results.items():
            f.write(f"### {model}\n\n")
            f.write(f"- **Final Validation Accuracy**: {data['final_acc']:.4f}\n")
            f.write(f"- **Maximum Validation Accuracy**: {data['max_acc']:.4f}\n")
            f.write(f"- **Trend (Last 3 Epochs)**: {data['trend_last_three']:.6f}\n")
            if data['trend_last_five'] is not None:
                f.write(f"- **Trend (Last 5 Epochs)**: {data['trend_last_five']:.6f}\n")
            if data['trend_last_ten'] is not None:
                f.write(f"- **Trend (Last 10 Epochs)**: {data['trend_last_ten']:.6f}\n")
            f.write(f"- **Trend (Overall)**: {data['trend_overall']:.6f}\n")
            f.write(f"- **Stability (Variance)**: {data['variance']:.6f}\n")
            f.write(f"- **Overall Improvement**: {data['improvement_overall']:.4f}\n")
            f.write(f"- **Improvement in Second Half**: {data['improvement_half']:.4f}\n")
            f.write(f"- **Number of Epochs Analyzed**: {len(data['values'])}\n")
            f.write(f"- **All Validation Accuracies**: {', '.join(f'{v:.4f}' for v in data['values'])}\n\n")
    
    return report_path

def main():
    # Configuration
    ssh_key = "aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
    ec2_instance = "ec2-user@3.235.76.0"
    num_epochs = 20  # Fetch 20 epochs instead of just 5
    
    # Fetch extended training data
    training_data = fetch_extended_training_data(ssh_key, ec2_instance, num_epochs)
    
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
    
    print(f"Extended analysis complete. Report saved to {report_path}")
    
    # Generate a script for training continuation
    create_continuation_script(recommendations, "extended_training_continuation.sh")

def create_continuation_script(recommendations, script_path):
    """Create a bash script to continue training for recommended models"""
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script to continue training for models showing positive trends\n")
        f.write("# Based on extended trend analysis\n\n")
        
        f.write('echo "=========================================================="\n')
        f.write('echo "Extended Training Continuation Script"\n')
        f.write('echo "=========================================================="\n')
        f.write('echo\n\n')
        
        # Define variables
        f.write("# Settings for continued training\n")
        f.write("ADDITIONAL_EPOCHS=30\n")
        f.write('AWS_INSTANCE="ec2-user@3.235.76.0"\n')
        f.write('SSH_KEY="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"\n\n')
        
        # Define models to train
        f.write("# Definite continue models\n")
        f.write("DEFINITE_CONTINUE_MODELS=(\n")
        for model, _, _ in recommendations['definite_continue']:
            f.write(f'  "{model}"\n')
        f.write(")\n\n")
        
        f.write("# Consider continue models\n")
        f.write("CONSIDER_CONTINUE_MODELS=(\n")
        for model, _, _ in recommendations['consider_continue']:
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
        f.write('echo "Models that will DEFINITELY be trained for $ADDITIONAL_EPOCHS more epochs:"\n')
        f.write('for model in "${DEFINITE_CONTINUE_MODELS[@]}"; do\n')
        f.write('  echo "- $model"\n')
        f.write('done\n')
        f.write('echo\n\n')
        
        f.write('echo "Models that will be CONSIDERED for further training:"\n')
        f.write('for model in "${CONSIDER_CONTINUE_MODELS[@]}"; do\n')
        f.write('  echo "- $model"\n')
        f.write('done\n')
        f.write('echo\n\n')
        
        f.write('# Ask user which models to train\n')
        f.write('read -p "Train definitely recommended models? (y/n): " train_definite\n')
        f.write('read -p "Train models to consider as well? (y/n): " train_consider\n')
        f.write('echo\n\n')
        
        f.write('# Execute training continuation for definite models\n')
        f.write('if [[ "$train_definite" == "y" ]]; then\n')
        f.write('  for model in "${DEFINITE_CONTINUE_MODELS[@]}"; do\n')
        f.write('    continue_training "$model" $ADDITIONAL_EPOCHS\n')
        f.write('  done\n')
        f.write('fi\n\n')
        
        f.write('# Execute training continuation for considered models\n')
        f.write('if [[ "$train_consider" == "y" ]]; then\n')
        f.write('  for model in "${CONSIDER_CONTINUE_MODELS[@]}"; do\n')
        f.write('    continue_training "$model" $ADDITIONAL_EPOCHS\n')
        f.write('  done\n')
        f.write('fi\n\n')
        
        f.write('echo "=========================================================="\n')
        f.write('echo "Monitoring training progress"\n')
        f.write('echo "=========================================================="\n')
        f.write('echo "To monitor training progress, use:"\n')
        f.write('echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \\"tail -f ~/emotion_training/training_MODEL_NAME.log\\""\n')
        f.write('echo\n')
        f.write('echo "To check validation accuracy at the end of training:"\n')
        f.write('echo "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $AWS_INSTANCE \\"grep -a \'val_accuracy:\' ~/emotion_training/training_MODEL_NAME.log | tail -5\\""\n')
        f.write('echo\n')
        f.write('echo "Script completed."\n')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Created training continuation script: {script_path}")

if __name__ == "__main__":
    main()
