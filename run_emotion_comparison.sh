#!/bin/bash
# Script to run the Emotion Recognition Method Comparison Framework

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_DIR="${SCRIPT_DIR}/emotion_comparison"
OUTPUT_DIR="${SCRIPT_DIR}/comparison_results"
RAVDESS_DIR="${SCRIPT_DIR}/downsampled_videos/RAVDESS"
CREMAD_DIR="${SCRIPT_DIR}/downsampled_videos/CREMA-D-audio-complete"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if required packages are installed
echo "Checking required Python packages..."
echo "Note: This is a demonstration framework. For a full setup install the packages listed in emotion_comparison/requirements.txt"
# To actually install packages uncomment the following line:
# pip install -r "${COMPARISON_DIR}/requirements.txt"

# Function to run comparison with specific parameters
run_comparison() {
    local dataset=$1
    local dataset_path=$2
    local methods=$3
    local max_videos=$4
    local extract=${5:-false}
    local classifiers=${6:-"random_forest"}
    local grid_search=${7:-false}

    echo "===================================================="
    echo "Running comparison on $dataset dataset"
    echo "Methods: $methods"
    echo "Classifiers: $classifiers"
    echo "===================================================="

    CMD="python ${COMPARISON_DIR}/run_comparison.py \
        --video_dir $dataset_path \
        --output_dir ${OUTPUT_DIR}/${dataset}_results \
        --methods \"$methods\" \
        --classifiers \"$classifiers\" \
        --cross_validate \
        --dataset_name \"$dataset\" \
        --visualize"

    # Add optional arguments
    if [ "$extract" = true ]; then
        CMD="$CMD --extract_features"
    fi

    if [ ! -z "$max_videos" ]; then
        CMD="$CMD --max_videos $max_videos"
    fi
    
    # Add grid search if requested
    if [ "$grid_search" = true ]; then
        CMD="$CMD --grid_search --param_grid_json ${COMPARISON_DIR}/param_grid_example.json"
    fi

    # Run the command
    echo "Executing: $CMD"
    eval $CMD

    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Error running comparison for $dataset with methods: $methods"
        echo "Please check logs for details."
        return 1
    fi

    echo "Completed $dataset comparison with methods: $methods"
    echo "Results saved to: ${OUTPUT_DIR}/${dataset}_results"
    
    # Open HTML report if available
    local report_path="${OUTPUT_DIR}/${dataset}_results/${dataset}_report.html"
    if [ -f "$report_path" ]; then
        echo "View the HTML report at: $report_path"
        
        # Check if we have a browser to open the report
        if command -v open &> /dev/null || command -v xdg-open &> /dev/null; then
            if [ "$AUTO_MODE" = false ]; then
                read -p "Open the visualization report now? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    if command -v open &> /dev/null; then
                        open "$report_path"
                    else
                        xdg-open "$report_path"
                    fi
                    echo "Report opened in browser."
                fi
            fi
        fi
    else
        echo "HTML report not generated. Check logs for errors."
    fi
    
    echo
    return 0
}

# Function to visualize existing results
visualize_results() {
    local dataset=$1
    
    if [ ! -d "${OUTPUT_DIR}/${dataset}_results" ]; then
        echo "Error: Results directory not found for $dataset"
        return 1
    fi
    
    echo "Generating visualizations for $dataset..."
    python ${COMPARISON_DIR}/run_comparison.py \
        --video_dir "dummy" \
        --output_dir "${OUTPUT_DIR}/${dataset}_results" \
        --methods "dummy" \
        --classifiers "dummy" \
        --dataset_name "$dataset" \
        --visualize
        
    # Open HTML report if available
    local report_path="${OUTPUT_DIR}/${dataset}_results/${dataset}_report.html"
    if [ -f "$report_path" ]; then
        echo "View the HTML report at: $report_path"
        
        # Check if we have a browser to open the report
        if command -v open &> /dev/null || command -v xdg-open &> /dev/null; then
            if [ "$AUTO_MODE" = false ]; then
                read -p "Open the visualization report now? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    if command -v open &> /dev/null; then
                        open "$report_path"
                    else
                        xdg-open "$report_path"
                    fi
                    echo "Report opened in browser."
                fi
            fi
        fi
    else
        echo "HTML report not generated. Check logs for errors."
    fi
    
    return 0
}

# Ensure the emotion_comparison module is in the Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Display help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Emotion Recognition Method Comparison Framework"
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --test, -t            Run a quick test on 10 videos"
    echo "  --sample, -s          Run on a sample of 50 videos"
    echo "  --auto, -a            Run without interactive prompts"
    echo "  --ravdess, -r         Run only on RAVDESS dataset"
    echo "  --cremad, -c          Run only on CREMA-D dataset"
    echo "  --visualize, -v       Generate visualizations for existing results"
    echo "  --visualize-ravdess   Generate visualizations for RAVDESS results"
    echo "  --visualize-cremad    Generate visualizations for CREMA-D results"
    echo "  --grid-search, -g     Enable hyperparameter grid search for classifiers"
    echo "  --help, -h            Display this help message"
    echo
    echo "All runs include visualization reports with:"
    echo "  - Accuracy comparison charts"
    echo "  - Confusion matrices"
    echo "  - HTML summary report"
    echo
    echo "Grid search options:"
    echo "  Use --grid-search to enable hyperparameter optimization"
    echo "  Edit ${COMPARISON_DIR}/param_grid_example.json to customize parameter grids"
    echo
    exit 0
fi

# Check for auto mode (no prompts)
AUTO_MODE=false
if [ "$1" = "--auto" ] || [ "$1" = "-a" ]; then
    AUTO_MODE=true
    shift # Remove this argument and continue with the next ones
fi

# Check for grid search
GRID_SEARCH=false
if [ "$1" = "--grid-search" ] || [ "$1" = "-g" ]; then
    GRID_SEARCH=true
    shift # Remove this argument and continue with the next ones
fi

# Check for visualization requests (for existing results)
if [ "$1" = "--visualize" ] || [ "$1" = "-v" ]; then
    if [ -d "${OUTPUT_DIR}/RAVDESS_results" ]; then
        visualize_results "RAVDESS"
    fi
    
    if [ -d "${OUTPUT_DIR}/CREMAD_results" ]; then
        visualize_results "CREMAD"
    fi
    
    if [ ! -d "${OUTPUT_DIR}/RAVDESS_results" ] && [ ! -d "${OUTPUT_DIR}/CREMAD_results" ]; then
        echo "No existing results found to visualize."
        echo "Run a comparison first or specify a dataset with --visualize-ravdess or --visualize-cremad"
    fi
    
    exit 0
fi

if [ "$1" = "--visualize-ravdess" ]; then
    visualize_results "RAVDESS"
    exit 0
fi

if [ "$1" = "--visualize-cremad" ]; then
    visualize_results "CREMAD"
    exit 0
fi

# Check if we should run a small test first
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    echo "Running small test on 10 videos..."
    run_comparison "RAVDESS_TEST" "$RAVDESS_DIR" "cnn3d facs" 10 true "random_forest" $GRID_SEARCH
    exit 0
fi

# Run on a small sample first if no arguments provided or specifically requested
if [ -z "$1" ] || [ "$1" = "--sample" ] || [ "$1" = "-s" ]; then
    echo "Running on a small sample of RAVDESS videos first..."
    run_comparison "RAVDESS_SAMPLE" "$RAVDESS_DIR" "cnn3d facs multi_region pretrained" 50 true "random_forest neural_network" $GRID_SEARCH

    if [ "$AUTO_MODE" = false ]; then
        read -p "Continue with full dataset comparison? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting without running full comparison."
            exit 0
        fi
    else
        echo "Auto mode: continuing with full dataset comparison"
    fi
fi

# Run full comparison on RAVDESS
if [ "$1" = "--ravdess" ] || [ "$1" = "-r" ] || [ -z "$1" ]; then
    echo "Running full comparison on RAVDESS dataset..."

    # First extract features with all methods
    run_comparison "RAVDESS" "$RAVDESS_DIR" "cnn3d facs multi_region pretrained" "" true "random_forest" false

    if [ $? -ne 0 ]; then
        echo "Feature extraction failed. Skipping evaluation step."
    else
        # Then train and evaluate with different classifiers
        run_comparison "RAVDESS" "$RAVDESS_DIR" "cnn3d facs multi_region pretrained" "" false "random_forest svm neural_network" $GRID_SEARCH
    fi
fi

# Run full comparison on CREMA-D
if [ "$1" = "--cremad" ] || [ "$1" = "-c" ]; then
    echo "Running full comparison on CREMA-D dataset..."

    # First extract features with all methods
    run_comparison "CREMAD" "$CREMAD_DIR" "cnn3d facs multi_region pretrained" "" true "random_forest" false

    if [ $? -ne 0 ]; then
        echo "Feature extraction failed. Skipping evaluation step."
    else
        # Then train and evaluate with different classifiers
        run_comparison "CREMAD" "$CREMAD_DIR" "cnn3d facs multi_region pretrained" "" false "random_forest svm neural_network" $GRID_SEARCH
    fi
fi

# Generate a combined report if both datasets were processed
if [ -d "${OUTPUT_DIR}/RAVDESS_results" ] && [ -d "${OUTPUT_DIR}/CREMAD_results" ]; then
    echo "Generating combined cross-dataset report..."

    # This would be a Python script to compare results across datasets
    python ${COMPARISON_DIR}/cross_dataset_analysis.py \
       --ravdess_dir "${OUTPUT_DIR}/RAVDESS_results" \
       --cremad_dir "${OUTPUT_DIR}/CREMAD_results" \
       --output_dir "${OUTPUT_DIR}/combined_results" 2>/dev/null || \
       echo "Cross-dataset comparison script not yet implemented"
fi

echo "All comparison tasks completed!"
echo "Results are available in: $OUTPUT_DIR"

# Provide a summary of the best methods
if [ -f "${OUTPUT_DIR}/RAVDESS_results/comparison_summary.txt" ]; then
    echo -e "\nRAVDESS Dataset Summary:"
    grep -A 5 "best-performing method" "${OUTPUT_DIR}/RAVDESS_results/comparison_summary.txt" ||
    echo "No performance summary found. Check the results directory for details."
fi

if [ -f "${OUTPUT_DIR}/CREMAD_results/comparison_summary.txt" ]; then
    echo -e "\nCREMA-D Dataset Summary:"
    grep -A 5 "best-performing method" "${OUTPUT_DIR}/CREMAD_results/comparison_summary.txt" ||
    echo "No performance summary found. Check the results directory for details."
fi

# Provide links to HTML reports
echo -e "\nVisualization Reports:"
if [ -f "${OUTPUT_DIR}/RAVDESS_results/RAVDESS_report.html" ]; then
    echo "  RAVDESS: ${OUTPUT_DIR}/RAVDESS_results/RAVDESS_report.html"
fi
if [ -f "${OUTPUT_DIR}/CREMAD_results/CREMAD_report.html" ]; then
    echo "  CREMA-D: ${OUTPUT_DIR}/CREMAD_results/CREMAD_report.html"
fi
if [ -f "${OUTPUT_DIR}/RAVDESS_SAMPLE_results/RAVDESS_SAMPLE_report.html" ]; then
    echo "  Sample:  ${OUTPUT_DIR}/RAVDESS_SAMPLE_results/RAVDESS_SAMPLE_report.html"
fi

echo -e "\nCommand reference:"
echo "  ./run_emotion_comparison.sh --test           # Run a quick test on 10 videos"
echo "  ./run_emotion_comparison.sh --sample         # Run on a sample of 50 videos"
echo "  ./run_emotion_comparison.sh --auto           # Run without interactive prompts"
echo "  ./run_emotion_comparison.sh --ravdess        # Run only on RAVDESS dataset"
echo "  ./run_emotion_comparison.sh --cremad         # Run only on CREMA-D dataset"
echo "  ./run_emotion_comparison.sh --visualize      # Generate visualizations for existing results"
echo "  ./run_emotion_comparison.sh --help           # Display complete help"
