#!/bin/bash

# Parallel evaluation script for multiple merged models
# This script runs eval_image_all.py for each model on different GPUs in parallel

# Configuration
BASE_MODEL_DIR="/home6/fzy/repos/EAGLE/checkpoints/Images/merged_model/renamed"
LOG_DIR="/home6/fzy/repos/EAGLE/eval_image/logs"
SCRIPT_PATH="/home6/fzy/repos/EAGLE/eval_image/eval_image_all.py"

# Available GPUs (modify according to your setup)
GPUS=(0 )

# Model directories
MODELS=(
    # "0.6_0.4"
    # "0.7_0.3" 
    # "0.8_0.2"
    "0.9_0.1"
    # "0.99_0.01"
    # "0.999_0.001"
    # "0.9999_0.0001"
)

# Default datasets to evaluate (can be overridden)
DATASETS="${1:-all}"
PARALLEL_MODE="${2:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run evaluation for a single model
run_single_evaluation() {
    local model_name=$1
    local gpu_id=$2
    local process_id=$3
    
    local model_path="${BASE_MODEL_DIR}/${model_name}"
    local log_file="${LOG_DIR}/eval_${model_name}_gpu${gpu_id}_proc${process_id}.log"
    local error_log="${LOG_DIR}/eval_${model_name}_gpu${gpu_id}_proc${process_id}.err"
    
    print_status $BLUE "🚀 [Process $process_id] Starting evaluation for model: $model_name on GPU $gpu_id"
    print_status $YELLOW "📝 Log file: $log_file"
    print_status $YELLOW "📝 Error log: $error_log"
    
    # Check if model directory exists
    if [ ! -d "$model_path" ]; then
        print_status $RED "❌ [Process $process_id] Model directory not found: $model_path"
        echo "[$(date)] ERROR: Model directory not found: $model_path" > "$error_log"
        return 1
    fi
    
    # Start timestamp
    echo "[$(date)] Starting evaluation for model: $model_name on GPU $gpu_id" > "$log_file"
    echo "Model path: $model_path" >> "$log_file"
    echo "Datasets: $DATASETS" >> "$log_file"
    echo "===========================================" >> "$log_file"
    
    # Run the evaluation
    cd /home6/fzy/repos/EAGLE/eval_image || {
        print_status $RED "❌ [Process $process_id] Failed to change to eval_image directory"
        echo "[$(date)] ERROR: Failed to change to eval_image directory" >> "$error_log"
        return 1
    }
    
    # Execute the Python script with proper environment
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --model_path "$model_path" \
        --datasets "$DATASETS" \
        --gpus "$gpu_id" \
        --sequential \
        >> "$log_file" 2>> "$error_log"
    
    local exit_code=$?
    
    # End timestamp and status
    echo "===========================================" >> "$log_file"
    echo "[$(date)] Evaluation completed with exit code: $exit_code" >> "$log_file"
    
    if [ $exit_code -eq 0 ]; then
        print_status $GREEN "✅ [Process $process_id] Successfully completed evaluation for model: $model_name"
    else
        print_status $RED "❌ [Process $process_id] Failed evaluation for model: $model_name (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to wait for a process and report its completion
wait_for_process() {
    local pid=$1
    local model_name=$2
    local process_id=$3
    
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_status $GREEN "🎉 [Process $process_id] Model $model_name evaluation completed successfully"
    else
        print_status $RED "💥 [Process $process_id] Model $model_name evaluation failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Main execution function
main() {
    print_status $BLUE "🔥 Starting parallel model evaluation"
    print_status $YELLOW "📊 Datasets to evaluate: $DATASETS"
    print_status $YELLOW "🖥️  Available GPUs: ${GPUS[*]}"
    print_status $YELLOW "🤖 Models to evaluate: ${#MODELS[@]}"
    print_status $YELLOW "📁 Log directory: $LOG_DIR"
    echo ""
    
    # Check if eval_image_all.py exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_status $RED "❌ Evaluation script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    # Array to store background process IDs
    declare -a PIDS=()
    declare -a PROCESS_INFO=()
    
    # Start evaluations in parallel
    local process_id=1
    for i in "${!MODELS[@]}"; do
        local model_name="${MODELS[$i]}"
        local gpu_id="${GPUS[$((i % ${#GPUS[@]}))]}"  # Cycle through available GPUs
        
        # Run evaluation in background
        run_single_evaluation "$model_name" "$gpu_id" "$process_id" &
        local pid=$!
        
        PIDS+=($pid)
        PROCESS_INFO+=("$process_id:$model_name:$gpu_id")
        
        print_status $BLUE "📋 [Process $process_id] Started background process (PID: $pid) for model: $model_name on GPU $gpu_id"
        
        process_id=$((process_id + 1))
        
        # Small delay to prevent overwhelming the system
        sleep 2
    done
    
    print_status $YELLOW "⏳ All evaluations started. Waiting for completion..."
    echo ""
    
    # Wait for all processes to complete
    local failed_count=0
    local success_count=0
    
    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local info="${PROCESS_INFO[$i]}"
        IFS=':' read -r proc_id model_name gpu_id <<< "$info"
        
        wait_for_process "$pid" "$model_name" "$proc_id"
        if [ $? -eq 0 ]; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    done
    
    # Final summary
    echo ""
    print_status $BLUE "📊 Evaluation Summary:"
    print_status $GREEN "✅ Successful evaluations: $success_count"
    print_status $RED "❌ Failed evaluations: $failed_count"
    print_status $YELLOW "📁 Log files location: $LOG_DIR"
    
    # List log files
    echo ""
    print_status $YELLOW "📄 Generated log files:"
    ls -la "$LOG_DIR"/*.log 2>/dev/null | while read -r line; do
        echo "   $line"
    done
    
    # Check for error files
    if ls "$LOG_DIR"/*.err 1> /dev/null 2>&1; then
        echo ""
        print_status $RED "⚠️  Error log files found:"
        ls -la "$LOG_DIR"/*.err | while read -r line; do
            echo "   $line"
        done
    fi
    
    if [ $failed_count -eq 0 ]; then
        print_status $GREEN "🎉 All evaluations completed successfully!"
        exit 0
    else
        print_status $RED "💥 Some evaluations failed. Check error logs for details."
        exit 1
    fi
}

# Handle script arguments and help
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [DATASETS] [PARALLEL_MODE]"
        echo ""
        echo "Arguments:"
        echo "  DATASETS      Datasets to evaluate (default: 'all')"
        echo "                Options: 'all', 'mmlu', 'mme', 'docvqa', 'textvqa', 'chartqa', 'ocrbenchv2'"
        echo "                Or comma-separated list: 'mmlu,mme,docvqa'"
        echo "  PARALLEL_MODE Whether to run in parallel mode (default: 'true')"
        echo ""
        echo "Examples:"
        echo "  $0                           # Run all datasets"
        echo "  $0 mmlu                      # Run only MMLU"
        echo "  $0 'mmlu,mme'                # Run MMLU and MME"
        echo "  $0 all true                  # Run all datasets in parallel (default)"
        echo ""
        echo "Models to be evaluated:"
        for model in "${MODELS[@]}"; do
            echo "  - $model"
        done
        echo ""
        echo "Log files will be saved to: $LOG_DIR"
        exit 0
        ;;
    *)
        main
        ;;
esac