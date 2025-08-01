#!/bin/bash

# Monitor script for parallel evaluations
LOG_DIR="/home6/fzy/repos/EAGLE/eval_image/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

case "${1:-status}" in
    "status"|"")
        print_status $BLUE "📊 Evaluation Status:"
        echo ""
        
        if [ ! -d "$LOG_DIR" ]; then
            print_status $RED "❌ Log directory not found: $LOG_DIR"
            exit 1
        fi
        
        # Count log files
        log_count=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
        err_count=$(ls "$LOG_DIR"/*.err 2>/dev/null | wc -l)
        
        print_status $YELLOW "📄 Log files: $log_count"
        print_status $YELLOW "⚠️  Error files: $err_count"
        echo ""
        
        # Show recent activity
        if [ $log_count -gt 0 ]; then
            print_status $BLUE "🔄 Recent activity (last 10 lines from each log):"
            for log_file in "$LOG_DIR"/*.log; do
                if [ -f "$log_file" ]; then
                    echo ""
                    print_status $GREEN "📝 $(basename "$log_file"):"
                    tail -n 20 "$log_file" | sed 's/^/   /'
                fi
            done
        fi
        ;;
        
    "errors")
        print_status $RED "❌ Error Analysis:"
        echo ""
        
        if ls "$LOG_DIR"/*.err 1> /dev/null 2>&1; then
            for err_file in "$LOG_DIR"/*.err; do
                if [ -s "$err_file" ]; then  # Check if file is not empty
                    echo ""
                    print_status $RED "💥 $(basename "$err_file"):"
                    cat "$err_file" | sed 's/^/   /'
                fi
            done
        else
            print_status $GREEN "✅ No error files found!"
        fi
        ;;
        
    "results")
        print_status $BLUE "📊 Results Summary:"
        echo ""
        
        # Look for result files in the expected locations
        BASE_RESULTS_DIR="/home6/fzy/repos/EAGLE/eval_image/res_folder/images"
        
        if [ -d "$BASE_RESULTS_DIR" ]; then
            for model_dir in "$BASE_RESULTS_DIR"/*/; do
                if [ -d "$model_dir" ]; then
                    model_name=$(basename "$model_dir")
                    print_status $GREEN "🤖 Model: $model_name"
                    
                    # Look for JSON summary files
                    json_files=$(find "$model_dir" -name "summary_*.json" 2>/dev/null)
                    if [ -n "$json_files" ]; then
                        echo "$json_files" | while read -r json_file; do
                            echo "   📄 $(basename "$json_file")"
                            # Extract key metrics if possible
                            if command -v jq >/dev/null 2>&1; then
                                accuracy=$(jq -r '.summary_statistics | to_entries[] | select(.value.overall_accuracy != null) | "\(.key): \(.value.overall_accuracy)"' "$json_file" 2>/dev/null)
                                if [ -n "$accuracy" ] && [ "$accuracy" != "null" ]; then
                                    echo "   📈 $accuracy"
                                fi
                            fi
                        done
                    else
                        echo "   ⚠️  No summary files found"
                    fi
                    echo ""
                fi
            done
        else
            print_status $YELLOW "⚠️  Results directory not found: $BASE_RESULTS_DIR"
        fi
        ;;
        
    "clean")
        print_status $YELLOW "🧹 Cleaning old log files..."
        rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.err
        print_status $GREEN "✅ Log files cleaned!"
        ;;
        
    "tail")
        model_pattern="${2:-*}"
        print_status $BLUE "📄 Tailing logs for pattern: $model_pattern"
        echo ""
        
        for log_file in "$LOG_DIR"/eval_${model_pattern}_*.log; do
            if [ -f "$log_file" ]; then
                print_status $GREEN "📝 Following: $(basename "$log_file")"
                tail -f "$log_file" &
            fi
        done
        
        # Wait for all tail processes
        wait
        ;;
        
    "help"|"-h"|"--help")
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  status     Show current evaluation status (default)"
        echo "  errors     Show error logs"
        echo "  results    Show evaluation results summary"
        echo "  clean      Clean old log files"
        echo "  tail [PATTERN]  Follow log files (use Ctrl+C to stop)"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Show status"
        echo "  $0 errors             # Show errors"
        echo "  $0 results            # Show results"
        echo "  $0 tail 0.9_0.1       # Follow logs for specific model"
        echo "  $0 clean              # Clean log files"
        ;;
        
    *)
        print_status $RED "❌ Unknown command: $1"
        print_status $YELLOW "💡 Use '$0 help' for usage information"
        exit 1
        ;;
esac
