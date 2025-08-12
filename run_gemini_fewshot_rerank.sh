#!/bin/bash

# LLM Re-ranking with Few-Shot Prompting Script
# This script runs the LLM re-ranking with few-shot prompting using Gemini API

set -e  # Exit on any error

# Default values
DATA_NAME=""
USER_QUANTILE=0.25
API_CALLS_PER_USER=8
GEMINI_MODEL="gemini-2.0-flash"
API_KEY=""
CANDIDATE_COUNT=10
START_USER_INDEX=0
END_USER_INDEX=""
API_DELAY=1.0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 --data_name <dataset> --api_key <key> [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --data_name           Dataset name (e.g., amazon-scientific, movielen-100k)"
    echo "  --api_key             Google API key for Gemini"
    echo ""
    echo "Optional Arguments:"
    echo "  --user_quantile       User interaction count quantile (default: 0.25)"
    echo "  --api_calls_per_user  Number of API calls per user (default: 8)"
    echo "  --gemini_model        Gemini model name (default: gemini-2.0-flash)"
    echo "  --candidate_count     Number of candidate items (default: 10)"
    echo "  --start_user_index    Start user index (default: 0)"
    echo "  --end_user_index      End user index (default: all users)"
    echo "  --api_delay           Delay between API calls in seconds (default: 1.0)"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --data_name movielen-100k --api_key YOUR_API_KEY"
    echo "  $0 --data_name movielen-100k --api_key YOUR_API_KEY --user_quantile 0.3 --api_calls_per_user 5"
    echo ""
    echo "Note: The script will automatically look for a model file (.pkl) in the dataset folder for user similarity computation."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_name)
            DATA_NAME="$2"
            shift 2
            ;;
        --user_quantile)
            USER_QUANTILE="$2"
            shift 2
            ;;
        --api_calls_per_user)
            API_CALLS_PER_USER="$2"
            shift 2
            ;;
        --gemini_model)
            GEMINI_MODEL="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --candidate_count)
            CANDIDATE_COUNT="$2"
            shift 2
            ;;
        --start_user_index)
            START_USER_INDEX="$2"
            shift 2
            ;;
        --end_user_index)
            END_USER_INDEX="$2"
            shift 2
            ;;
        --api_delay)
            API_DELAY="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_color $RED "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_NAME" ]]; then
    print_color $RED "Error: --data_name is required"
    show_usage
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    print_color $RED "Error: --api_key is required"
    show_usage
    exit 1
fi

# Validate files and directories
if [[ ! -d "dataset/$DATA_NAME" ]]; then
    print_color $RED "Error: Dataset directory 'dataset/$DATA_NAME' not found"
    exit 1
fi

if [[ ! -f "dataset/$DATA_NAME/rec_list_all_user.pkl" ]]; then
    print_color $RED "Error: Recommendation list 'dataset/$DATA_NAME/rec_list_all_user.pkl' not found"
    print_color $YELLOW "Please run create_rec_list.py first to generate the recommendation list"
    exit 1
fi

if [[ ! -f "dataset/$DATA_NAME/train.txt" ]]; then
    print_color $RED "Error: Training data 'dataset/$DATA_NAME/train.txt' not found"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p results

# Print configuration
print_color $BLUE "=" 60
print_color $BLUE "LLM RE-RANKING WITH FEW-SHOT PROMPTING CONFIGURATION"
print_color $BLUE "=" 60
echo "Dataset: $DATA_NAME"
echo "User quantile: $USER_QUANTILE"
echo "API calls per user: $API_CALLS_PER_USER"
echo "Gemini model: $GEMINI_MODEL"
echo "Candidate count: $CANDIDATE_COUNT"
if [[ -n "$END_USER_INDEX" ]]; then
    echo "User range: $START_USER_INDEX to $END_USER_INDEX"
else
    echo "User range: $START_USER_INDEX to end"
fi
echo "API delay: $API_DELAY seconds"
echo "Model lookup: Automatic (from dataset folder)"
print_color $BLUE "=" 60

# Prepare command
PYTHON_CMD="python3 LLM_rerank_gemini_fewshot.py"
PYTHON_CMD="$PYTHON_CMD --data_name $DATA_NAME"
PYTHON_CMD="$PYTHON_CMD --user_quantile $USER_QUANTILE"
PYTHON_CMD="$PYTHON_CMD --api_calls_per_user $API_CALLS_PER_USER"
PYTHON_CMD="$PYTHON_CMD --gemini_model $GEMINI_MODEL"
PYTHON_CMD="$PYTHON_CMD --api_key $API_KEY"
PYTHON_CMD="$PYTHON_CMD --candidate_count $CANDIDATE_COUNT"
PYTHON_CMD="$PYTHON_CMD --start_user_index $START_USER_INDEX"
PYTHON_CMD="$PYTHON_CMD --api_delay $API_DELAY"

if [[ -n "$END_USER_INDEX" ]]; then
    PYTHON_CMD="$PYTHON_CMD --end_user_index $END_USER_INDEX"
fi

# Run the command
print_color $GREEN "Starting LLM re-ranking with few-shot prompting..."
echo "Command: $PYTHON_CMD"
echo ""

# Execute the Python script
if eval $PYTHON_CMD; then
    print_color $GREEN "✓ LLM re-ranking with few-shot prompting completed successfully!"
    
    # Show output files
    print_color $BLUE "\nGenerated files:"
    if [[ -n "$END_USER_INDEX" ]]; then
        RANGE_SUFFIX="${START_USER_INDEX}_${END_USER_INDEX}"
    else
        RANGE_SUFFIX="${START_USER_INDEX}_end"
    fi
    
    echo "  - dataset/$DATA_NAME/gemini_fewshot_reranked_responses_${RANGE_SUFFIX}.csv"
    echo "  - dataset/$DATA_NAME/token_summary_fewshot_${RANGE_SUFFIX}.csv"
    echo "  - dataset/$DATA_NAME/gemini_fewshot_reranked_prediction_${RANGE_SUFFIX}.csv"
    echo "  - dataset/$DATA_NAME/train_augmented_gemini_fewshot.txt"
    
else
    print_color $RED "✗ LLM re-ranking with few-shot prompting failed!"
    exit 1
fi

print_color $BLUE "=" 60
print_color $GREEN "LLM re-ranking with few-shot prompting completed!"
print_color $BLUE "=" 60
