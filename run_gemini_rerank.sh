#!/bin/bash

# Script to run Gemini LLM Re-ranking for Recommendation Systems
# Usage: ./run_gemini_rerank.sh <dataset_name> <api_key> [options]
# Example: ./run_gemini_rerank.sh movielen-100k YOUR_GOOGLE_API_KEY

print_usage() {
    echo "Usage: $0 --data_name <dataset> --api_key <key> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --data_name DATASET          Dataset folder name (e.g., movielen-100k, amazon-scientific)"
    echo "  --api_key KEY                Google API key for Gemini"
    echo ""
    echo "Optional arguments:"
    echo "  --user_quantile FLOAT        User interaction count quantile (default: 0.25)"
    echo "  --api_calls_per_user INT     Number of API calls per user (default: 8)"
    echo "  --gemini_model STRING        Gemini model name (default: gemini-2.0-flash)"
    echo "  --candidate_count INT        Number of candidate items (default: 10)"
    echo "  --start_user_index INT       Start user index for processing (default: 0)"
    echo "  --end_user_index INT         End user index for processing (default: all users)"
    echo "  --api_delay FLOAT            Delay between API calls in seconds (default: 1.0)"
    echo "  --help, -h                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --data_name movielen-100k --api_key YOUR_GOOGLE_API_KEY"
    echo "  $0 --data_name amazon-scientific --api_key YOUR_KEY --user_quantile 0.5 --candidate_count 10"
    echo "  $0 --data_name yelp2018 --api_key YOUR_KEY --start_user_index 0 --end_user_index 50"
    echo ""
    echo "Available datasets:"
    if [ -d "dataset" ]; then
        ls -1 dataset/ | grep -v ".*\\..*" | sed 's/^/  /'
    else
        echo "  (dataset directory not found)"
    fi
}

if [ $# -lt 1 ]; then
    echo "Error: Missing required arguments"
    echo ""
    print_usage
    exit 1
fi

DATASET_NAME=""
API_KEY=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
USER_QUANTILE="0.25"
API_CALLS_PER_USER="8"
GEMINI_MODEL="gemini-2.0-flash"
CANDIDATE_COUNT="10"
START_USER_INDEX="0"
END_USER_INDEX=""
API_DELAY="1.0"

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
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
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo ""
            print_usage
            exit 1
            ;;
    esac
done

# Validation
if [ -z "$DATASET_NAME" ] || [ -z "$API_KEY" ]; then
    echo "Error: Dataset name and API key are required"
    print_usage
    exit 1
fi

# Check if dataset directory exists
DATASET_DIR="$SCRIPT_DIR/dataset/$DATASET_NAME"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist!"
    echo ""
    echo "Available datasets:"
    if [ -d "$SCRIPT_DIR/dataset" ]; then
        ls -1 "$SCRIPT_DIR/dataset/" | grep -v ".*\\..*" | sed 's/^/  /'
    fi
    exit 1
fi

# Check if required files exist
REC_LIST_FILE="$DATASET_DIR/rec_list_all_user.pkl"
TRAIN_FILE="$DATASET_DIR/train.txt"

if [ ! -f "$REC_LIST_FILE" ]; then
    echo "Error: Recommendation list file '$REC_LIST_FILE' does not exist!"
    echo "Please run create_rec_list.py first to generate the recommendation list."
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file '$TRAIN_FILE' does not exist!"
    exit 1
fi

# Check metadata file
if [ "$DATASET_NAME" = "movielen-100k" ]; then
    META_FILE="$DATASET_DIR/movies.csv"
else
    META_FILE="$DATASET_DIR/meta_${DATASET_NAME}.json"
fi

if [ ! -f "$META_FILE" ]; then
    echo "Warning: Metadata file '$META_FILE' does not exist!"
    echo "Item details will be incomplete, but processing will continue."
fi

echo "============================================"
echo "Running Gemini LLM Re-ranking"
echo "============================================"
echo "Dataset: $DATASET_NAME"
echo "Model: $GEMINI_MODEL"
echo "User quantile: $USER_QUANTILE"
echo "API calls per user: $API_CALLS_PER_USER"
echo "Candidate count: $CANDIDATE_COUNT"
echo "Start user index: $START_USER_INDEX"
if [ -n "$END_USER_INDEX" ]; then
    echo "End user index: $END_USER_INDEX"
else
    echo "End user index: (all users)"
fi
echo "API delay: ${API_DELAY}s"
echo "============================================"
echo "Files:"
echo "  Rec list: $REC_LIST_FILE"
echo "  Train file: $TRAIN_FILE"
echo "  Metadata: $META_FILE"
echo "============================================"

# Change to script directory
cd "$SCRIPT_DIR"

# Build command arguments
CMD_ARGS=(
    "--data_name" "$DATASET_NAME"
    "--api_key" "$API_KEY"
    "--user_quantile" "$USER_QUANTILE"
    "--api_calls_per_user" "$API_CALLS_PER_USER"
    "--gemini_model" "$GEMINI_MODEL"
    "--candidate_count" "$CANDIDATE_COUNT"
    "--start_user_index" "$START_USER_INDEX"
    "--api_delay" "$API_DELAY"
)

if [ -n "$END_USER_INDEX" ]; then
    CMD_ARGS+=("--end_user_index" "$END_USER_INDEX")
fi

# Run the Gemini re-ranking script
echo "Starting Gemini LLM re-ranking..."
echo "Command: python LLM_rerank_gemini.py ${CMD_ARGS[*]}"
echo "============================================"

python LLM_rerank_gemini.py "${CMD_ARGS[@]}"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Gemini LLM re-ranking completed successfully!"
    echo ""
    echo "Output files in dataset/$DATASET_NAME/:"
    echo "  - gemini_reranked_responses_${START_USER_INDEX}_*.csv (detailed results)"
    echo "  - token_summary_${START_USER_INDEX}_*.csv (token usage)"
    echo "  - gemini_reranked_prediction_${START_USER_INDEX}_*.csv (ensemble predictions)"
    echo "  - train_augmented.txt (augmented training data)"
    echo "============================================"
else
    echo "============================================"
    echo "Error: Gemini LLM re-ranking failed!"
    echo "Check the error messages above for details."
    echo "============================================"
    exit 1
fi
