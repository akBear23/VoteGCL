#!/bin/bash

# GPT Few-Shot Re-ranking Script for Recommendation Systems
# This script runs LLM-based re-ranking using OpenAI GPT API with few-shot prompting

# Default values
DATA_NAME=""
USER_QUANTILE=0.25
API_CALLS_PER_USER=8
GPT_MODEL="gpt-3.5-turbo"
API_KEY=""
CANDIDATE_COUNT=10
START_USER_INDEX=0
END_USER_INDEX=""
API_DELAY=1.0
TEMPERATURE=0.8
MAX_TOKENS=1000

# Function to display help
show_help() {
    echo "Usage: $0 --data_name DATA_NAME --api_key API_KEY [OPTIONS]"
    echo ""
    echo "GPT Few-Shot Re-ranking Script for Recommendation Systems"
    echo "This script performs LLM-based re-ranking using OpenAI GPT API with few-shot prompting."
    echo "It automatically detects model files in the dataset folder for user similarity calculations."
    echo ""
    echo "Required arguments:"
    echo "  --data_name DATA_NAME           Name of the dataset folder (e.g., movielen-100k, amazon-scientific)"
    echo "  --api_key API_KEY               OpenAI API key"
    echo ""
    echo "Optional arguments:"
    echo "  --user_quantile QUANTILE        User interaction count quantile (default: 0.25)"
    echo "  --api_calls_per_user CALLS      Number of API calls per user (default: 8)"
    echo "  --gpt_model MODEL               GPT model name (default: gpt-3.5-turbo)"
    echo "  --candidate_count COUNT         Number of candidate items (default: 10)"
    echo "  --start_user_index INDEX        Start user index for processing (default: 0)"
    echo "  --end_user_index INDEX          End user index for processing (default: all users)"
    echo "  --api_delay DELAY               Delay between API calls in seconds (default: 1.0)"
    echo "  --temperature TEMP              Temperature for GPT API calls (default: 0.8)"
    echo "  --max_tokens TOKENS             Maximum tokens for GPT response (default: 1000)"
    echo "  --model_file FILE               Path to saved model file (optional, auto-detected if not provided)"
    echo "  --help                          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --data_name movielen-100k --api_key YOUR_API_KEY"
    echo "  $0 --data_name movielen-100k --api_key YOUR_API_KEY --candidate_count 10 --temperature 0.7"
    echo "  $0 --data_name movielen-100k --api_key YOUR_API_KEY --start_user_index 0 --end_user_index 100"
    echo ""
    echo "Note:"
    echo "  - Model files are automatically detected in the dataset folder"
    echo "  - The script will look for model files with patterns like *.pkl, *LightGCN*.pkl, etc."
    echo "  - Few-shot prompting uses similar user behavior patterns for better recommendations"
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
        --gpt_model)
            GPT_MODEL="$2"
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
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --model_file)
            MODEL_FILE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_NAME" ]]; then
    echo "Error: --data_name is required"
    echo "Use --help for usage information."
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    echo "Error: --api_key is required"
    echo "Use --help for usage information."
    exit 1
fi

# Build the command
CMD="python3 LLM_rerank_gpt_fewshot.py"
CMD="$CMD --data_name \"$DATA_NAME\""
CMD="$CMD --user_quantile $USER_QUANTILE"
CMD="$CMD --api_calls_per_user $API_CALLS_PER_USER"
CMD="$CMD --gpt_model \"$GPT_MODEL\""
CMD="$CMD --api_key \"$API_KEY\""
CMD="$CMD --candidate_count $CANDIDATE_COUNT"
CMD="$CMD --start_user_index $START_USER_INDEX"
CMD="$CMD --api_delay $API_DELAY"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max_tokens $MAX_TOKENS"

if [[ -n "$END_USER_INDEX" ]]; then
    CMD="$CMD --end_user_index $END_USER_INDEX"
fi

if [[ -n "$MODEL_FILE" ]]; then
    CMD="$CMD --model_file \"$MODEL_FILE\""
fi

# Display the command being executed
echo "Executing GPT Few-Shot Re-ranking..."
echo "Dataset: $DATA_NAME"
echo "Model: $GPT_MODEL"
echo "User Quantile: $USER_QUANTILE"
echo "API Calls per User: $API_CALLS_PER_USER"
echo "Candidate Count: $CANDIDATE_COUNT"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
if [[ -n "$MODEL_FILE" ]]; then
    echo "Model File: $MODEL_FILE"
else
    echo "Model File: Auto-detected from dataset folder"
fi
echo ""

# Execute the command
eval $CMD
