#!/bin/bash

# Script to run VoteGCL with specified dataset
# Usage: ./run_votegcl.sh <dataset_name>
# Example: ./run_votegcl.sh movielens-100k

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 movielens-100k"
    echo "Available datasets: movielens-100k, amazon-book, amazon-scientific, yelp2018, etc."
    exit 1
fi

DATASET_NAME=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/conf/VoteGCL.yaml"

# Check if dataset directory exists
DATASET_DIR="$SCRIPT_DIR/dataset/$DATASET_NAME"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist!"
    echo "Available datasets:"
    ls -1 "$SCRIPT_DIR/dataset/" | grep -v ".*\\..*"
    exit 1
fi

# Check if required files exist
TRAIN_FILE="$DATASET_DIR/train.txt"
TEST_FILE="$DATASET_DIR/test.txt"
AUGMENTED_FILE="$DATASET_DIR/train_augmented.txt"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file '$TRAIN_FILE' does not exist!"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file '$TEST_FILE' does not exist!"
    exit 1
fi

if [ ! -f "$AUGMENTED_FILE" ]; then
    echo "Warning: Augmented file '$AUGMENTED_FILE' does not exist!"
    echo "Using regular training file instead."
    AUGMENTED_FILE="$TRAIN_FILE"
fi

echo "============================================"
echo "Running VoteGCL with dataset: $DATASET_NAME"
echo "============================================"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "Augmented file: $AUGMENTED_FILE"
echo "Config file: $CONFIG_FILE"
echo "============================================"

# Create backup of original config
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

# Update the configuration file
sed -i "s|^training.set:.*|training.set: ./dataset/$DATASET_NAME/train.txt|" "$CONFIG_FILE"
sed -i "s|^test.set:.*|test.set: ./dataset/$DATASET_NAME/test.txt|" "$CONFIG_FILE"
sed -i "s|^augmented.set:.*|augmented.set: ./dataset/$DATASET_NAME/train_augmented.txt|" "$CONFIG_FILE"

echo "Updated configuration file:"
echo "============================================"
grep -E "^(training|test|augmented)\.set:" "$CONFIG_FILE"
echo "============================================"

# Change to script directory
cd "$SCRIPT_DIR"

# Run the main script with VoteGCL model
echo "Starting VoteGCL training..."
echo "VoteGCL" | python main.py -i 0

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "VoteGCL training completed successfully!"
    echo "Results saved in: ./results/"
    echo "============================================"
else
    echo "============================================"
    echo "Error: VoteGCL training failed!"
    echo "============================================"
    # Restore backup config
    mv "$CONFIG_FILE.backup" "$CONFIG_FILE"
    exit 1
fi

# Restore backup config
mv "$CONFIG_FILE.backup" "$CONFIG_FILE"
echo "Configuration file restored to original state."
