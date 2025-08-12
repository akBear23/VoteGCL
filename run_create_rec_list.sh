#!/bin/bash

# Example usage script for create_rec_list.py
# This script shows how to use the cleaned recommendation list creation script

echo "Creating recommendation list for different datasets..."

# Example 1: MovieLen-100k dataset
echo "Example 1: Processing MovieLen-100k dataset"
python create_rec_list.py \
    --data_name "movielen-100k" \
    --model_name "LightGCN" \
    --num_candidates 10 \
    --max_epoch 100 \
    --embedding_size 256 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --reg_lambda 0.0001 \
    --n_layers 2

echo "MovieLen-100k processing complete!"

# # Example 2: Amazon Scientific dataset
# echo "Example 2: Processing Amazon Scientific dataset"
# python create_rec_list.py \
#     --data_name "amazon-scientific" \
#     --model_name "LightGCN" \
#     --num_candidates 10 \
#     --max_epoch 100 \
#     --embedding_size 256 \
#     --batch_size 2048 \
#     --learning_rate 0.001 \
#     --reg_lambda 0.0001 \
#     --n_layers 3

# echo "Amazon Scientific processing complete!"

# # Example 3: Skip training (if model already exists)
# echo "Example 3: Generate candidates without training"
# python create_rec_list.py \
#     --data_name "movielen-100k" \
#     --num_candidates 10 \
#     --skip_training

# echo "All examples complete!"
