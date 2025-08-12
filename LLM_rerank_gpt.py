#!/usr/bin/env python3
"""
LLM Re-ranking for Recommendation Systems using OpenAI GPT API

This script performs LLM-based re-ranking of recommendation candidates using OpenAI GPT API.
It processes users based on interaction count quantiles and generates ensemble predictions.
"""

import os
import sys
import argparse
import pickle
import json
import time
import re
import csv
from typing import Dict, List, Tuple, Optional

import pandas as pd
import openai
from openai import OpenAI


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Re-ranking for Recommendation Systems using GPT')
    
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset folder (e.g., amazon-scientific)')
    parser.add_argument('--user_quantile', type=float, default=0.25,
                        help='User interaction count quantile (default: 0.25)')
    parser.add_argument('--api_calls_per_user', type=int, default=8,
                        help='Number of API calls per user (default: 8)')
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo',
                        help='GPT model name (default: gpt-3.5-turbo)')
    parser.add_argument('--api_key', type=str, required=True,
                        help='OpenAI API key')
    parser.add_argument('--candidate_count', type=int, default=10,
                        help='Number of candidate items (default: 10)')
    parser.add_argument('--start_user_index', type=int, default=0,
                        help='Start user index for processing (default: 0)')
    parser.add_argument('--end_user_index', type=int, default=None,
                        help='End user index for processing (default: all users)')
    parser.add_argument('--api_delay', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for GPT API calls (default: 0.8)')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum tokens for GPT response (default: 1000)')
    
    return parser.parse_args()


def load_data(data_name: str) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Load recommendation list, training data, and metadata."""
    
    # Load recommendation list
    rec_list_file = f'dataset/{data_name}/rec_list_all_user.pkl'
    print(f"Loading recommendations from '{rec_list_file}'...")
    try:
        with open(rec_list_file, 'rb') as f:
            rec_list = pickle.load(f)
        print(f"Successfully loaded {len(rec_list)} users from '{rec_list_file}'")
        if not isinstance(rec_list, dict):
            raise ValueError(f"Loaded data is not a dictionary")
    except Exception as e:
        print(f"Error loading pickle file '{rec_list_file}': {e}")
        sys.exit(1)
    
    # Load training data
    train_file = f'dataset/{data_name}/train.txt'
    print(f"Loading training data from '{train_file}'...")
    try:
        train_df = pd.read_csv(train_file, names=['user', 'item', 'rating'], sep=' ')
        print(f"Successfully loaded {len(train_df)} training interactions")
    except Exception as e:
        print(f"Error loading training file '{train_file}': {e}")
        sys.exit(1)
    
    # Load metadata
    if data_name == 'movielen-100k' or data_name == 'movielen-1M':
        meta_file = f'dataset/{data_name}/movies.csv'
        meta_data_dict = {}
        print(f"Loading metadata from '{meta_file}'...")
        try:
            meta_df = pd.read_csv(meta_file, usecols=['movieId', 'title', 'genres'])
            for _, row in meta_df.iterrows():
                meta_data_dict[str(row['movieId'])] = {
                    'title': row['title'],
                    'genres': row['genres'].split('|')
                }
            print(f"Loaded metadata for {len(meta_data_dict)} items")
        except FileNotFoundError:
            print(f"Warning: Metadata file '{meta_file}' not found. Item details will be incomplete.")
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            sys.exit(1)
    elif data_name == 'yelp2018':
        meta_file = f'dataset/{data_name}/meta_{data_name}.json'
        meta_data_dict = {}
        print(f"Loading metadata from '{meta_file}'...")
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        item_meta = json.loads(line)
                        # For Yelp dataset, use 'business_id' as the key
                        if 'business_id' in item_meta:
                            meta_data_dict[str(item_meta['business_id'])] = item_meta
                    except json.JSONDecodeError:
                        print(f"Warning: Line {line_number} in meta file is not valid JSON. Skipping.")
                        pass  # Silently skip malformed JSON lines
            print(f"Loaded metadata for {len(meta_data_dict)} items.")
            if not meta_data_dict:
                print("Warning: No metadata loaded. Item details in prompts will be incomplete.")
        except FileNotFoundError:
            print(f"Error: Metadata file '{meta_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            sys.exit(1)
    else:
        meta_file = f'dataset/{data_name}/meta_{data_name}.json'
        meta_data_dict = {}
        print(f"Loading metadata from '{meta_file}'...")
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item_meta = json.loads(line)
                        if 'asin' in item_meta:
                            meta_data_dict[str(item_meta['asin'])] = item_meta
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded metadata for {len(meta_data_dict)} items")
        except FileNotFoundError:
            print(f"Warning: Metadata file '{meta_file}' not found. Item details will be incomplete.")
        except Exception as e:
            print(f"Error loading metadata file: {e}")
    
    return rec_list, train_df, meta_data_dict


def filter_users_by_quantile(train_df: pd.DataFrame, rec_list: Dict, user_quantile: float) -> Tuple[Dict, int]:
    """Filter users based on interaction count quantile and calculate history count."""
    
    user_interaction_counts = train_df['user'].value_counts().reset_index()
    user_interaction_counts.columns = ['user', 'interaction_count']

    # Calculate the specified percentile of interaction counts
    percentile_threshold = user_interaction_counts['interaction_count'].quantile(user_quantile)
    print(f"The {user_quantile*100}th percentile of user interaction counts is: {percentile_threshold}")
    
    # Filter users whose interaction counts are at or below the percentile
    users_below_percentile = user_interaction_counts[
        user_interaction_counts['interaction_count'] <= percentile_threshold
    ]
    users_to_process = users_below_percentile['user'].unique()
    
    print(f"Number of users with interaction numbers at or below the {user_quantile*100}th percentile: {len(users_to_process)}")
    
    # Convert users_to_process to string to match rec_list keys (which are strings)
    users_to_process_str = set(str(user) for user in users_to_process)
    
    # Filter rec_list to only include these users with proper type conversion
    filtered_rec_list = {
        user: recommendations for user, recommendations in rec_list.items() 
        if str(user) in users_to_process_str
    }
    print(f"Filtered recommendation list contains {len(filtered_rec_list)} users")
    
    # Calculate history count: max(75% user interaction number quantile, 30)
    percentile_75 = user_interaction_counts['interaction_count'].quantile(0.75)
    history_count = max(int(percentile_75 * 0.75), 30)
    print(f"History count set to: {history_count}")
    
    return filtered_rec_list, history_count


def initialize_openai_client(api_key: str) -> OpenAI:
    """Initialize OpenAI client with the provided API key."""
    print("Configuring OpenAI API...")
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI client configured successfully.")
        return client
    except Exception as e:
        print(f"Error configuring OpenAI API: {e}")
        sys.exit(1)


def count_tokens_approximate(text: str) -> int:
    """Approximate token count (rough estimation: 1 token ≈ 4 characters)."""
    return len(text) // 4


def get_item_details_for_prompt(item_asin: str, meta_lookup_dict: Dict) -> Tuple[str, str, str]:
    """Get item title, brand, and category from metadata."""
    item_asin = str(item_asin)
    if item_asin in meta_lookup_dict:
        item_meta = meta_lookup_dict[item_asin]
        title = item_meta.get('title', 'N/A Title')
        brand = str(item_meta.get('brand', 'N/A Brand'))
        
        # Handle different category formats
        if 'category' in item_meta:
            category_str = item_meta.get('category', 'N/A Category')
        elif 'genres' in item_meta:
            genres = item_meta.get('genres', [])
            category_str = ', '.join(genres) if isinstance(genres, list) else str(genres)
        else:
            category_str = 'N/A Category'
            
        return title, brand, category_str
    else:
        return f"Unknown Item (ID: {item_asin})", "N/A Brand", "N/A Category"


def create_system_message() -> str:
    """Create system message for GPT."""
    return """You are a recommendation system expert. Given a user's interaction history and a list of candidate items, your task is to reorder the candidates based on the user's preferences. Analyze the user's past preferences and provide a ranking that puts the most likely preferred items first."""


def create_user_prompt(user_history: str, candidate_list: str, candidate_count: int) -> str:
    """Create user prompt for GPT."""
    template = """USER HISTORY:
{user_history}

The user history shows the items the user has interacted with in chronological order.
The first item is the oldest one the user interacted with, and the last item is the most recent one.
Each entry follows this format: [Position]. [Item Title] (Brand: [Brand], Categories: [Categories]) - Rating: [User Rating]
The user rating ranges from 1.0 to 5.0, with 5.0 being the highest level of enjoyment.

TOP {candidate_count} CANDIDATE ITEM LIST:
{candidate_list}

Each candidate item is listed with an index letter (A, B, C, etc.) followed by the item title, brand, and categories.

Your task is to reorder these candidates based on the user's preferences, where the first item should be the one the user would most likely enjoy, and subsequent items represent decreasing levels of preference.

Please analyze and summarize the user's preferences in paragraph form and write it in <think></think> tags at the beginning of your response.

Finally, provide your recommended ordering of ALL candidate items as a hyphen-separated list of indices (e.g., A-B-C-D-E-F-G-H-I-J-K-L-M-N-O-P-Q-R-S-T) and place it in <output></output> tags.

IMPORTANT: 
- You must include ALL {candidate_count} item indices (A through {last_letter}) in your response
- Use the exact format: <output>A-B-C-D-E-F-G-H-I-J</output>
- The first index should represent the item you believe the user would enjoy most
- Make sure to use hyphens (-) to separate the letters"""

    last_letter = chr(ord('A') + candidate_count - 1)
    return template.format(
        user_history=user_history,
        candidate_count=candidate_count,
        candidate_list=candidate_list,
        last_letter=last_letter
    )


def call_gpt_for_reranking(user_prompt: str, client: OpenAI, model_name: str, 
                          temperature: float, max_tokens: int, candidate_count: int) -> Tuple[str, str, str, int, int]:
    """Call GPT API for re-ranking and parse response."""
    
    system_message = create_system_message()
    
    # Approximate token counts
    input_tokens = count_tokens_approximate(system_message + user_prompt)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_text_response = response.choices[0].message.content
            output_tokens = count_tokens_approximate(raw_text_response)
            
            # Parse thinking and output - handle both <think> and ```think formats
            think_match = re.search(r"<think>(.*?)</think>", raw_text_response, re.DOTALL)
            if not think_match:
                # Try alternative format with ```think
                think_match = re.search(r"```think\s*(.*?)```", raw_text_response, re.DOTALL)
            
            output_match = re.search(r"<output>([A-T](?:-[A-T])*)</output>", raw_text_response, re.DOTALL)
            if not output_match:
                # Try alternative format with ```output
                output_match = re.search(r"```output\s*([A-T](?:-[A-T])*)\s*```", raw_text_response, re.DOTALL)
            if not output_match:
                # Try to find letter sequence pattern anywhere in the response
                output_match = re.search(r"([A-T](?:-[A-T]){" + str(candidate_count-1) + r"})", raw_text_response)
            
            thinking = think_match.group(1).strip() if think_match else "N/A (think)"
            reordered_indices_str = "N/A (output)"
            
            if output_match:
                temp_indices_str = output_match.group(1).strip()
                indices = temp_indices_str.split('-')
                if len(indices) == candidate_count and all(len(idx) == 1 and 'A' <= idx <= chr(ord('A') + candidate_count - 1) for idx in indices):
                    reordered_indices_str = temp_indices_str
                else:
                    print(f"    Warning: Output format incorrect: '{temp_indices_str}' (expected {candidate_count} items).")
                    reordered_indices_str = "FORMAT_ERROR"
            else:
                print(f"    Warning: No output pattern found in response.")
            
            return thinking, reordered_indices_str, raw_text_response, input_tokens, output_tokens
            
        except Exception as e:
            print(f"    ERROR during GPT API call (Attempt {attempt + 1}, Temp: {temperature}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error: {e}", "API_CALL_ERROR", "API_CALL_ERROR", input_tokens, 0


def process_users(rec_list: Dict, train_df: pd.DataFrame, meta_data_dict: Dict, 
                 client: OpenAI, args, history_count: int) -> List[Dict]:
    """Process users for re-ranking."""
    
    # Prepare sorted ratings data
    train_df['user'] = train_df['user'].astype(str)
    train_df['item'] = train_df['item'].astype(str)
    if 'timestamp' in train_df.columns:
        train_df['timestamp'] = pd.to_numeric(train_df['timestamp'], errors='coerce')
        ratings_df_sorted = train_df.sort_values(['user', 'timestamp'], ascending=[True, False])
    else:
        print("Warning: 'timestamp' column not found. Using original order.")
        ratings_df_sorted = train_df
    
    all_reranking_calls = []
    processed_user_count = 0
    
    user_list = list(rec_list.items())
    start_idx = args.start_user_index
    end_idx = args.end_user_index if args.end_user_index is not None else len(user_list) - 1
    
    for user_index, (user_id_str, original_candidate_asins) in enumerate(user_list):
        if user_index < start_idx or user_index > end_idx:
            continue
            
        processed_user_count += 1
        print(f"\nProcessing User: {user_id_str} (Index: {user_index}, Count: {processed_user_count})")
        
        if len(original_candidate_asins) != args.candidate_count:
            print(f"  Warning: User {user_id_str} has {len(original_candidate_asins)} candidates, expected {args.candidate_count}. Skipping.")
            continue
        
        # Get user history using the passed history_count parameter
        user_history_df = ratings_df_sorted[ratings_df_sorted['user'] == str(user_id_str)].head(history_count + 1)
        user_history_df = user_history_df.iloc[1:history_count + 1]
        
        # Format user history
        history_prompt_part = "- (No interaction history found)"
        if not user_history_df.empty:
            history_lines = []
            for i, (_, hist_row) in enumerate(user_history_df.iloc[::-1].iterrows(), 1):
                item_asin_hist = str(hist_row['item'])
                rating_val = hist_row['rating']
                title, brand, category = get_item_details_for_prompt(item_asin_hist, meta_data_dict)
                history_lines.append(
                    f"{i}. {title} (Brand: {brand}, Categories: {category}) - Rating: {rating_val}"
                )
            history_prompt_part = "\n".join(history_lines)
        
        # Format candidate list
        candidate_list_lines = []
        candidate_index_to_asin_map = {}
        index_letters = [chr(ord('A') + i) for i in range(args.candidate_count)]
        
        for i, asin in enumerate(original_candidate_asins):
            letter = index_letters[i]
            candidate_index_to_asin_map[letter] = asin
            title, brand, category = get_item_details_for_prompt(asin, meta_data_dict)
            candidate_list_lines.append(
                f"{letter}. {title}, Brand: {brand}, Categories: {category}"
            )
        candidate_list_prompt = "\n".join(candidate_list_lines)
        
        # Create user prompt
        user_prompt = create_user_prompt(history_prompt_part, candidate_list_prompt, args.candidate_count)
        
        # Call GPT multiple times
        for call_id in range(args.api_calls_per_user):
            print(f"  API Call {call_id + 1}/{args.api_calls_per_user}...")
            
            thinking, reordered_indices_str, raw_response, input_tokens, output_tokens = call_gpt_for_reranking(
                user_prompt, client, args.gpt_model, args.temperature, args.max_tokens, args.candidate_count
            )
            
            # Convert indices to ASINs
            reordered_ids_final = []
            if reordered_indices_str not in ["N/A (output)", "API_CALL_ERROR", "FORMAT_ERROR"]:
                try:
                    indices = reordered_indices_str.split('-')
                    if len(indices) == args.candidate_count and all(idx in candidate_index_to_asin_map for idx in indices):
                        reordered_ids_final = [candidate_index_to_asin_map[idx] for idx in indices]
                except Exception as e:
                    print(f"Error parsing indices: {e}")
            
            all_reranking_calls.append({
                'user_id': user_id_str,
                'original_candidate_ids_str': str(original_candidate_asins),
                'llm_thinking': thinking,
                'llm_reordered_indices_str': reordered_indices_str,
                'reordered_ids_final_str': str(reordered_ids_final),
                'temperature_used': args.temperature,
                'call_id_within_user': call_id + 1,
                'raw_llm_response_snippet': raw_response[:500] + "..." if len(raw_response) > 500 else raw_response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            })
            
            time.sleep(args.api_delay)
        
        # Save periodically
        if processed_user_count % 10 == 0:
            print(f"Auto-saving at user count {processed_user_count}...")
            temp_df = pd.DataFrame(all_reranking_calls)
            temp_df.to_csv(f'dataset/{args.data_name}/gpt_reranked_responses_{start_idx}_{end_idx}.csv', index=False)
    
    return all_reranking_calls


def ensemble_predictions(results_df: pd.DataFrame, weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Create ensemble predictions from multiple API call results."""
    if weights is None:
        weights = [1.0] * 8  # Default equal weights
    
    # Pivot the dataframe
    pivoted_df = results_df.set_index(['user_id', 'call_id_within_user'])['reordered_ids_final_str'].unstack()
    pivoted_df.columns = [f'prediction_{i}' for i in pivoted_df.columns]
    pivoted_df.reset_index(inplace=True)
    
    def custom_blend(row):
        """Blend predictions using weighted ranking."""
        rec_lists = []
        for i in range(len(weights)):
            pred_col = f'prediction_{i+1}'
            if pred_col in row and pd.notna(row[pred_col]):
                # Parse the string representation of list
                pred_str = row[pred_col].replace('[', '').replace(']', '').replace("'", "")
                if pred_str.strip():
                    rec_lists.append([item.strip() for item in pred_str.split(',')])
        
        # Create weighted scores
        item_scores = {}
        for list_idx, rec_list in enumerate(rec_lists):
            weight = weights[list_idx] if list_idx < len(weights) else 1.0
            for pos, item in enumerate(rec_list):
                if item and item != '':
                    if item in item_scores:
                        item_scores[item] += weight / (pos + 1)
                    else:
                        item_scores[item] = weight / (pos + 1)
        
        # Sort by score and return top items
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])
        return ' '.join([item for item, _ in sorted_items])
    
    pivoted_df['prediction'] = pivoted_df.apply(custom_blend, axis=1)
    return pivoted_df


def save_token_summary(results_df: pd.DataFrame, output_file: str):
    """Save token usage summary."""
    total_input_tokens = results_df['input_tokens'].sum()
    total_output_tokens = results_df['output_tokens'].sum()
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Total Input Tokens', 'Total Output Tokens'])
        writer.writerow([total_input_tokens, total_output_tokens])
    
    print(f"Token summary saved to {output_file}")
    print(f"Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}")


def create_augmented_train_file(ensemble_df: pd.DataFrame, data_name: str, default_rating: float = 4.0):
    """Create augmented train file by adding top predicted items to original train.txt."""
    
    # Read original train.txt
    original_train_file = f'dataset/{data_name}/train.txt'
    try:
        original_train_df = pd.read_csv(original_train_file, names=['user', 'item', 'rating'], sep=' ')
        print(f"Loaded original train.txt with {len(original_train_df)} interactions")
    except Exception as e:
        print(f"Error loading original train file '{original_train_file}': {e}")
        return
    
    # Extract top predicted items for each user
    augment_rows = []
    for _, row in ensemble_df.iterrows():
        user_id = row['user_id']
        prediction = row['prediction']
        
        if pd.notna(prediction) and prediction.strip():
            # Get first item from prediction (space-separated)
            top_item = prediction.strip().split()[0]
            augment_rows.append({
                'user': user_id,
                'item': top_item,
                'rating': default_rating
            })
    
    if not augment_rows:
        print("Warning: No valid predictions found to augment train file")
        return
    
    print(f"Adding {len(augment_rows)} new interactions to train file")
    
    # Create augment DataFrame
    augment_df = pd.DataFrame(augment_rows)
    
    # Combine original and augmented data
    augmented_train_df = pd.concat([original_train_df, augment_df], ignore_index=True)
    
    # Save augmented train file
    augmented_train_file = f'dataset/{data_name}/train_augmented.txt'
    augmented_train_df.to_csv(augmented_train_file, sep=' ', header=False, index=False)
    
    print(f"Saved augmented train file to '{augmented_train_file}'")
    print(f"Original: {len(original_train_df)} interactions, Augmented: {len(augmented_train_df)} interactions")
    print(f"Added {len(augment_df)} new interactions with default rating {default_rating}")
    
    return augmented_train_file


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load data
    rec_list, train_df, meta_data_dict = load_data(args.data_name)
    
    # Filter users by quantile
    filtered_rec_list, history_count = filter_users_by_quantile(train_df, rec_list, args.user_quantile)
    print(f"Processing {len(filtered_rec_list)} users with history count: {history_count}")
    
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)
    
    # Process users
    all_reranking_calls = process_users(filtered_rec_list, train_df, meta_data_dict, client, args, history_count)
    
    # Save results
    if all_reranking_calls:
        results_df = pd.DataFrame(all_reranking_calls)
        
        # Save detailed results
        start_idx = args.start_user_index
        end_idx = args.end_user_index if args.end_user_index is not None else len(filtered_rec_list) - 1
        
        results_file = f'dataset/{args.data_name}/gpt_reranked_responses_{start_idx}_{end_idx}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved detailed results to {results_file}")
        
        # Save token summary
        token_file = f'dataset/{args.data_name}/token_summary_gpt_{start_idx}_{end_idx}.csv'
        save_token_summary(results_df, token_file)
        
        # Create ensemble predictions
        ensemble_df = ensemble_predictions(results_df)
        prediction_file = f'dataset/{args.data_name}/gpt_reranked_prediction_{start_idx}_{end_idx}.csv'
        ensemble_df.to_csv(prediction_file, index=False)
        print(f"Saved ensemble predictions to {prediction_file}")
        
        # Create augmented train file
        create_augmented_train_file(ensemble_df, args.data_name)
        
        print(f"\nProcessing complete! Processed {len(results_df) // args.api_calls_per_user} users with {len(results_df)} total API calls.")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
