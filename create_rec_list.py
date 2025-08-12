#!/usr/bin/env python3
"""
Create Recommendation List for Dataset

This script processes a dataset, splits it into train/test sets, trains a recommendation model,
and generates recommendation candidates for users.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.conf import ModelConf
from data.loader import FileIO
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.algorithm import find_k_largest


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create recommendation list for dataset')
    
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset folder (e.g., movielen-100k, amazon-scientific)')
    parser.add_argument('--model_name', type=str, default='LightGCN',
                        help='Recommendation model to use (default: LightGCN)')
    parser.add_argument('--num_candidates', type=int, default=10,
                        help='Number of recommendation candidates per user (default: 10)')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Maximum training epochs (default: 50)')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Embedding size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size (default: 2048)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--reg_lambda', type=float, default=0.0001,
                        help='Regularization lambda (default: 0.0001)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GCN layers for LightGCN (default: 2)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training if model already exists')
    
    return parser.parse_args()


def get_data_paths(data_name: str):
    """Get file paths for the given dataset."""
    base_path = f'dataset/{data_name}'
    
    # Different datasets have different file structures
    if data_name == 'movielen-100k' or data_name == 'movielen-1M':
        return {
            'ratings': f'{base_path}/ratings.csv',
            'train': f'{base_path}/train.txt',
            'test': f'{base_path}/test.txt',
            'config': f'{base_path}/config.yaml',
            'rec_list': f'{base_path}/rec_list_all_user.pkl',
            'movies': f'{base_path}/movies.csv'  # metadata
        }
    else:
        # For other datasets like amazon-scientific
        return {
            'ratings': f'{base_path}/ratings.csv',  # or ratings.txt
            'train': f'{base_path}/train.txt',
            'test': f'{base_path}/test.txt',
            'config': f'{base_path}/config.yaml',
            'rec_list': f'{base_path}/rec_list_all_user.pkl',
            'metadata': f'{base_path}/meta_{data_name}.json'  # metadata
        }


def load_and_split_data(data_paths: dict, args):
    """Load and split data into train/test sets."""
    # Check if ratings file exists
    ratings_file = data_paths['ratings']
    if not os.path.exists(ratings_file):
        # Try alternative file extensions
        alt_ratings = ratings_file.replace('.csv', '.txt')
        if os.path.exists(alt_ratings):
            ratings_file = alt_ratings
        else:
            print(f"Error: Ratings file not found at {ratings_file} or {alt_ratings}")
            sys.exit(1)
    
    print(f"Loading data from '{ratings_file}'...")
    
    # Load data based on file format
    if ratings_file.endswith('.csv'):
        df = pd.read_csv(ratings_file)
        # Standardize column names
        if 'userId' in df.columns and 'movieId' in df.columns:
            df = df.rename(columns={'userId': 'user', 'movieId': 'item'})
        elif 'user_id' in df.columns and 'item_id' in df.columns:
            df = df.rename(columns={'user_id': 'user', 'item_id': 'item'})
    else:
        # Assume space-separated format
        df = pd.read_csv(ratings_file, names=['user', 'item', 'rating'], sep=' ')
    
    print(f"Loaded {len(df)} interactions for {df['user'].nunique()} users and {df['item'].nunique()} items")
    
    # Check if train/test files already exist
    if os.path.exists(data_paths['train']) and os.path.exists(data_paths['test']):
        print("Train/test files already exist. Skipping data splitting.")
        return
    
    print("Sorting data by user and timestamp...")
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values(['user', 'timestamp'], ascending=[True, True])
    else:
        print("Warning: No timestamp column found. Using original order.")
        df_sorted = df.sort_values(['user'], ascending=[True])
    
    print("Creating test set (last item per user)...")
    test = df_sorted.groupby('user').tail(1)
    
    print("Creating training set (all but last item per user)...")
    train = df_sorted.groupby('user').apply(lambda x: x.head(-1)).reset_index(drop=True)
    
    train_to_save = train[['user', 'item', 'rating']]
    test_to_save = test[['user', 'item', 'rating']]
    
    print(f"\nNumber of ratings in training set: {len(train_to_save)}")
    print(f"Number of ratings in test set: {len(test_to_save)}")
    print(f"Total ratings processed: {len(train_to_save) + len(test_to_save)}")
    
    unique_users_original = df['user'].nunique()
    unique_users_test = test_to_save['user'].nunique()
    print(f"Number of unique users in original data: {unique_users_original}")
    print(f"Number of unique users in test set: {unique_users_test}")
    
    if unique_users_original != unique_users_test:
        print("Warning: Number of unique users in test set does not match original.")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(data_paths['train']), exist_ok=True)
    os.makedirs(os.path.dirname(data_paths['test']), exist_ok=True)
    
    print(f"\nSaving training data to '{data_paths['train']}'...")
    train_to_save.to_csv(data_paths['train'], sep=' ', index=False, header=False)
    print("Training data saved.")
    
    print(f"Saving test data to '{data_paths['test']}'...")
    test_to_save.to_csv(data_paths['test'], sep=' ', index=False, header=False)
    print("Test data saved.")
    
    print("\nLeave-last-out splitting complete.")


def create_config_file(data_paths: dict, args):
    """Create configuration file for the model."""
    config_data = {
        "training.set": data_paths['train'],
        "test.set": data_paths['test'],
        "model": {
            "name": args.model_name,
            "type": "graph"
        },
        "item.ranking.topN": [10, 20],
        "embedding.size": args.embedding_size,
        "max.epoch": args.max_epoch,
        "batch.size": args.batch_size,
        "learning.rate": args.learning_rate,
        "reg.lambda": args.reg_lambda,
        f"{args.model_name}": {
            "n_layer": args.n_layers
        },
        "output": "./results/"
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(data_paths['config']), exist_ok=True)
    
    # Save configuration as YAML file
    with open(data_paths['config'], "w") as file:
        yaml.dump(config_data, file, default_flow_style=False)
    
    print(f"Configuration saved to '{data_paths['config']}'")
    return config_data


# Split data
# df = pd.read_csv(RATINGS_FILE_PATH)



# Training classes
class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model']['type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model']['type'])

        self.kwargs = {}
        print('Reading data and preprocessing...')

    def execute(self):
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        return eval(recommender)


# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCN']
        self.n_layers = int(args['n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        # Initialize best embeddings
        with torch.no_grad():
            self.user_emb, self.item_emb = model()
            self.best_user_emb, self.best_item_emb = self.user_emb, self.item_emb
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        
        # Ensure final embeddings are set
        with torch.no_grad():
            self.user_emb, self.item_emb = model()
            if hasattr(self, 'best_user_emb') and hasattr(self, 'best_item_emb'):
                self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


# Get list of user and their candidate items to perform data augment 
def test_and_generate_candidates(rec, args, data_paths):
    """Generate recommendation candidates for all users."""
    def process_bar(num, total):
        rate = float(num) / total
        ratenum = int(50 * rate)
        print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

    rec_list = {}
    data_train = pd.DataFrame(rec.data.training_data, columns=['uid', 'iid', 'rating'])
    
    # Get all users instead of filtering by degree
    all_users = data_train['uid'].unique()
    user_count = len(all_users)
    print(f"Processing {user_count} users (generating {args.num_candidates} candidates per user)")
    
    for i, user in enumerate(all_users):
        candidates = rec.predict(user)
        rated_list, _ = rec.data.user_rated(user)
        for item in rated_list:
            candidates[rec.data.item[item]] = -10e8

        ids, scores = find_k_largest(args.num_candidates, candidates)

        rec_list[user] = []
        for id, score in zip(ids, scores):
            rec_list[user].append(rec.data.id2item[id])
        if i % 1000 == 0:
            process_bar(i, user_count)
    process_bar(user_count, user_count)
    print('')
    return rec_list


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"Processing dataset: {args.data_name}")
    print(f"Model: {args.model_name}")
    print(f"Number of candidates: {args.num_candidates}")
    
    # Get data paths
    data_paths = get_data_paths(args.data_name)
    
    # Load and split data
    load_and_split_data(data_paths, args)
    
    # Create config file
    config_data = create_config_file(data_paths, args)
    
    # Training
    print("\n--- Starting Model Training ---")
    conf = ModelConf(data_paths['config'])
    rec = SELFRec(conf).execute()
    
    if not args.skip_training:
        print("Training model...")
        rec.train()
        print("Training completed!")
    else:
        print("Skipping training as requested")
        # Still need to initialize embeddings even if skipping training
        # Move model to CUDA first
        rec.model = rec.model.cuda()
        with torch.no_grad():
            rec.user_emb, rec.item_emb = rec.model()
    
    # Generate recommendation candidates
    print("\n--- Generating Recommendation Candidates ---")
    rec_list = test_and_generate_candidates(rec, args, data_paths)
    
    # Save recommendation list
    print(f"\n--- Saving Results ---")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_paths['rec_list']), exist_ok=True)
        
        with open(data_paths['rec_list'], 'wb') as f:
            pickle.dump(rec_list, f)
        print(f"Successfully saved {len(rec_list)} user recommendations to '{data_paths['rec_list']}'")
    except Exception as e:
        print(f"Error saving recommendations to pickle: {e}")
        sys.exit(1)
    
    print(f"\nProcess complete! Generated recommendations for {len(rec_list)} users.")


if __name__ == "__main__":
    main()
