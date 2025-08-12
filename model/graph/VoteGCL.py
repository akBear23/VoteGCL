import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.ui_graph import Interaction

class VoteGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(VoteGCL, self).__init__(conf, training_set, test_set)
        args = self.config['VoteGCL']
        self.n_layers = int(args['n_layer'])
        self.cl_rate = float(args['lambda'])  # contrastive learning weight
        self.temp = float(args['temp'])       # temperature parameter for InfoNCE
                
        # Setup augmented data graph if provided
        if 'augmented_set' in kwargs:
            self.aug_data = Interaction(conf, kwargs['augmented_set'], test_set)
            self.has_augmented = True
        else:
            # If no augmented data provided, raise error since we need augmented data
            raise ValueError("VoteGCL requires augmented data. Please provide 'augmented_set' in kwargs.")
            
        # Initialize model
        self.model = VoteGCL_Encoder(self.data, self.aug_data, self.emb_size, self.n_layers, self.temp)

    def build(self):
        # Best performance tracker for evaluation
        self.bestPerformance = []

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.aug_data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                
                # Get embeddings from original data
                rec_user_emb, rec_item_emb = model(self.model.aug_sparse_norm_adj)  # Uses original adjacency matrix by default
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                
                # Calculate recommendation loss (BPR loss) from original graph
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                # Calculate L2 regularization loss
                reg_loss = l2_reg_loss(self.reg, 
                                      model.embedding_dict['user_emb'][user_idx],
                                      model.embedding_dict['item_emb'][pos_idx],
                                      model.embedding_dict['item_emb'][neg_idx]) / self.batch_size
                
                # Total loss
                batch_loss = rec_loss + reg_loss 
                
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                
                # Get embeddings from original data
                rec_user_emb, rec_item_emb = model()  # Uses original adjacency matrix by default
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                
                # Calculate recommendation loss (BPR loss) from original graph
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                # Calculate L2 regularization loss
                reg_loss = l2_reg_loss(self.reg, 
                                      model.embedding_dict['user_emb'][user_idx],
                                      model.embedding_dict['item_emb'][pos_idx],
                                      model.embedding_dict['item_emb'][neg_idx]) / self.batch_size
                
                # Calculate contrastive loss between original graph and augmented graph
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx])
                
                # Total loss
                batch_loss = rec_loss + reg_loss + cl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Log progress
                if n % 100 == 0 and n > 0:
                    print(f'Training: Epoch {epoch + 1}, Batch {n}, '
                          f'Rec Loss: {rec_loss.item():.4f}, CL Loss: {cl_loss.item():.4f}, '
                          f'Total Loss: {batch_loss.item():.4f}')

            # Store embeddings for evaluation
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                
            # Evaluate model periodically
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
                
        # After training, use the best embeddings (saved by fast_evaluation)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        """
        Make predictions for user u using the original graph embeddings
        """
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class VoteGCL_Encoder(nn.Module):
    def __init__(self, orig_data, aug_data, emb_size, n_layers, temp):
        super(VoteGCL_Encoder, self).__init__()
        self.orig_data = orig_data
        self.aug_data = aug_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.temp = temp
        
        # Initialize embeddings - shared between both original and augmented graphs
        self.embedding_dict = self._init_model()
        
        # Prepare normalized adjacency matrices
        self.orig_norm_adj = orig_data.norm_adj
        self.aug_norm_adj = aug_data.norm_adj
        
        # Convert to sparse tensors
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.orig_norm_adj).cuda()
        self.aug_sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.aug_norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.orig_data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.orig_data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self, perturbed_adj=None):
        """
        Forward pass using specified adjacency matrix
        If perturbed_adj is None, use original adjacency matrix
        """
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        
        for k in range(self.layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings = all_embeddings[:self.orig_data.user_num]
        item_all_embeddings = all_embeddings[self.orig_data.user_num:]
        
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx):
        """
        Calculate contrastive loss between original and augmented graph embeddings
        idx: list containing [user_idx, item_idx]
        """
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        
        # Get embeddings from both graphs
        user_view_orig, item_view_orig = self.forward()  # Use original adjacency matrix
        user_view_aug, item_view_aug = self.forward(self.aug_sparse_norm_adj)  # Use augmented adjacency matrix
        
        # Concat user and item embeddings for contrastive learning
        view_orig = torch.cat((user_view_orig[u_idx], item_view_orig[i_idx]), 0)
        view_aug = torch.cat((user_view_aug[u_idx], item_view_aug[i_idx]), 0)
        
        # Calculate InfoNCE loss
        return InfoNCE(view_orig, view_aug, self.temp)