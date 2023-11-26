from sklearn.decomposition import PCA
from sklearn import preprocessing
import torch
from torch import nn
from torch_sparse import SparseTensor
from torch_geometric.utils import structured_negative_sampling
import random



# convert from r_mat (interaction matrix) edge index to adjescency matrix's edge index
def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, num_brands, num_influencers):
    R = torch.zeros((num_brands, num_influencers))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((num_brands + num_influencers , num_brands + num_influencers))
    adj_mat[: num_brands, num_brands :] = R.clone()
    adj_mat[num_brands :, : num_brands] = R_transpose.clone()
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo

# convert from adjescency matrix's edge index to r_mat (interaction matrix) edge index
def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index,num_brands, num_influencers):
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0], 
                                           col=input_edge_index[1], 
                                           sparse_sizes=((num_brands + num_influencers), num_brands + num_influencers))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_brands, num_brands :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index

# Random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    """
    Generates a mini batch
    Args:
        - batch_size (int): batch size
        - edge_index: 
    Returns:
        brand_indices (list): indices of brands in the batch
        pos_influencer_indices (list): indices of postive influencer in the batch
        neg_influencer_indices (list)L indices of negtive influencer in the batch
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    brand_indices, pos_influencer_indices, neg_influencer_indices = batch[0], batch[1], batch[2]
    return brand_indices, pos_influencer_indices, neg_influencer_indices

def get_brand_positive_influencers(edge_index):
    """
    Generates dictionary of positive items for each user
    Args:
        - edge_index (torch.Tensor): 2 by N list of edges 
    Returns:
        dict: {user : list of positive items for each}
    """
    brand_positive_influencers = {}
    for i in range(edge_index.shape[1]):
        brand = edge_index[0][i].item()
        influencer = edge_index[1][i].item()
        if brand not in brand_positive_influencers:
            brand_positive_influencers[brand] = []
        brand_positive_influencers[brand].append(influencer)
    return brand_positive_influencers


def dim_reduct_pca(raw_emb, n_components):
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(raw_emb)
    pca_emb = torch.tensor(pca_emb, dtype = torch.float32) 
    pca_emb = nn.init.normal_(pca_emb,std = 0.15)
    pca_emb = preprocessing.normalize(pca_emb)
    pca_emb = torch.tensor(pca_emb, dtype = torch.float32)
    print(f"PCA offers explained variance ratio: {pca.explained_variance_ratio_.sum()}")   
    print(f'Variance explained ratio by dimension: {pca.explained_variance_ratio_}') 
    return pca_emb

