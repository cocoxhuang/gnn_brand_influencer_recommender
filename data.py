import dgl
import torch
import Util
from torch import nn
from sklearn.model_selection import train_test_split

def get_graph(n_components_vis = 128, n_components_word = 16):
    g = dgl.load_graphs("graph.bin")[0][0]

    # Reduce the dimensions of graph vision PCA embbeddings
    vis_emb  = g.ndata['vis_emb']
    pca_vis_emb = vis_emb[:,: n_components_vis]
    pca_vis_emb = nn.init.normal_(pca_vis_emb,std = 0.15)
    print(f'pca_vis_emb has dim {pca_vis_emb.size()}')
    g.ndata['pca_vis'] = pca_vis_emb

    # Reduce the dimensions of graph word PCA embbeddings
    word_emb  = g.ndata['word_emb']
    pca_word_emb = word_emb[:,: n_components_word]
    pca_word_emb = nn.init.normal_(pca_word_emb,std = 0.15)
    print(f'pca_word_emb has dim {pca_word_emb.size()}')
    g.ndata['pca_word'] = word_emb

    return g

def get_x(g, input_g_ndata = ['follower','following','category','pca_vis']):
    x = torch.tensor([],dtype=torch.float)
    for data in input_g_ndata:
        if x.size()[0] > 0:
            x = torch.concat([x,g.ndata[data]],dim = 1)
        else:
            x = g.ndata[data]
    x -= x.mean(dim=0)
    x /= x.std(dim=0)
    return x

def get_train_val_test_edges(edge_index, num_interactions, num_brands, num_influencers, train_size = 0.8, tes_size = 0.1, device = 'cpu'):
    # split the edges of the graph using a 80/10/10 train/validation/test split
    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices, test_size=1-train_size, random_state=1)
    val_indices, test_indices = train_test_split(test_indices, test_size=tes_size/(1-train_size), random_state=1)
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    print(f"Train edges size: {train_edge_index.size()[1]}")
    print(f"Val edges size: {val_edge_index.size()[1]}")
    print(f"Test edges size: {test_edge_index.size()[1]}")
    print(f'Total num of unique brands in train: {torch.unique(train_edge_index[0]).size()[0]}')

    # convert edge indices to adj matrices to be in put in the model
    train_edge_index = Util.convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, num_brands, num_influencers)
    val_edge_index = Util.convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, num_brands, num_influencers)
    test_edge_index = Util.convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, num_brands, num_influencers)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)

    return train_edge_index,val_edge_index,test_edge_index