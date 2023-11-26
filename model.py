# defines LightGCN model 
from torch_geometric.nn.conv import MessagePassing
from typing import Any, Dict, List, Optional, Union
from torch_geometric.nn.aggr import Aggregation
import torch
from torch import nn, Tensor
from torch import nn, Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm



class LightGCN(MessagePassing):

    def __init__(self, num_brands, 
                 num_influencers, 
                 in_channels,
                 out_channels,
                 K=2, 
                 add_self_loops=False):
        super(LightGCN,self).__init__()
        self.num_brands = num_brands
        self.num_influencers = num_influencers
        self.lin = nn.Linear(in_channels,out_channels)
        self.K = K
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index: Tensor):
        edge_index_norm = gcn_norm(edge_index=edge_index, 
                                   add_self_loops=self.add_self_loops)
        emb_0 = self.lin(x)
        # brands_emb_0, influencers_emb_0 = torch.split(emb_0, [self.num_brands, self.num_influencers]) 
        embs = [emb_0] 
        emb_k = emb_0

        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)   
        emb_final = torch.mean(embs, dim=1) 
        brands_emb_final, influencers_emb_final = torch.split(emb_final, [self.num_brands, self.num_influencers]) 
        return brands_emb_final, influencers_emb_final

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j