import torch
import Util
import numpy as np

def bpr_loss(brands_emb_final, 
             pos_influencers_emb_final, 
             neg_influencers_emb_final
             ):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618
    Args:
        - users_emb_final (torch.Tensor): e_u_k
        - pos_items_emb_final (torch.Tensor): positive e_i_k
        - neg_items_emb_final (torch.Tensor): negative e_i_k
    Returns:
        torch.Tensor: scalar bpr loss value
    """
    # bpr loss
    pos_scores = torch.mul(brands_emb_final, pos_influencers_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(brands_emb_final, neg_influencers_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples
    # bpr_loss = - torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))
    bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    loss = bpr_loss
    return bpr_loss

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k
    Args:
        - groundTruth (list[list[long]]): list of lists of item_ids. Cntaining highly rated items of each user. 
                            In other words, this is the list of true_relevant_items for each user
        - r (list[list[boolean]]): list of lists indicating whether each top k item recommended to each user
                            is a top k ground truth (true relevant) item or not
        - k (int): determines the top k items to compute precision and recall on
    Returns:
        tuple: recall @ k, precision @ k
    """

    # number of correctly predicted items per user
    # -1 here means I want to sum at the inner most dimension
    num_correct_pred = torch.sum(r, dim=-1)  
    
    # number of items liked by each user in the test set
    brand_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])
    
    recall = torch.mean(num_correct_pred / brand_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k
    Args:
        - groundTruth (list): list of lists containing highly rated items of each user
        - r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        - k (int): determines the top k items to compute ndcg on
    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# wrapper function to get evaluation metrics
def get_metrics(model, 
                input_edge_index, 
                input_exclude_edge_indices, 
                brand_embedding_final,
                influencers_emb_final,
                k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k
    Args:
        - model (LighGCN): LighGCN model
        - edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        - exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        - k (int): determines the top k items to compute metrics on
    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """     

    # convert adj_mat based edge index to r_mat based edge index so we have have 
    # the first list being user_ids and second list being item_ids for the edge index 
    edge_index = Util.convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index,model.num_brands,model.num_influencers)

    # This is to exclude the edges we have seen before in our predicted interaction matrix (r_mat_rating)
    # E.g: for validation set, we want want to exclude all the edges in training set
    exclude_edge_indices = [Util.convert_adj_mat_edge_index_to_r_mat_edge_index(exclude_edge_index,model.num_brands,model.num_influencers) 
                                      for exclude_edge_index in input_exclude_edge_indices]

    # Generate predicted interaction matrix (r_mat_rating)    
    r_mat_rating = torch.matmul(brand_embedding_final, influencers_emb_final.T)
    
    # shape: num_brands x num_item
    rating = r_mat_rating
   
    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive influencers for each user from the edge index
        # it's a dict: user -> positive item list
        brand_pos_influencers = Util.get_brand_positive_influencers(exclude_edge_index)
        
        # get coordinates of all edges to exclude
        exclude_brands = []
        exclude_influencers = []
        for brand, influencers in brand_pos_influencers.items():
            # [user] * len(item) can give us [user1, user1, user1...] with len of len(item)
            # this makes easier to do the masking below
            exclude_brands.extend([brand] * len(influencers))
            exclude_influencers.extend(influencers)
   
        # set the excluded entry in the rat_mat_rating matrix to a very small number
        rating[exclude_brands, exclude_influencers] = -(1 << 10) 

    # get the top k recommended influencers for each user
    _, top_K_influencers = torch.topk(rating, k=k)

    # get all unique brands in evaluated split
    brands = edge_index[0].unique()

    # dict of user -> pos_item list
    test_brand_pos_influencers = Util.get_brand_positive_influencers(edge_index)

    # convert test user pos influencers dictionary into a list of lists
    test_brand_pos_influencers_list = [test_brand_pos_influencers[brand.item()] for brand in brands]

    # r here is "pred_relevant_influencers ∩ actually_relevant_influencers" list for each user
    r = []
    for brand in brands:
        brand_true_relevant_influencer = test_brand_pos_influencers[brand.item()]
        # list of Booleans to store whether or not a given item in the top_K_influencers for a given user 
        # is also present in user_true_relevant_item.
        # this is later on used to compute n_rel_and_rec_k
        label = list(map(lambda x: x in brand_true_relevant_influencer, top_K_influencers[brand]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_brand_pos_influencers_list, r, k)
    ndcg = NDCGatK_r(test_brand_pos_influencers_list, r, k)

    return recall, precision, ndcg

# wrapper function to evaluate model
def evaluation(model, 
               edge_index, # adj_mat based edge index
               exclude_edge_indices,  # adj_mat based exclude edge index
               num_brands, 
               num_influencers,
               k, 
               x
              ):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        num_brands (int): number of brands
        num_influencers (int): number of influencers
        k (int): determines the top k influencers to compute metrics on
        x (torch.Tensor): input data

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    brands_emb_final, influencers_emb_final = model.forward(x, edge_index)
    r_mat_edge_index = Util.convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index,num_brands, num_influencers)
    edges = Util.structured_negative_sampling(r_mat_edge_index, contains_neg_self_loops=False)
    brand_indices, pos_influencer_indices, neg_influencer_indices = edges[0], edges[1], edges[2]
    brands_emb_final = brands_emb_final[brand_indices]
    pos_influencers_emb_final = influencers_emb_final[pos_influencer_indices]
    neg_influencers_emb_final = influencers_emb_final[neg_influencer_indices]

    loss = bpr_loss(brands_emb_final, 
                    pos_influencers_emb_final, 
                    neg_influencers_emb_final).item()

    recall, precision, ndcg = get_metrics(model, 
                                          edge_index, 
                                          exclude_edge_indices,
                                          brands_emb_final,
                                          influencers_emb_final,
                                          k)

    return loss, recall, precision, ndcg

def get_embs_for_bpr(model, input_edge_index, x, BATCH_SIZE, num_brands , num_influencers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    brands_emb_final, influencers_emb_final = model.forward(x,input_edge_index)
    edge_index_to_use = Util.convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, num_brands , num_influencers)
    
    # mini batching for eval and calculate loss 
    brand_indices, pos_influencer_indices, neg_influencer_indices = Util.sample_mini_batch(BATCH_SIZE, edge_index_to_use)
    
    # This is to push tensor to device so if we are using GPU
    brand_indices, pos_influencer_indices, neg_influencer_indices = brand_indices.to(device), pos_influencer_indices.to(device), neg_influencer_indices.to(device)
 
    # we need layer0 embeddings and the final embeddings (computed from 0...K layer) for BPR loss computing
    brands_emb_final= brands_emb_final[brand_indices]
    pos_influencers_emb_final = influencers_emb_final[pos_influencer_indices]
    neg_influencers_emb_final = influencers_emb_final[neg_influencer_indices]
   
    return brands_emb_final, pos_influencers_emb_final, neg_influencers_emb_final