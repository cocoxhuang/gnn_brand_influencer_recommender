import torch
import Util
import data
from model import LightGCN
from torch import optim
from tqdm.notebook import tqdm
import metric

# define contants
ITERATIONS = 2000
EPOCHS = 10
BATCH_SIZE = 1000
LR = 0.01
ITERS_PER_EVAL = 10
ITERS_PER_LR_DECAY = 10
K = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get data
g= data.get_graph()
num_influencers = int(g.ndata["if_influencer"].sum())
num_brands = g.num_nodes() - num_influencers

print(f'Graph has ndata: {g.ndata.keys()}')
edge_index = Util.convert_adj_mat_edge_index_to_r_mat_edge_index(g.edges(),num_brands, num_influencers)
edge_index = torch.LongTensor(edge_index) 
num_interactions = edge_index.shape[1]
print(f"We have num_brand: {num_brands}, num_influencers: {num_influencers}, num_edges: {num_interactions}")
print(f'We have node data: {g.ndata.keys()}')

# get x and train/val/test edges
x = data.get_x(g)
input_dim = x.size()[1]
train_edge_index,val_edge_index,test_edge_index = data.get_train_val_test_edges(edge_index, num_interactions, num_brands, num_influencers)

# setup model
layers = 3
model = LightGCN(num_brands = num_brands, 
                 num_influencers = num_influencers, 
                 in_channels = input_dim,
                 out_channels = 32,            
                 K = layers)
print(f"Using device {device}.")
model = model.to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# training loop
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

train_losses = []
val_losses = []
val_recall_at_ks = []
for iter in tqdm(range(ITERATIONS)):
    # training set
    # forward propagation  
    brands_emb_final, pos_influencers_emb_final, neg_influencers_emb_final = metric.get_embs_for_bpr(model, train_edge_index, x, BATCH_SIZE, num_brands , num_influencers)
    # loss computation
    train_loss = metric.bpr_loss(brands_emb_final, 
                          pos_influencers_emb_final, 
                          neg_influencers_emb_final, 
                          )
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # validation set
    if iter % ITERS_PER_EVAL == 0:
        model.eval()
        with torch.no_grad():
            val_loss, recall, precision, ndcg = metric.evaluation(model, 
                                                           val_edge_index, 
                                                           [train_edge_index], 
                                                           num_brands, 
                                                            num_influencers,
                                                           K, 
                                                           x
                                                          )
            print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 1)}, val_loss: {round(val_loss, 1)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")

            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            val_recall_at_ks.append(round(recall, 5))
        model.train()

    if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
        scheduler.step()