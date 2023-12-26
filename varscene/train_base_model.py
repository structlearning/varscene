import pickle
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os
import torch
from configure import get_default_config
import argparse
import networkx as nx
from utils import build_model, get_graph, get_star_embedding, pack_batch, \
    get_label_embeddings, get_star_dict

total_run_time = time.time()
## argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, help="directory to store model outputs", required=True)
parser.add_argument("--dataset", type=str, choices=['vg', 'vrd', 'svg'], help="dataset name", required=True)
parser.add_argument("--load_checkpoint", action='store_true', help="load model from checkpoint")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=1000)
parser.add_argument("--batch_size", type=int, help="batch size for training", default=1024)

args = parser.parse_args()
out_dir = args.out_dir
dataset = args.dataset

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
os.system('cp configure.py %s' % (os.path.join(out_dir, 'configure.py')))

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

#%%
t1 = time.time()
## load graphs
if dataset == 'vg': ## visual genome
    train_data_dir = '../vg_data/data'
elif dataset == 'vrd': ## visual relationship detection
    train_data_dir = '../vrd_data/data'
elif dataset == 'svg': ## small-sized visual genome
    train_data_dir = '../svg_data/data'
else:
    raise ValueError('invalid dataset %s, exiting...' % dataset)

with open(os.path.join(train_data_dir, 'graphs_train.pkl'), 'rb') as f:
    training_set = pickle.load(f)

print('loaded graphs in %.2fs' % (time.time()-t1))

training_set = [nx.convert_node_labels_to_integers(g) for g in training_set]

t1 = time.time()
node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set)
training_star_dict = get_star_dict(training_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

training_star_embeddings = get_star_embedding(training_star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)

print('computed stars and embeddings in %.2fs' % (time.time()-t1))

## initialize model and optimizer
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

train_epochs, train_loss_hist = [], []
start_epoch = 0

model_file = os.path.join(out_dir, ('star_vae_%s.pt' % dataset))
optimizer_file = os.path.join(out_dir, ('optimizer_%s.pt' % dataset))
loss_hist_file = os.path.join(out_dir, 'loss_hist.pkl')
if args.load_checkpoint:
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
    if os.path.isfile(optimizer_file):
        optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))
    if os.path.isfile(loss_hist_file):
        with open(loss_hist_file, 'rb') as f:
            (train_epochs, train_loss_hist) = pickle.load(f)
            start_epoch = len(train_epochs)

model_file = os.path.join(out_dir, ('star_vae_%s_%s.pt' % (dataset, start_epoch+args.epochs)))
optimizer_file = os.path.join(out_dir, ('optimizer_%s_%s.pt' % (dataset, start_epoch+args.epochs)))
loss_hist_file = os.path.join(out_dir, 'loss_hist_%s.pkl' % (start_epoch+args.epochs))

batch_size = args.batch_size
n_training_steps = args.epochs

## prepare batches
t1 = time.time()

print('starting training...')
for i_iter in range(start_epoch, start_epoch+n_training_steps):
    train_running_loss = 0
    t1 = time.time()

    # for batch in batch_data:
    for i_batch in range(0, len(training_set), batch_size):
        batch = pack_batch(training_set[i_batch : i_batch+batch_size],
                        node_label_embeddings, edge_label_embeddings,
                        training_star_embeddings, training_star_dict)
    
        model.train(mode=True)
        optimizer.zero_grad()
        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        # t2 = time.time()
        loss = model(node_features.to(device), edge_features.to(device), star_features.to(device),
            candidate_star_features.to(device), target_star_idx.to(device),
            star_z_mask.to(device), candidate_star_z_mask.to(device),
            from_idx.to(device), to_idx.to(device), graph_idx.to(device), batch_size, 0.1,
            graph_depth_range.to(device), node_graph_depth_idx.to(device), False)

        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
        # print('%.4fs for batch update' % (time.time()-t2))
        
    train_running_loss /= len(training_set)
    train_loss_hist.append(train_running_loss)
    train_epochs.append(i_iter+1)

    ## save model
    if i_iter % config['training']['save_model_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
    
    print('epoch %s, train loss %.6f, time %.2fs' %(i_iter+1, train_running_loss, time.time()-t1))

    if i_iter % config['training']['save_loss_hist_after'] == 0 or \
        i_iter+1 == n_training_steps+start_epoch:
        with open(loss_hist_file, 'wb') as f:
            pickle.dump((train_epochs, train_loss_hist), f)
        
        plt.figure()
        plt.plot(train_epochs, train_loss_hist, label='train loss')
        plt.legend()
        plt.title('Loss history')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(loss_hist_file.replace('pkl', 'png'), bbox_inches='tight')
        plt.close()


print('total run time %.2fm' % ((time.time()-total_run_time)/60))