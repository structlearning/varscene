import pickle
import networkx as nx
from utils import batch_sample_graphs, get_node_star, plot, get_label_embeddings, get_star_embedding, get_star_dict,\
    build_model, batch_sample_graphs_from_z, nx_to_json, graph_in_vocab, pack_batch, get_graph, sample_graphs
import time
import torch
import importlib
from collections import Counter
import os
import numpy as np
import random
import argparse

## argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="directory containing model", required=True)
parser.add_argument("--model_path", type=str, help="path of model", required=True)
parser.add_argument("-n", '--n_trials', type=int, default=50)
parser.add_argument("-c", '--cutoff_size', type=int, default=50)
args = parser.parse_args()

device = torch.device('cuda:0')
data_dir = '../svg_data/data/'
model_dir = args.model_dir
config_file = (os.path.join(model_dir, 'configure')).replace('/','.')
config = importlib.import_module(config_file).get_default_config()

# Set random seeds
seed = config['seed']
np.random.seed(23)
random.seed(23)
torch.cuda.manual_seed(42)
torch.manual_seed(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(data_dir, 'dir_graphs_train.pkl'), 'rb') as f:
    training_set = pickle.load(f)
    training_set = [nx.convert_node_labels_to_integers(g) for g in training_set]
with open(os.path.join(data_dir, 'dir_graphs_val.pkl'), 'rb') as f:
    validation_set = pickle.load(f)
    validation_set = [nx.convert_node_labels_to_integers(g) for g in validation_set]

star_dict = get_star_dict(training_set + validation_set)

node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set+validation_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

star_embeddings = get_star_embedding(star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)
## initialize model and optimizer
model, _ = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

model_dir = args.model_dir
model_file_name = args.model_path
model.load_state_dict(torch.load(os.path.join(model_dir, model_file_name), map_location=device))
model.eval()

sampled_gphs, util_f_star, first_star_list = list(), list(), list()
batch_size = 500
graphs_to_sample = 15000 #1000
t1 = time.time()
with torch.no_grad():
    for i in range(0, 15000, batch_size):

        if i >= len(validation_set):
            break

        t2 = time.time()

        batch = pack_batch(validation_set[i : i+batch_size], node_label_embeddings, edge_label_embeddings,
                            star_embeddings, star_dict)
        first_star_list.extend(batch.graph_first_star)

        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, _, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        _, z_list = model(node_features.to(device),
        				edge_features.to(device), star_features.to(device),
        				candidate_star_features.to(device), target_star_idx.to(device),
        				star_z_mask.to(device), candidate_star_z_mask.to(device),
        				from_idx.to(device), to_idx.to(device), graph_idx.to(device), 1+int(torch.max(graph_idx)),
        				0, graph_depth_range.to(device), node_graph_depth_idx.to(device), True)
        z_list = [o.to(device) for o in z_list]

        gphs, _, _, f_star = batch_sample_graphs_from_z(model, z_list,
                                        batch.graph_first_star, 30,
                                		1, star_dict, star_embeddings, 20, False)
        sampled_gphs.extend([nx.DiGraph(nx.convert_node_labels_to_integers(o)) for o in gphs])
        util_f_star.extend(f_star)
        print('sampled %s/%s gphs in %ds, total %s' % (len(gphs), len(batch.graph_first_star), time.time()-t2, len(sampled_gphs)))

with open(os.path.join(model_dir, 'cond_%s_generated.pkl' % (model_file_name.replace('.pt', ''))), 'wb') as f:
    pickle.dump(sampled_gphs, f)

print('time %.2fm, %s graphs generated' % ((time.time()-t1)/60, len(sampled_gphs)))
