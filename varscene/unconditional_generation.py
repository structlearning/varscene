import pickle
import numpy as np
import time
import random
import os
import argparse
import torch
import importlib
import networkx as nx
from utils import batch_sample_graphs, build_model, get_graph, get_label_embeddings, get_node_star, get_star_dict, get_star_embedding, \
    batch_sample_graphs_from_z, pack_batch

## argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="directory containing model", required=True)
parser.add_argument('--dataset', type=str, choices=['vrd', 'vg', 'svg'], help='dataset name', required=True)
parser.add_argument("-n", '--n_trials', type=int, default=50)
parser.add_argument("-c", '--cutoff_size', type=int, default=50)
parser.add_argument('-k', '--kernel', choices=['star_node_edge', 'sp_node_edge', 'node_edge', 'star', 'star_sim', 'node_bi', 'edge_bi', 'nspd', 'wl', 'sp'], type=str, default='star')

args = parser.parse_args()
model_dir = args.model_dir
dataset = args.dataset

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

config_file = (os.path.join(model_dir, 'configure')).replace('/','.')
config = importlib.import_module(config_file).get_default_config()

# Set random seeds
seed = config['seed']
random.seed(seed + 2)
np.random.seed(seed + 3)
torch.manual_seed(seed + 4)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

## load graphs
if dataset == 'vrd':
    data_dir = '../vrd_data/data/'
elif dataset == 'vg':
    data_dir = '../vg_data/data/'
elif dataset == 'svg':
    data_dir = '../svg_data/data/'

t1 = time.time()
with open(os.path.join(data_dir, 'graphs_train.pkl'), 'rb') as f:
    training_set = pickle.load(f)
with open(os.path.join(data_dir, 'graphs_val.pkl'), 'rb') as f:
    validation_set = pickle.load(f)

training_set = [nx.convert_node_labels_to_integers(g) for g in training_set]
validation_set = [nx.convert_node_labels_to_integers(g) for g in validation_set]

print('data loaded in %.2fs' % (time.time()-t1))

t1 = time.time()
node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set+validation_set)
print('loaded sentence embedding in %.2fs' % (time.time()-t1))

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

star_dict = get_star_dict(training_set + validation_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

star_embeddings = get_star_embedding(star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)
## initialize model and optimizer
model, _ = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

t1 = time.time()
try:
    model_file_name = 'mmd_log_model_1000.0_%s_0.001.pt' % args.kernel
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file_name), map_location=device))
except:
    model_file_name = 'mmd_log_model_1000_%s_0.001.pt' % args.kernel
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file_name), map_location=device))
model.eval()

all_stars = []
for g in training_set+validation_set:
    all_stars.extend(list(get_node_star(g).values()))

sampled_set = list()
batch_size = 500
graphs_to_sample = 5000
z_length = 10
for _ in range(1+(graphs_to_sample//len(training_set))):
    with torch.no_grad():
        for i in range(0, min(int(1.5*len(training_set)), 15000), batch_size):
            t2 = time.time()

            z_list = [torch.randn((z_length, 64)).to(device) \
                for _ in range(batch_size)]
            first_star_list = random.sample(all_stars, batch_size)

            gphs, _, _, f_star = batch_sample_graphs_from_z(model, z_list,
                                        first_star_list, 50,
                                		1, star_dict, star_embeddings, 20, False)
            sampled_set.extend(gphs)
            print('sampled %s gphs, %s/%s, time %.1fs' % (len(gphs), len(sampled_set), i+batch_size, time.time()-t2))

with open(os.path.join(model_dir, 'unc_%s_generated.pkl' % (model_file_name.replace('.pt', ''))), 'wb') as f:
    pickle.dump(sampled_set, f)

print('time %.2fm, %s graphs generated' % ((time.time()-t1)/60, len(sampled_set)))
