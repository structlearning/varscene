import pickle
import importlib
import numpy as np
import time
from collections import Counter
import random
import matplotlib.pyplot as plt
import os
import torch
import argparse
import networkx as nx
from utils import batch_sample_graphs_from_z, build_model, get_graph, get_star_embedding, pack_batch, \
    get_label_embeddings, get_star_dict
from grakel.graph_kernels import ShortestPath, NeighborhoodSubgraphPairwiseDistance, WeisfeilerLehman
import grakel.utils
from calc_metrics import graph_structure_worker
from calc_metrics import get_node_bigrams as get_node_bi, get_edge_bigrams as get_edge_bi, make_dist_mmd_opt
from calc_metrics import cos_sim, get_all_stars as get_star_sim, compute_mmd, single_star_worker_mmd_opt, gaussian

t1 = time.time()
## argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, help="directory to store model outputs", required=True)
parser.add_argument("--dataset", type=str, choices=['vg', 'vrd', 'svg'], help="dataset name", required=True)
mmd_choices = ['node_edge', 'star_node_edge', 'sp_node_edge', 'sp', 'wl', 'nspd', 'node_bi', 'edge_bi', 'star_sim', 'star']
parser.add_argument("--mmd", type=str, choices=mmd_choices, help="kernel for mmd", default='star')
parser.add_argument("--load_checkpoint", action='store_true', help="load model from checkpoint")
parser.add_argument('--kl_wt', type=float, help='kl loss weight', default=1000.0)
parser.add_argument("--epochs", type=int, help="number of training epochs", default=1000)
parser.add_argument("--batch_size", type=int, help="number of graphs in batch for training", default=1024)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)

args = parser.parse_args()
out_dir = args.out_dir

if not os.path.isdir(out_dir):
    raise ValueError('%s does not exist, provide valid out_dir' % out_dir)

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

config_file = (os.path.join(out_dir, 'configure')).replace('/','.')
config = importlib.import_module(config_file).get_default_config()

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

t1 = time.time()
## load graphs
if args.dataset == 'vg':
    data_dir = '../vg_data/data'
elif args.dataset == 'vrd':
    data_dir = '../vrd_data/data'
elif args.dataset == 'svg':
    data_dir = '../svg_data/data'

with open(os.path.join(data_dir, 'graphs_train.pkl'), 'rb') as f:
    training_set = pickle.load(f)
with open(os.path.join(data_dir, 'graphs_val.pkl'), 'rb') as f:
    validation_set = pickle.load(f)
with open(os.path.join(data_dir, 'all_stars.pkl'), 'rb') as f:
    all_stars = pickle.load(f)
with open(os.path.join(data_dir, 'all_pair_stars.pkl'), 'rb') as f:
    all_pairs = pickle.load(f)

print('loaded graphs in %.2fs' % (time.time()-t1))

training_set = [nx.convert_node_labels_to_integers(g) for g in training_set]
validation_set = [nx.convert_node_labels_to_integers(g) for g in validation_set]

t1 = time.time()
node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set+validation_set)
training_star_dict = get_star_dict(training_set)
validation_star_dict = get_star_dict(validation_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

training_star_embeddings = get_star_embedding(training_star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)
validation_star_embeddings = get_star_embedding(validation_star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)

## compute ids of stars for MMD
sample_star_for_MMD = 1000
idx2sampled_stars = [star for star, _ in Counter(all_stars).most_common(sample_star_for_MMD)]
sampled_stars2idx = {}
for i, e in enumerate(idx2sampled_stars): sampled_stars2idx[e] = i

star_embeddings = training_star_embeddings.copy()
star_embeddings.update(validation_star_embeddings)
star_dict = training_star_dict.copy()
star_dict.update(validation_star_dict)

print('computed stars and embeddings in %.2fs' % (time.time()-t1))

if args.mmd == 'star':
    mmd_kernel = single_star_worker_mmd_opt
    mmd_kernel_kwargs = dict(sampled_stars2idx=sampled_stars2idx)
elif args.mmd == 'sp' or args.mmd == 'sp_node_edge':
    grkl_kernel = ShortestPath(normalize=False)
elif args.mmd == 'nspd':
    grkl_kernel = NeighborhoodSubgraphPairwiseDistance(normalize=False)
elif args.mmd == 'wl':
    grkl_kernel = WeisfeilerLehman(normalize=False)

def structure_cos_sim(gphs1, gphs2, worker):
    all_strucs = list()
    for g in gphs1+gphs2:
        all_strucs.extend(worker(g))
    all_strucs = list(set(all_strucs))
    idx2strucs = all_strucs
    struc2idx = {}
    for i, e in enumerate(idx2strucs): struc2idx[e] = i
    dist1 = graph_structure_worker(gphs1, struc2idx, worker)
    dist2 = graph_structure_worker(gphs2, struc2idx, worker)
    return cos_sim(dist1, dist2)

## initialize model
model, _ = build_model(config, node_feature_dim, edge_feature_dim)
## set optimizer only for decoder
optimizer = torch.optim.Adam((model._decoder.parameters()),
        lr=args.lr, weight_decay=1e-5)
model.to(device)
base_model, _ = build_model(config, node_feature_dim, edge_feature_dim)
base_model.to(device)

base_model_file = os.path.join(out_dir, ('star_vae_%s.pt' % args.dataset))
base_model.load_state_dict(torch.load(base_model_file, map_location=device))

batch_size = args.batch_size
n_training_steps = args.epochs
kl_weight = args.kl_wt
num_graphs_mmd = 1000
kl_graphs_batch_size = 1024

train_epochs, train_loss_hist = list(), list()
train_mmd_hist = list()
train_mmd_loss_component_hist = list()
train_kl_loss_component_hist = list()
start_epoch = 0

model_file = os.path.join(out_dir, 'mmd_log_model_%s_%s_%s.pt' % (kl_weight, args.mmd, args.lr))
optimizer_file = os.path.join(out_dir, 'mmd_log_model_optimizer_%s_%s_%s.pt' % (kl_weight, args.mmd, args.lr))
loss_hist_file = os.path.join(out_dir, 'mmd_log_loss_hist_%s_%s_%s.pkl' % (kl_weight, args.mmd, args.lr))
if args.load_checkpoint:
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
    if os.path.isfile(optimizer_file):
        optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))
    if os.path.isfile(loss_hist_file):
        with open(loss_hist_file, 'rb') as f:
            (train_epochs, train_loss_hist, train_mmd_hist, train_mmd_loss_component_hist, train_kl_loss_component_hist) = pickle.load(f)
            start_epoch = len(train_epochs)

loss_hist_plots_dir = os.path.join(out_dir, 'mmd_log_model_%s_%s_%s_loss_plots' % (kl_weight, args.mmd, args.lr))
if not os.path.isdir(loss_hist_plots_dir):
    os.mkdir(loss_hist_plots_dir)

print('starting training...')
for i_iter in range(start_epoch, start_epoch+n_training_steps):

    t1 = time.time()
    optimizer.zero_grad()
    loss = 0
    kl_loss_component = 0
    mmd_loss_component = 0
    base_model.eval()
    total_time = 0 

    ## compute kl-div loss over entire training graphs
    kl_graphs_num = i_iter % max(1, (len(training_set)//kl_graphs_batch_size))
    kl_graphs = training_set[kl_graphs_num*kl_graphs_batch_size : (kl_graphs_num+1)*kl_graphs_batch_size]
    # for i_batch in range(0, len(kl_graphs), batch_size):
    for i_batch in range(0, 1):
        train_batch = kl_graphs[i_batch : i_batch + batch_size]

        ## sample z values and first stars from generation graphs
        batch = pack_batch(train_batch,
                        node_label_embeddings, edge_label_embeddings,
                        star_embeddings, star_dict)
        first_star_list = batch.graph_first_star

        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        node_features = node_features.to(device)
        edge_features = edge_features.to(device)
        star_features = star_features.to(device)
        candidate_star_features = candidate_star_features.to(device)
        target_star_idx = target_star_idx.to(device)
        star_z_mask = star_z_mask.to(device)
        candidate_star_z_mask = candidate_star_z_mask.to(device)
        from_idx = from_idx.to(device)
        to_idx = to_idx.to(device)
        graph_idx = graph_idx.to(device)
        edge_graph_idx = edge_graph_idx.to(device)
        graph_depth_range = graph_depth_range.to(device)
        node_graph_depth_idx = node_graph_depth_idx.to(device)

        tg = time.time()
        with torch.no_grad():
            _, z_list = base_model(node_features,
                edge_features, star_features,
                candidate_star_features, target_star_idx,
                star_z_mask, candidate_star_z_mask,
                from_idx, to_idx, graph_idx, 1+int(torch.max(graph_idx)),
                0, graph_depth_range, node_graph_depth_idx, True)
        ## for each graph, obtained z values in `z_list` and first star in `first_star_list`

        log_prob_list = model.log_prob_given_z(z_list, star_features,
                                candidate_star_features, target_star_idx,
                                star_z_mask, candidate_star_z_mask,
                                to_idx, edge_graph_idx, 1+int(torch.max(graph_idx)),
                                graph_depth_range)

        with torch.no_grad():
            base_log_prob_list = base_model.log_prob_given_z(z_list, star_features,
                                        candidate_star_features, target_star_idx,
                                        star_z_mask, candidate_star_z_mask,
                                        to_idx, edge_graph_idx, 1+int(torch.max(graph_idx)),
                                        graph_depth_range)

        loss = (kl_weight/2*torch.sum((log_prob_list-base_log_prob_list)**2)\
            +kl_weight*torch.sum(log_prob_list)) / len(kl_graphs)
        kl_loss_component += loss.item()
        loss.backward()
        total_time += time.time()-tg
    

    ## mmd loss computation
    gen_graphs_num = i_iter % (len(training_set)//num_graphs_mmd)
    generation_graphs = training_set[gen_graphs_num*num_graphs_mmd : (gen_graphs_num+1)*num_graphs_mmd]
    mmd_graphs = random.sample(validation_set, min(len(validation_set), num_graphs_mmd))
    first_star_list, z_list = list(), list()

    ## sample z values and first stars from generation graphs
    for i_batch in range(0, len(generation_graphs), batch_size):
        gen_graph_batch = generation_graphs[i_batch : i_batch+batch_size]

        batch = pack_batch(gen_graph_batch,
                        node_label_embeddings, edge_label_embeddings,
                        star_embeddings, star_dict)
        first_star_list.extend(batch.graph_first_star)

        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, _, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        tg = time.time()
        with torch.no_grad():
            _, batch_z_list = base_model(node_features.to(device),
                edge_features.to(device), star_features.to(device),
                candidate_star_features.to(device), target_star_idx.to(device),
                star_z_mask.to(device), candidate_star_z_mask.to(device),
                from_idx.to(device), to_idx.to(device), graph_idx.to(device), 1+int(torch.max(graph_idx)),
                0, graph_depth_range.to(device), node_graph_depth_idx.to(device), True)
        total_time += time.time()-tg
        z_list.extend(batch_z_list)
    ## for each graph, obtained z values in `z_list` and first star in `first_star_list`

    ## sample graphs conditioned on `z_list`
    t2 = time.time()
    tg = time.time()
    new_graphs, utilized_z_list, _, _ = batch_sample_graphs_from_z(model, z_list, first_star_list,
                                    cutoff_size=20, n_trials=50, n_sampling_epochs=1,
                                    verbose=False, star_dict=star_dict, star_embeddings=star_embeddings)
    print('sampled %s/%s graphs in %.2fs' % (len(new_graphs), len(generation_graphs), time.time()-t2))
    total_time += time.time()-tg

    ## compute MMD
    t2 = time.time()
    if args.mmd == 'star':
        dist_1 = make_dist_mmd_opt(mmd_graphs, mmd_kernel, **mmd_kernel_kwargs)
        dist_2 = make_dist_mmd_opt(new_graphs, mmd_kernel, **mmd_kernel_kwargs)
        star_mmd = compute_mmd(dist_1, dist_2, kernel=gaussian, sigma=1)
    elif args.mmd in ['star_sim', 'node_bi', 'edge_bi']:
        ## take negative since star_mmd is minimized
        star_mmd = -structure_cos_sim(new_graphs, mmd_graphs, eval('get_%s' % args.mmd))
    elif args.mmd in ['sp', 'wl', 'nspd']:
        ref_gphs = grakel.utils.graph_from_networkx(mmd_graphs, 'label', 'label')
        pred_gphs = grakel.utils.graph_from_networkx([nx.Graph(o) for o in new_graphs], 'label', 'label')
        grkl_kernel.fit_transform(ref_gphs)
        K_pred = grkl_kernel.transform(pred_gphs)
        K_pred = np.nan_to_num(K_pred)
        star_mmd = -np.mean(K_pred) ## take negative since star_mmd is minimized
    elif 'node_edge' in args.mmd:
        ## take negative since star_mmd is minimized
        star_mmd = -structure_cos_sim(new_graphs, mmd_graphs, get_node_bi)
        star_mmd += -structure_cos_sim(new_graphs, mmd_graphs, get_edge_bi)
        if args.mmd == 'star_node_edge':
            star_mmd += -structure_cos_sim(new_graphs, mmd_graphs, get_star_sim)
        elif args.mmd == 'sp_node_edge':
            ref_gphs = grakel.utils.graph_from_networkx(mmd_graphs, 'label', 'label')
            pred_gphs = grakel.utils.graph_from_networkx([nx.Graph(o) for o in new_graphs], 'label', 'label')
            grkl_kernel.fit_transform(ref_gphs)
            K_pred = grkl_kernel.transform(pred_gphs)
            K_pred = np.nan_to_num(K_pred)
            star_mmd += -np.mean(K_pred) ## take negative since star_mmd is minimized
        
    print('%s mmd %.6f computed in %.2fs' % (args.mmd, star_mmd, time.time()-t2))

    ## probability of `new_graphs` given their z-representations `utilized_z_list`
    for i_batch in range(0, len(new_graphs), batch_size):
        new_graphs_batch = new_graphs[i_batch : i_batch + batch_size]

        batch = pack_batch(new_graphs_batch,
                    node_label_embeddings, edge_label_embeddings,
                    star_embeddings, star_dict)
        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        tg = time.time()
        log_prob_list = model.log_prob_given_z(
                                [o.to(device) for o in utilized_z_list], star_features.to(device),
                                candidate_star_features.to(device), target_star_idx.to(device),
                                star_z_mask.to(device), candidate_star_z_mask.to(device),
                                to_idx.to(device), edge_graph_idx.to(device), 1+int(torch.max(graph_idx)),
                                graph_depth_range.to(device))

        loss = torch.sum(log_prob_list)*star_mmd
        mmd_loss_component += loss.item()
        loss.backward()
        total_time += time.time()-tg

    # tg = time.time()
    optimizer.step()
    # total_time += time.time()-tg
    
    train_loss_hist.append(mmd_loss_component+kl_loss_component)
    train_mmd_hist.append(star_mmd)
    train_mmd_loss_component_hist.append(mmd_loss_component)
    train_kl_loss_component_hist.append(kl_loss_component)
    train_epochs.append(i_iter+1)

    ## save model
    if i_iter % config['training']['save_model_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        with open(os.path.join(out_dir, 'mmd_log_generated_%s_%s.pkl' % (kl_weight, args.mmd)), 'wb') as f:
            pickle.dump(new_graphs, f)

    ## plot loss history
    if i_iter % config['training']['save_loss_hist_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
        with open(loss_hist_file, 'wb') as f:
            pickle.dump((train_epochs, train_loss_hist, train_mmd_hist, train_mmd_loss_component_hist, train_kl_loss_component_hist), f)
        
        for v in ['loss', 'mmd', 'mmd_loss_component', 'kl_loss_component']:
            plt.figure()
            plt.plot(train_epochs, eval('train_%s_hist' % v), label='train %s' % v)
            plt.legend()
            plt.title('MMD Opt. %s hist, %s dataset, %s kl_wt, %s mmd, %s lr' % (v, args.dataset, kl_weight, args.mmd, args.lr))
            plt.xlabel('epoch')
            plt.ylabel(v)
            plt.savefig(os.path.join(loss_hist_plots_dir, '%s_hist.png' % v), bbox_inches='tight')
            plt.close()

    # print('%.4fs per batch' % total_time)

    print('epoch %s, loss %.4f, %s_mmd %.6f, mmd_loss %.4f, kl_loss %.4f, time %.2fs' %\
        (i_iter+1, mmd_loss_component+kl_loss_component, args.mmd, star_mmd, mmd_loss_component, kl_loss_component, time.time()-t1))

