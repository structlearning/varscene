import os
import argparse
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from models import Generator, Discriminator
import networkx as nx

import sys
sys.path.append('./data')

### PARAMETERS ###
n_sample = 10000
sample_from_iter_no = 200000
small_architecture = False
### PARAMETERS ###

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, choices=['vg', 'vrd', 'svg'])

# Model configuration.
if small_architecture:
    parser.add_argument('--z_dim', type=int, default=2, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[8,16,32], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[8, 2], 8, [8, 2]], help='number of conv filters in the first layer of D')
else:
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128,256,512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])
parser.add_argument('--vertexes', type=int, default=50, help='dimension of domain labels')

# Training configuration.
if small_architecture:
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
else:
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# Miscellaneous.
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--use_tensorboard', type=str2bool, default=False)

# Step size.
parser.add_argument('--log_step', type=int, default=200000)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=200000)
parser.add_argument('--lr_update_step', type=int, default=1000)

config = parser.parse_args()
print(config)

edge_idx2lbl = pickle.load(open('data/'+config.dataset+'/edge_idx2lbl.pkl', 'rb'))
edge_lbl2idx = pickle.load(open('data/'+config.dataset+'/edge_lbl2idx.pkl', 'rb'))
node_lbl2idx = pickle.load(open('data/'+config.dataset+'/node_lbl2idx.pkl', 'rb'))
node_idx2lbl = pickle.load(open('data/'+config.dataset+'/node_idx2lbl.pkl', 'rb'))

config.m_dim = len(node_lbl2idx)
config.b_dim = len(edge_lbl2idx)
config.atom_num_types = len(node_lbl2idx)
config.bond_num_types = len(edge_lbl2idx)

config.mol_data_dir = 'data/' + config.dataset + '/SG.pkl'
config.model_save_dir = 'models/' + config.dataset

G = Generator(config.g_conv_dim, config.z_dim, config.vertexes, config.bond_num_types+1, config.atom_num_types+1, config.dropout)
D = Discriminator(config.d_conv_dim, config.m_dim+1, config.b_dim+1, config.dropout)
V = Discriminator(config.d_conv_dim, config.m_dim+1, config.b_dim+1, config.dropout)

print('Models initiated...')

G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(sample_from_iter_no))
D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(sample_from_iter_no))
V_path = os.path.join(config.model_save_dir, '{}-V.ckpt'.format(sample_from_iter_no))
G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

print('Models restored...')

def sample_z(batch_size): return np.random.normal(0, 1, size=(batch_size, config.z_dim))

def postprocess(inputs, method, temperature=1.):
    def listify(x): return x if type(x) == list or type(x) == tuple else [x]
    def delistify(x): return x if len(x) > 1 else x[0]
    if method == 'soft_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                   / temperature, hard=False).view(e_logits.size())
                   for e_logits in listify(inputs)]
    elif method == 'hard_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                   / temperature, hard=True).view(e_logits.size())
                   for e_logits in listify(inputs)]
    else:
        softmax = [F.softmax(e_logits / temperature, -1)
                   for e_logits in listify(inputs)]
    return [delistify(e) for e in (softmax)]

graphs = []
ii = 0
while ii<n_sample:
    try:
        z = sample_z(2)
        z = torch.from_numpy(z).float()
        edges_logits, nodes_logits = G(z)
        (edges_hard, nodes_hard) = postprocess((edges_logits, nodes_logits), 'hard_gumbel')
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1].numpy(), torch.max(nodes_hard, -1)[1].numpy()
        nodes = [node_idx2lbl[n] for n in nodes_hard[0]]
        edges = [[edge_idx2lbl[n] for n in m] for m in edges_hard[0]]
        Gr = nx.Graph()
        for i, n in enumerate(nodes): Gr.add_nodes_from([(i, {'label' : nodes[i]})])
        for i in range(config.vertexes):
            for j in range(i+1, config.vertexes):
                if edges[i][j]!='NO':
                    Gr.add_edge(i, j, label=edges[i][j])
        graphs.append(Gr)
        ii+=1
    except:
        pass

with open('data/'+config.dataset+'/molgan_generated_graphs.pkl', 'wb') as f: pickle.dump(graphs, f)
