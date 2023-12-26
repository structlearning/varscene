import pickle
import argparse
import numpy as np
import os
import networkx as nx
from uncond_sggen.parser import parse_eval_args

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def create_nx_graphs(dataset, node_label, edge_label):
    gphs = list()
    directed_gphs_count = 0
    for n, e in dataset:
        g = nx.MultiGraph()
        if not check_symmetric(e):
            directed_gphs_count += 1
        for i, n_label_id in enumerate(n):
            if n_label_id > len(node_label):
                print(n)
            g.add_node(i, label=node_label[n_label_id-1])
        for i in range(n.shape[0]):
            for j in range(i+1, n.shape[0]):
                if e[i, j] != 0 and e[i, j] <= len(edge_label):
                    g.add_edge(i, j, label=edge_label[int(e[i, j])-1])
        gphs.append(g)
    print('%s/%s asymmetric adj matrices' % (directed_gphs_count, len(dataset)))
    return gphs

def create_directed_nx_graphs(dataset, node_label, edge_label):
    gphs = list()
    directed_gphs_count = 0
    for n, e in dataset:
        g = nx.DiGraph()
        if not check_symmetric(e):
            directed_gphs_count += 1
        for i, n_label_id in enumerate(n):
            if n_label_id > len(node_label):
                print(n)
            g.add_node(i, label=node_label[n_label_id-1])
        for i in range(n.shape[0]):
            for j in range(n.shape[0]):
                if e[i, j] != 0 and e[i, j] <= len(edge_label):
                    g.add_edge(i, j, label=edge_label[int(e[i, j])-1])
        gphs.append(g)
    print('%s/%s asymmetric adj matrices' % (directed_gphs_count, len(dataset)))
    return gphs

parser = argparse.ArgumentParser()
parser.add_argument('--directed', action='store_true')
parser.add_argument('--dataset', required=True, choices=['vg', 'vrd', 'svg'])
parser.add_argument('--eval_config', required=True)
args = parser.parse_args()
eval_args = parse_eval_args(args.eval_config)

with open(os.path.join(eval_args.eval_path, 'generated_sg.p'), 'rb') as f:
    graphs = pickle.load(f)
with open('../%s_data/data/categories.p' % args.dataset, 'rb') as f:
    categories = pickle.load(f)

node_label, edge_label, label = categories

if args.directed:
    graphs = create_directed_nx_graphs(graphs, node_label, edge_label)
else:
    graphs = create_nx_graphs(graphs, node_label, edge_label)

with open(os.path.join(eval_args.eval_path, 'scenegen_generated.pkl'), 'wb') as f:
    pickle.dump(graphs, f)

