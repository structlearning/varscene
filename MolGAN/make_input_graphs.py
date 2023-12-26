import numpy as np
import os
import networkx as nx
import pickle
import argparse

### PARAMETERS ###
min_graph_size = 2 ## min size of a graph to be considered in the dataset
max_graph_size = 50 ## max size of a graph to be considered in the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['vg', 'vrd', 'svg'])
args = parser.parse_args()

with open('../%s_data/data/graphs_train.pkl' % args.dataset, 'rb') as f: all_graphs = pickle.load(f)

graphs = []
for G in all_graphs:
    if len(G)>=min_graph_size and len(G)<=max_graph_size: graphs.append(G)

del all_graphs

node_labels = set()
edge_labels = set()
for G in graphs:
    node_labels.update(list(nx.get_node_attributes(G,'label').values()))
    edge_labels.update(list(nx.get_edge_attributes(G,'label').values()))

print(len(node_labels), len(edge_labels), len(graphs))

node_idx2lbl, edge_idx2lbl = ['NO']+sorted(list(node_labels)), ['NO']+sorted(list(edge_labels))
node_lbl2idx, edge_lbl2idx = {}, {}
for i, l in enumerate(node_idx2lbl): node_lbl2idx[l] = i
for i, l in enumerate(edge_idx2lbl): edge_lbl2idx[l] = i

def make_A_X(G, s=max_graph_size):
    nodes = list(G.nodes())
    node2idx = {}
    for i, l in enumerate(nodes): node2idx[l] = i
    A = np.zeros((s, s))
    X = np.zeros(s)
    for n, d in G.nodes(data=True):
        X[node2idx[n]] = node_lbl2idx[d['label']]
    for n1, n2, d in G.edges(data=True):
        A[node2idx[n1]][node2idx[n2]] = edge_lbl2idx[d['label']]
        A[node2idx[n2]][node2idx[n1]] = edge_lbl2idx[d['label']]
    return A.astype(np.int32), X.astype(np.int32)

A, X = [], []
for G in graphs:
    a, x = make_A_X(G)
    A.append(a)
    X.append(x)
A = np.array(A, dtype=np.int32)
X = np.array(X, dtype=np.int32)

folder = args.dataset
if not os.path.exists(folder):
    os.makedirs(folder)
with open('data/'+folder+'/SG.pkl', 'wb') as f:
    pickle.dump((A, X, max_graph_size, len(edge_idx2lbl), len(node_idx2lbl)), f)
with open('data/'+folder+'/edge_idx2lbl.pkl', 'wb') as f:
    pickle.dump(edge_idx2lbl, f)
with open('data/'+folder+'/edge_lbl2idx.pkl', 'wb') as f:
    pickle.dump(edge_lbl2idx, f)
with open('data/'+folder+'/node_lbl2idx.pkl', 'wb') as f:
    pickle.dump(node_lbl2idx, f)
with open('data/'+folder+'/node_idx2lbl.pkl', 'wb') as f:
    pickle.dump(node_idx2lbl, f)

