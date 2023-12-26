from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import networkx as nx
import os

### parameters
from args import args

def create_nx_graphs(dataset, node_label, edge_label):
    gphs = list()
    for n, e in dataset:
        g = nx.Graph()
        for i, n_label_id in enumerate(n):
            g.add_node(i, label=node_label[n_label_id-1])
        for i in range(n.shape[0]):
            for j in range(n.shape[0]):
                if e[i, j] != 0:
                    g.add_edge(i, j, label=edge_label[int(e[i, j])-1])
        connected_comps = [g.subgraph(c).copy() for c in nx.connected_components(g) if len(c)>1]
        gphs.extend(connected_comps)
    return gphs

with open(os.path.join(args['in_data_path'], 'train_dataset.p'), 'rb') as f:
    train_dataset = pickle.load(f)
with open(os.path.join(args['in_data_path'], 'test_dataset.p'), 'rb') as f:
    test_dataset = pickle.load(f)
with open(os.path.join(args['in_data_path'], 'categories.p'), 'rb') as f:
    categories = pickle.load(f)

node_label, edge_label, label = categories

graphs_train, graphs_val, graphs_test = list(), list(), list()
graphs_train = create_nx_graphs(train_dataset, node_label, edge_label)
graphs_test = create_nx_graphs(test_dataset, node_label, edge_label)
graphs_train, graphs_val = train_test_split(graphs_train, test_size=args['val_size'], random_state=42)

print(len(graphs_train), len(graphs_val), len(graphs_test))

### Init all_stars which is an array of start stars from all graphs
### Init idx2stars that gives an array of next stars from all graphs given a node label and an edge label
def get_all_stars(G):
    stars = []
    nodes = G.nodes
    for node in nodes():
        edges = []
        if G.edges(node):
            for a, b in G.edges(node):
                edges.append(G[a][b]['label'])
            stars.append(nodes[node]['label']+'$%'+'$%'.join(sorted(edges)))
    return stars

def get_star(G, node, label):
    edges = []
    for a, b in G.edges(node):
        edges.append(G[a][b]['label'])
    edges.remove(label)
    return '$%'.join(sorted(edges))

def get_paired_stars(G):
    pairs = []
    for a, b, l in G.edges(data=True):
        pairs.append(G.nodes[a]['label']+'$%'+get_star(G, a, l['label'])+'$$$'+l['label']+'$$$'+G.nodes[b]['label']+'$%'+get_star(G, b, l['label']))
        pairs.append(G.nodes[b]['label']+'$%'+get_star(G, b, l['label'])+'$$$'+l['label']+'$$$'+G.nodes[a]['label']+'$%'+get_star(G, a, l['label']))
    return pairs

all_stars = []
idx2stars = defaultdict(list)
for G in graphs_train:
    all_stars.extend(get_all_stars(G))
    node_labels = G.nodes.data()
    for u, v, l in G.edges(data=True):
        idx2stars[node_labels[v]['label']+'$%'+l['label']].append(node_labels[u]['label'] + '$%' + get_star(G, u, l['label']))
        idx2stars[node_labels[u]['label']+'$%'+l['label']].append(node_labels[v]['label'] + '$%' + get_star(G, v, l['label']))

all_pairs = []
for G in graphs_train: all_pairs.extend(get_paired_stars(G))

### Storing datas needed for this method
with open(os.path.join(args['out_data_path'], 'all_stars.pkl'), 'wb') as f: pickle.dump(all_stars, f)
with open(os.path.join(args['out_data_path'], 'all_pair_stars.pkl'), 'wb') as f: pickle.dump(all_pairs, f)
with open(os.path.join(args['out_data_path'], 'idx2stars.pkl'), 'wb') as f: pickle.dump(idx2stars, f)
with open(os.path.join(args['out_data_path'], 'graphs_train.pkl'), 'wb') as f: pickle.dump(graphs_train, f)
with open(os.path.join(args['out_data_path'], 'graphs_test.pkl'), 'wb') as f: pickle.dump(graphs_test, f)
with open(os.path.join(args['out_data_path'], 'graphs_val.pkl'), 'wb') as f: pickle.dump(graphs_val, f)
