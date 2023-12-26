from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import networkx as nx
import os

### parameters
from args import args

def create_nx_dir_graphs(dataset, node_label, edge_label):
    gphs = list()
    for n, e in dataset:
        g = nx.DiGraph()
        undir_g = nx.Graph()
        for i, n_label_id in enumerate(n):
            g.add_node(i, label=node_label[n_label_id-1])
            undir_g.add_node(i, label=node_label[n_label_id-1])
        for i in range(n.shape[0]):
            for j in range(n.shape[0]):
                if e[i, j] != 0:
                    g.add_edge(i, j, label=edge_label[int(e[i, j])-1])
                    undir_g.add_edge(i, j, label=edge_label[int(e[i, j])-1])
        connected_comps = [g.subgraph(c).copy() for c in nx.connected_components(undir_g) if len(c)>1]
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
graphs_train = create_nx_dir_graphs(train_dataset, node_label, edge_label)
graphs_test = create_nx_dir_graphs(test_dataset, node_label, edge_label)
graphs_train, graphs_val = train_test_split(graphs_train, test_size=args['val_size'], random_state=42)

print(len(graphs_train), len(graphs_val), len(graphs_test))

### Storing datas needed for this method
with open(os.path.join(args['out_data_path'], 'dir_graphs_train.pkl'), 'wb') as f: pickle.dump(graphs_train, f)
with open(os.path.join(args['out_data_path'], 'dir_graphs_test.pkl'), 'wb') as f: pickle.dump(graphs_test, f)
with open(os.path.join(args['out_data_path'], 'dir_graphs_val.pkl'), 'wb') as f: pickle.dump(graphs_val, f)
