import networkx as nx
import pickle
import numpy as np
import os

data_path = '../vrd_data/data'
out_path = 'iccv_format_graphs/vrd_dir'

with open(os.path.join(data_path, 'graphs_train_dir.pkl'), 'rb') as f:
    graphs_train = pickle.load(f)
with open(os.path.join(data_path, 'graphs_val_dir.pkl'), 'rb') as f:
    graphs_val = pickle.load(f)
with open(os.path.join(data_path, 'graphs_test_dir.pkl'), 'rb') as f:
    graphs_test = pickle.load(f)

graphs_train = [nx.convert_node_labels_to_integers(g) for g in graphs_train]
graphs_val = [nx.convert_node_labels_to_integers(g) for g in graphs_val]
graphs_test = [nx.convert_node_labels_to_integers(g) for g in graphs_test]

node_labels = set()
edge_labels = set()

for g in graphs_train+graphs_test+graphs_val:
    for n, l in g.nodes(data='label'):
        node_labels.add(l)
    for u, v, l in g.edges(data='label'):
        edge_labels.add(l)

node_labels = list(node_labels)
edge_labels = list(edge_labels)
node_labels2idx = dict()
for i, l in enumerate(node_labels):
    node_labels2idx[l] = i+1
edge_labels2idx = dict()
for i, l in enumerate(edge_labels):
    edge_labels2idx[l] = i+1

def make_dataset(gphs):
    out_list = list()
    for g in gphs:
        n = g.number_of_nodes()
        node_ids = np.zeros(n)
        edge_ids = np.zeros((n, n))
        for i in range(n):
            node_ids[i] = node_labels2idx[g.nodes[i]['label']]
        for u, v, label in g.edges(data='label'):
            edge_ids[u, v] = edge_labels2idx[label]
        out_list.append((node_ids, edge_ids))
    return out_list

train_dataset = make_dataset(graphs_train+graphs_val)
test_dataset = make_dataset(graphs_test)

with open(os.path.join(out_path, 'train_dataset.p'), 'wb') as f:
    pickle.dump(train_dataset, f)
with open(os.path.join(out_path, 'test_dataset.p'), 'wb') as f:
    pickle.dump(test_dataset, f)
with open(os.path.join(out_path, 'categories.p'), 'wb') as f:
    pickle.dump([node_labels, edge_labels, None], f)

print('%s train graphs, %s test graphs' % (len(train_dataset), len(test_dataset)))
print('%s node labels, %s edge labels' % (len(node_labels), len(edge_labels)))
