import pickle
import os
import numpy as np

def runner(dataset):
    if dataset == 'vrd':
        data_dir = '../vrd_data/data'
    elif dataset == 'vg':
        data_dir = '../vg_data/data'
    elif dataset == 'svg':
        data_dir = '../svg_data/data'
    with open(os.path.join(data_dir, 'graphs_test.pkl'), 'rb') as f: graphs_test = pickle.load(f)
    with open(os.path.join(data_dir, 'graphs_train.pkl'), 'rb') as f: graphs_train = pickle.load(f)
    with open(os.path.join(data_dir, 'graphs_val.pkl'), 'rb') as f: graphs_val = pickle.load(f)
    graphs = graphs_train+graphs_val+graphs_test

    avg_nodes = np.mean([o.number_of_nodes() for o in graphs])
    avg_edges = np.mean([o.number_of_edges() for o in graphs])
    node_labels, edge_labels = set(), set()
    for g in graphs:
        for _, l in g.nodes(data='label'):
            node_labels.add(l)
        for _, _, l in g.edges(data='label'):
            edge_labels.add(l)

    D = len(graphs_train+graphs_test+graphs_val)
    print('%s & %s & %d:%d:%d & %.2f & %.2f & %s & %s \\\\' % (dataset, D, 100*len(graphs_train)/D, \
        100*len(graphs_val)/D, 100*len(graphs_test)/D, avg_nodes, avg_edges, len(node_labels), len(edge_labels)))

runner('vg')
runner('svg')
runner('vrd')